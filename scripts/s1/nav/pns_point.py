#!/usr/bin/env python
"""点位录制与标签导航（S1 底盘）。

子命令:
  record LABEL [--note TEXT] [--force]   把当前位姿存成标签 LABEL
  go LABEL [--speed V] [--omni] [--dry-run] [--timeout S] [--ignore-map]
                                         直接导航到该标签
  list                                    列出已录点位
  rm LABEL                                删除标签

可选参数:
  --file PATH      点位文件，默认 ~/holy/data/s1_nav_points.json
  --note TEXT      record 时写备注
  --force          record 时覆盖同名标签
  --speed V        本任务速度上限 m/s（x/y 同值，角速度取 1.5V，默认 0.2）
  --omni           提高全向规划优先级 omni_plan=True（S1 车尾为雷达盲区，倒车慎用）
  --dry-run        只做可达性查询，不下发运动
  --timeout S      单次任务超时，默认 120 s
  --ignore-map     点位录制时的地图与当前地图不一致也强制执行

用法:
  cd ~/holy/scripts/s1/nav
  python pns_point.py record home --note "起点"
  python pns_point.py list
  python pns_point.py go home --speed 0.15
  python pns_point.py go home --dry-run
  python pns_point.py rm home

退出码: 0 成功 / 2 初始化或定位失败 / 3 不可达未下发 / 4 未到位 /
        5 控制器切换失败 / 6 标签不存在或重名 / 7 点位文件损坏 / 8 地图不一致

说明:
  record 只读取位姿，不切控制器、不下发运动，可在任意已定位时刻执行。
  go 走官方 navigate_to_goal_v2，绝对目标、pose_frame='map'，终点朝向取录制时的 yaw。
  点位按地图标识隔离：换图后旧点位会被判为地图不一致而拒绝执行。
  结束统一走 stop_navigation + 优雅三连 + os._exit（S1 退出段 segfault 教训）。
"""
import faulthandler
import json
import math
import os
import sys
import time

import numpy as np

SDK_LIB = "/userdata/update/manual_update/lib"
DEFAULT_FILE = os.path.expanduser("~/holy/data/s1_nav_points.json")
MAP_CUR = "/var/maps/cur"

faulthandler.dump_traceback_later(300, exit=True)

robot = None
nav = None


# ---------------------------------------------------------------- 点位文件

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_db(path):
    if not os.path.isfile(path):
        return {"schema": 1, "map": {}, "points": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception as e:
        print("[file] 读取失败 %s：%s" % (path, e))
        return None
    if not isinstance(db, dict):
        print("[file] 格式异常（顶层不是对象）：%s" % path)
        return None
    db.setdefault("schema", 1)
    db.setdefault("map", {})
    db.setdefault("points", {})
    if not isinstance(db["points"], dict):
        print("[file] points 字段异常：%s" % path)
        return None
    return db


def save_db(path, db):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    db["updated"] = now_iso()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def current_map():
    """当前使用的地图标识：cur 软链真实目录名 + cleaned 点云的 mtime/size。"""
    cur = os.path.realpath(MAP_CUR)
    info = {"name": os.path.basename(cur), "dir": cur, "pcd_mtime": 0, "pcd_size": 0}
    pcd = os.path.join(cur, "global_cloud_cleaned.pcd")
    try:
        st = os.stat(pcd)
        info["pcd_mtime"] = int(st.st_mtime)
        info["pcd_size"] = int(st.st_size)
    except OSError:
        pass
    return info


def map_mismatch(saved, cur):
    """返回 None 表示一致，否则返回差异描述。"""
    if not saved or not cur:
        return None
    if saved.get("name") and cur.get("name") and saved["name"] != cur["name"]:
        return "地图目录 %s → %s" % (saved["name"], cur["name"])
    if saved.get("pcd_mtime") and cur.get("pcd_mtime") \
            and saved["pcd_mtime"] != cur["pcd_mtime"]:
        return "点云 mtime %d → %d" % (saved["pcd_mtime"], cur["pcd_mtime"])
    return None


# ---------------------------------------------------------------- 几何与 SDK

def yaw_of(pose):
    qx, qy, qz, qw = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def pose_str(p):
    return "x=%+.3f y=%+.3f yaw=%+.2f°" % (
        float(p[0]), float(p[1]), math.degrees(yaw_of(p)))


def quat_from_yaw(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def sdk():
    if os.path.isdir(SDK_LIB) and SDK_LIB not in sys.path:
        sys.path.insert(0, SDK_LIB)
    import galbot_sdk.s1 as m
    return m


def bring_up(need_ctrl):
    """初始化 SDK 并等待定位。need_ctrl=True 时按官方顺序切 POSE 控制器。

    会话复用（s1_nav_worker 常驻模式，2026-11-12）：robot/nav 已在时跳过
    SDK 重建（init+1.5s 等待），只做每腿必做的切换+定位确认——省的是进程
    冷启动，安全闸（切换/is_localized）一项不省。CLI 首次调用 robot=None
    走原路径，行为逐位不变。
    """
    global robot, nav
    m = sdk()
    if robot is None or nav is None:
        robot = m.GalbotRobot()
        if not robot.init():
            print("[init] robot.init 失败")
            return None, 2
        nav = m.GalbotNavigation()
        if not nav.init():
            print("[init] nav.init 失败")
            return None, 2
        time.sleep(1.5)

    if need_ctrl:
        st = robot.switch_controller(m.S1ControllerName.SWERVE_CHASSIS_POSE_CTRL)
        print("[ctrl] switch_controller(SWERVE_CHASSIS_POSE_CTRL) →", st)
        if st != m.ControlStatus.SUCCESS:
            print("[ctrl] 控制器切换失败，中止")
            return None, 5

    ok = False
    for _ in range(20):
        try:
            if nav.is_localized():
                ok = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    print("[loc] is_localized →", ok)
    if not ok:
        print("[loc] 未定位，中止（先在应用侧或 tools 侧完成初始定位）")
        return None, 2
    return m, 0


# ---------------------------------------------------------------- 子命令

def do_list(opts):
    db = load_db(opts["file"])
    if db is None:
        return 7
    pts = db["points"]
    mp = db.get("map") or {}
    print("[file] %s" % opts["file"])
    print("[map]  %s" % (mp.get("name") or "（未记录）"))
    if not pts:
        print("[list] 暂无点位")
        return 0
    print("[list] 共 %d 个点位" % len(pts))
    print("%-18s %9s %9s %8s  %-19s %s"
          % ("标签", "x", "y", "yaw°", "创建时间", "备注"))
    for label in sorted(pts):
        p = pts[label]
        print("%-18s %+9.3f %+9.3f %+8.2f  %-19s %s"
              % (label, float(p.get("x", 0.0)), float(p.get("y", 0.0)),
                 float(p.get("yaw_deg", 0.0)),
                 str(p.get("created", "")).replace("T", " "),
                 p.get("note", "")))
    return 0


def do_rm(opts, labels):
    db = load_db(opts["file"])
    if db is None:
        return 7
    rc = 0
    for label in labels:
        p = db["points"].pop(label, None)
        if p is None:
            print("[rm] 标签不存在：%s" % label)
            rc = 6
        else:
            print("[rm] 已删除 %s（原 x=%+.3f y=%+.3f yaw=%+.2f°）"
                  % (label, float(p.get("x", 0.0)), float(p.get("y", 0.0)),
                     float(p.get("yaw_deg", 0.0))))
    save_db(opts["file"], db)
    print("[file] %s（剩余 %d 个点位）" % (opts["file"], len(db["points"])))
    return rc


def do_record(opts, label):
    db = load_db(opts["file"])
    if db is None:
        return 7
    old = db["points"].get(label)
    if old is not None and not opts["force"]:
        print("[record] 标签已存在：%s（x=%+.3f y=%+.3f yaw=%+.2f°），"
              "如需覆盖加 --force"
              % (label, float(old.get("x", 0.0)), float(old.get("y", 0.0)),
                 float(old.get("yaw_deg", 0.0))))
        return 6

    m, rc = bring_up(need_ctrl=False)
    if m is None:
        return rc

    p = nav.get_current_pose()
    yaw = yaw_of(p)
    mp = current_map()
    rec = {
        "x": round(float(p[0]), 4),
        "y": round(float(p[1]), 4),
        "z": 0.0,
        "yaw_deg": round(math.degrees(yaw), 3),
        "created": now_iso(),
        "note": opts["note"],
        "map": mp,
    }
    db["points"][label] = rec
    db["map"] = mp
    save_db(opts["file"], db)

    print("[point] 已记录 %s  %s" % (label, pose_str(p)))
    if opts["note"]:
        print("[point] 备注：%s" % opts["note"])
    print("[map]  %s（%s）" % (mp["name"], mp["dir"]))
    print("[file] %s（共 %d 个点位）" % (opts["file"], len(db["points"])))
    if old is not None:
        print("[record] 已覆盖旧值 x=%+.3f y=%+.3f yaw=%+.2f°"
              % (float(old.get("x", 0.0)), float(old.get("y", 0.0)),
                 float(old.get("yaw_deg", 0.0))))
    return 0


def do_go(opts, label):
    db = load_db(opts["file"])
    if db is None:
        return 7
    pt = db["points"].get(label)
    if pt is None:
        print("[go] 标签不存在：%s（可用 list 查看已录点位）" % label)
        return 6

    cur = current_map()
    diff = map_mismatch(pt.get("map") or {}, cur)
    if diff:
        print("[map] 警告：点位录制时的地图与当前不一致（%s）" % diff)
        print("[map] 当前 %s" % cur["name"])
        if not opts["ignore_map"]:
            print("[map] 目标坐标可能已失效，确认无碍后加 --ignore-map 强制执行")
            return 8

    gx = float(pt["x"])
    gy = float(pt["y"])
    gyaw = math.radians(float(pt.get("yaw_deg", 0.0)))
    print("[point] 目标 %s  x=%+.3f y=%+.3f yaw=%+.2f°%s"
          % (label, gx, gy, math.degrees(gyaw),
             ("  备注：" + pt["note"]) if pt.get("note") else ""))

    m, rc = bring_up(need_ctrl=True)
    if m is None:
        return rc

    p0 = nav.get_current_pose()
    print("[pose] 起点 " + pose_str(p0))
    dist = math.hypot(gx - float(p0[0]), gy - float(p0[1]))
    print("[goal] 直线距离 %.3f m" % dist)

    goal = np.array([gx, gy, 0.0] + quat_from_yaw(gyaw), dtype=np.float64)
    try:
        reach = bool(nav.check_path_reachability(goal, p0))
    except Exception as e:
        reach = None
        print("[reach] 查询异常：%s" % e)
    print("[reach] check_path_reachability →", reach)

    if opts["dry"]:
        print("[dry-run] 不下发运动")
        return 0
    if reach is False:
        print("[reach] pns 判不可达，不下发")
        return 3

    speed = opts["speed"]
    max_vel = np.array([speed, speed, min(0.5, speed * 1.5)], dtype=np.float64)
    timeout = opts["timeout"]
    print("[nav] 下发 pose_frame=map omni_plan=%s max_vel=%s timeout=%.0fs"
          % (opts["omni"], max_vel.round(3).tolist(), timeout))

    t0 = time.time()
    res = nav.navigate_to_goal_v2(goal, max_vel, pose_frame="map",
                                  enable_collision_check=True, is_blocking=False,
                                  timeout=timeout, omni_plan=opts["omni"])
    print("[nav] 返回", res)

    # 非阻塞模式下 check_goal_arrival 不随任务完成刷新，主判据=「平台真停 +
    # 目标残差 + 终点航向」三件。11-12 修：原地转动也算进展——pns 会自选
    # rotate→drive→align 三段路径（13:34 实录：旧判据把掉头段当「位移平台」
    # 2.6s 掐腿 rc=4；把到站航向对准段当「到位」rc=0 放行，方向差 106/130°，
    # 且 finally stop_navigation 把对准腰斩）。转动阈值 0.4°/0.4s 采样≈1°/s；
    # 成功航向容差 10°
    still = 0.0
    best = 0.0
    stopped = False
    arrived_sdk = False
    last_print = 0.0
    prev_yaw = None
    while time.time() - t0 < timeout:
        time.sleep(0.4)
        try:
            p = nav.get_current_pose()
        except Exception:
            continue
        el = time.time() - t0
        travelled = math.hypot(float(p[0]) - float(p0[0]), float(p[1]) - float(p0[1]))
        yaw = yaw_of(p)
        turning = prev_yaw is not None and abs(math.degrees(math.atan2(
            math.sin(yaw - prev_yaw), math.cos(yaw - prev_yaw)))) > 0.4
        prev_yaw = yaw
        if travelled > best + 0.004 or turning:
            best = max(best, travelled)
            still = 0.0
        else:
            still += 0.4
        if el - last_print >= 1.0:
            last_print = el
            print("  [%5.1fs] 已走 %.3f m  剩 %.3f m  %s"
                  % (el, travelled,
                     math.hypot(float(p[0]) - gx, float(p[1]) - gy), pose_str(p)),
                  flush=True)
        try:
            if nav.check_goal_arrival():
                arrived_sdk = True
                break
        except Exception:
            pass
        if still >= 1.6 and el > 1.0:
            stopped = True
            break

    el = time.time() - t0
    try:
        p1 = nav.get_current_pose()
    except Exception:
        p1 = None
    if p1 is not None:
        net = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
        resid = math.hypot(float(p1[0]) - gx, float(p1[1]) - gy)
        yaw_err = abs(math.degrees(math.atan2(
            math.sin(yaw_of(p1) - gyaw), math.cos(yaw_of(p1) - gyaw))))
        print("[pose] 终点 " + pose_str(p1))
        print("[pose] 净位移 %.3f m  用时 %.1f s  平均 %.3f m/s"
              % (net, el, net / max(el, 1e-3)))
        print("[nav] 目标残差 %.3f m（判据 ≤ 0.06）  yaw 误差 %.2f°（判据 ≤ 10）  "
              "平台判据 stopped=%s"
              % (resid, yaw_err, stopped))
        arrived = arrived_sdk or (stopped and resid <= 0.06
                                  and yaw_err <= 10.0)
    else:
        arrived = arrived_sdk
    print("[nav] 到位" if arrived else "[nav] 未确认到位")
    return 0 if arrived else 4


# ---------------------------------------------------------------- 入口

def parse(argv):
    opts = {"file": DEFAULT_FILE, "note": "", "force": False, "speed": 0.2,
            "omni": False, "dry": False, "timeout": 120.0, "ignore_map": False}
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--file":
            opts["file"] = argv[i + 1]; i += 2
        elif a == "--note":
            opts["note"] = argv[i + 1]; i += 2
        elif a == "--force":
            opts["force"] = True; i += 1
        elif a == "--speed":
            opts["speed"] = float(argv[i + 1]); i += 2
        elif a == "--omni":
            opts["omni"] = True; i += 1
        elif a == "--dry-run":
            opts["dry"] = True; i += 1
        elif a == "--timeout":
            opts["timeout"] = float(argv[i + 1]); i += 2
        elif a == "--ignore-map":
            opts["ignore_map"] = True; i += 1
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            pos.append(a); i += 1
    return opts, pos


def main():
    opts, pos = parse(sys.argv[1:])
    if not pos:
        print(__doc__)
        return 2
    cmd = pos[0]
    if cmd == "list":
        return do_list(opts)
    if cmd == "rm":
        if len(pos) < 2:
            print("[rm] 缺少标签，用法：rm LABEL [LABEL ...]")
            return 2
        return do_rm(opts, pos[1:])
    if cmd == "record":
        if len(pos) < 2:
            print("[record] 缺少标签，用法：record LABEL [--note TEXT] [--force]")
            return 2
        return do_record(opts, pos[1])
    if cmd == "go":
        if len(pos) < 2:
            print("[go] 缺少标签，用法：go LABEL [--speed V] [--omni] [--dry-run]")
            return 2
        return do_go(opts, pos[1])
    print("[cmd] 未知子命令：%s（可用 record / go / list / rm）" % cmd)
    return 2


if __name__ == "__main__":   # s1_nav_worker 常驻导入护栏（2026-11-12）：
    rc = 2                   # import 本模块不再触发 main/os._exit；CLI 路径逐位不变
    try:
        rc = main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        try:
            if nav is not None:
                nav.stop_navigation()
        except Exception as e:
            print("[exit] stop_navigation 异常（忽略）：%s" % e)
        try:
            if robot is not None:
                robot.request_shutdown()
                robot.wait_for_shutdown()
                robot.destroy()
                print("[exit] 优雅三连完成")
        except Exception as e:
            print("[exit] 关闭异常（忽略）：%s" % e)
    print("[exit] rc=%d" % rc, flush=True)
    os._exit(rc)
