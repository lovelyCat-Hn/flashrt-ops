#!/usr/bin/env python
"""warm 复位序列共享核心（g1_pose_warmup.py 与 run_g1_loop.py 折叠段共用）。

2026-10-13 折叠改造：编排器门触发后不再换 warmup 子进程，pick 进程同会话内
直接摆臂+横移（省一次 init+destroy+进程启动 ~4-5s）。本模块从
g1_pose_warmup.py 原样抽出共用序列——【改一处两处生效，勿在调用侧重写】。

契约要点（与 warmup 逐字同源）：
  - 23 维数据集序【右臂在前】：0-6 右臂 / 7 右夹爪 / 8-14 左臂 / 15 左夹爪
    / 16-20 腿 / 21-22 头（meta/info.json 权威定义；读序一律跟数据集序，
    曾因左右整块互换把右臂目标下给左臂、机械臂镜像甩到背后——用户实测）
  - 段②=臂+头 @speed(45s 超时) → 睡 1s → 腿 @0.2(45s) → 睡 1s → 读回判达标
  - 达标=归一化 [-1,1] 外的臂+夹爪维（dim0-15）为 0；腿/头窄维亮灯属噪声量级
  - 底盘横移=官方 GalbotNavigation.move_straight_to 四步
    （nav.init 0.4s 实测 / switch 只切 chassis 组不碰臂组 / stop 幂等）；
    ⚠ move_straight_to 不在 GalbotRobot 本体绑定上（10-13 AttributeError 案）
"""
import json
import math
import pathlib
import sys
import time

import numpy as np

if "/home/galbot/holy/FlashRT" not in sys.path:
    sys.path.insert(0, "/home/galbot/holy/FlashRT")
try:
    from flash_rt.core.utils.actions import normalize_state
except Exception:            # fold 侧 loop 本就带 FlashRT；纯语法检查时容错
    normalize_state = None

from galbot_sdk.g1 import GalbotNavigation, G1ControllerName  # noqa: E402

# ── 23 维数据集序常量（warmup 原样）──
LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
HEAD = ["head_joint1", "head_joint2"]
LEG = [f"leg_joint{i}" for i in range(1, 6)]
GRIP = ["right_gripper_joint1", "left_gripper_joint1"]   # 数据集序: dim7 右 / dim15 左
STATE_NAMES = RIGHT + GRIP[:1] + LEFT + GRIP[1:] + LEG + HEAD
IDX = {"right_arm": list(range(0, 7)), "left_arm": list(range(8, 15)),
       "head": list(range(21, 23)), "leg": list(range(16, 21))}
ARM_NAMES = RIGHT + LEFT + HEAD   # 名字与目标值逐位配对，顺序跟数据集维序


def load_target(ckpt_dir, pose_file=None, log=print):
    """目标位姿三级来源（warmup 原样）：CLI --pose-file > 部署目录
    episode_start_task0.json > norm_stats 的 state.mean（多 task 平均，
    不对应任何任务起点——仅兜底）。返回 (TARGET, ns, norm_mode)。"""
    ckpt_dir = pathlib.Path(ckpt_dir)
    ns_p = ckpt_dir / "norm_stats.json"
    # ⚠ state.mean 是 task0（夹爪 0%）与 task1（33%/高位）两组起始位的平均
    # ——不对应任何任务起点，臂维差达 0.83 rad（2026-09-23 实证）
    start_p = (pathlib.Path(pose_file) if pose_file
               else ckpt_dir / "episode_start_task0.json")
    if start_p.exists():
        blk = json.loads(start_p.read_text())
        target = blk["start_pose"]
        ns = json.loads(ns_p.read_text()) if ns_p.exists() else None
        mode = ns.get("norm_mode", "q01_q99") if isinstance(ns, dict) else "q01_q99"
        log(f"[warm] 目标 = {start_p}"
            + ("" if pose_file else "（部署目录零配置发现）")
            + f"（{blk.get('n_episodes', '?')} 条起始中位"
              f"{('，task: ' + blk['task']) if blk.get('task') else ''}，"
              f"臂维组内最大波动 {blk.get('spread_max_dev', 0):.2f} rad）")
    elif ns_p.exists():                 # prep 产物：{norm_mode, actions{...}, state{...}}
        ns = json.loads(ns_p.read_text())
        target = ns["state"]["mean"]
        mode = ns.get("norm_mode", "q01_q99")
        log("[warm] ⚠ 无位姿文件（--pose-file / config [warmup].pose_file / "
            "部署目录 episode_start_task0.json 都没有），退回 state.mean"
            "（多 task 起始的平均值，非任务起点——用 g1_pose_extract.py 生成）")
    else:                               # 兜底：lerobot 数据集 schema
        ms_p = ckpt_dir.parent / "meta" / "stats.json"
        if not ms_p.exists():
            raise SystemExit(f"找不到 stats：{ns_p} 或 {ms_p}")
        lg = json.loads(ms_p.read_text())
        target = lg["observation.state"]["mean"]
        mode = "q01_q99"
    if len(target) != 23:
        raise SystemExit(f"目标姿态应 23 维，实得 {len(target)}")
    return target, ns, mode


def read23(robot, state_names, grip_wmin, grip_wmax, die):
    """读 23 维关节（数据集序），夹爪维米→0~100% 换算。

    首读可能撞 DDS 发现未完成（前段刚 os._exit、本进程全新参与者刚入网；
    10-13 八跑 0 维实录，事后复测读数正常=瞬态）——带截止重试，不再一枪毙命。
    失败调 die(msg)（调用方决定退出语义）。"""
    vals, t0, n = None, time.time(), 0
    while time.time() - t0 < 10.0:
        n += 1
        vals = robot.get_joint_positions([], state_names)
        if vals and len(vals) == 23:
            break
        if n == 1:
            print("关节读取空/短维，等待 DDS 发现重试（最多 10s）...", flush=True)
        time.sleep(0.5)
    else:
        die(f"关节读取失败（10s 重试 {n} 次后仍 {len(vals) if vals else 0} 维）")
    st = [float(v) for v in vals]
    if grip_wmin is not None and grip_wmax is not None:
        for i in (7, 15):   # SDK 米 → 数据集 0~100%
            st[i] = (st[i] - grip_wmin) / (grip_wmax - grip_wmin + 1e-9) * 100.0
    return st


def report(cur, target, state_names):
    print(f"{'维':>4} {'关节':<22} {'当前':>8} {'目标':>8} {'差':>8}")
    for i, n in enumerate(state_names):
        d = target[i] - cur[i]
        mark = " ←" if abs(d) > 0.05 and n not in GRIP else ""
        print(f"{i:>4} {n:<22} {cur[i]:>8.3f} {target[i]:>8.3f} {d:>+8.3f}{mark}")


def send_warm(robot, target, speed, skip_leg=False, log=print):
    """段②：臂+头 → 目标位姿（@speed）→ 睡 1s；段②b：躯干/腿 @0.2 → 睡 1s。
    与 warmup 逐字同源（含状态打印不校验——达标与否由 judge 判）；
    各段打耗时（阻塞返回=到位，用时即实际运动时长）。"""
    arm_tgt = [target[i] for i in IDX["right_arm"] + IDX["left_arm"] + IDX["head"]]
    t0 = time.time()
    st = robot.set_joint_positions(arm_tgt, joint_names=ARM_NAMES,
                                   is_blocking=True, speed_rad_s=speed,
                                   timeout_s=45.0)
    log(f"set_joint_positions(臂+头) → {st}（{time.time() - t0:.1f}s）")
    time.sleep(1.0)
    if not skip_leg:
        leg_tgt = [target[i] for i in IDX["leg"]]
        t1 = time.time()
        st_l = robot.set_joint_positions(leg_tgt, joint_names=LEG,
                                         is_blocking=True, speed_rad_s=0.2,
                                         timeout_s=45.0)
        log(f"set_joint_positions(躯干/腿) → {st_l}（{time.time() - t1:.1f}s）")
        time.sleep(1.0)


def judge(cur, target, ns, skip_leg=False, log=print):
    """达标判定：归一化 [-1,1] 外维数；返回臂+夹爪段（dim0-15）外维列表。"""
    if normalize_state is None:
        raise RuntimeError("flash_rt normalize_state 不可用（FlashRT 不在 path）")
    st_n = normalize_state(np.array(cur, dtype=np.float32), ns)
    out_dims = [f"dim{i}({STATE_NAMES[i]})" for i in range(23) if abs(st_n[i]) > 1.0]
    arms_out = [d for d in out_dims if int(d[3:d.index("(")]) < 16]   # 臂+夹爪段
    log(f"\n== 达标判定 ==\n[-1,1] 外 {len(out_dims)} 维: {out_dims or '无'}")
    log(f"双臂维(0-7,8-15) 外 {len(arms_out)} 维 → "
        + ("✅ 达标，可跑推理" if not arms_out else "⚠ 仍偏，检查臂段是否到位"))
    log("（leg/head 窄维亮灯属传感器噪声量级， Part B 已知）")
    leg_dev = max(abs(cur[IDX['leg'][k]] - target[IDX['leg'][k]]) for k in range(5))
    log(f"躯干/腿最大偏差 {leg_dev:.3f} rad"
        + ("（已下发对齐；偏大=未到位，查上方状态）" if not skip_leg
           else "（--skip-leg 未下发，仅对照）"))
    return arms_out


def nav_straight_move(robot, x, y, yaw_deg, timeout, log=print,
                      ready_retries=15, ready_gap_s=4.0, loc_wait_s=90.0):
    """官方横移四步（docs/g1/zh/python/galbot_navigation/move_straight_to.py）：
    nav.init → switch_controller(chassis 组，不碰臂组) → move_straight_to
    （7 元 Pose 相对 base_link/odom 系，阻塞=到位才 True）→ stop_navigation。
    返回 (ok, detail)。

    WAIT_INITIALIZED（galbot_sdk_type.hpp:302「导航系统未就绪」）=即时拒绝
    非等待（is_blocking 也不等，10-13 折叠首跑实录 0s 即返）：--nav-suspend
    刚 CONT 醒来的导航栈 init 探活能过、内部还没就绪——按「还没好」语义
    限时重试，其他状态照旧一枪定音。
    ⚠ 10-14 实测（Warning 日志）：冻结期 pns_core 数据新鲜度看门狗记下
    global_pose/base_odom/localization_score 的 49.5s 陈旧缺口（冻结的 7 进程
    含 localization_server 自己），解冻后要等话题恢复新鲜+稳定窗口才接活
    （实测 ~60-90s 才干净）——18s 窗（6×3s）在 13:01:03 放弃时服务仍拒，
    13:01:09 才缓过来。窗口已扩到 15 次 × 4s ≈ 60s。
    另设定位闸（loc_wait_s）：is_localized 不就绪不发车（见函数体内注释）。"""
    half = math.radians(yaw_deg) / 2.0
    goal = [x, y, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)]
    log(f"[move] move_straight_to 相对位移 x={x:+.3f} y={y:+.3f} m "
        f"yaw={yaw_deg:+.2f}°（base_link 系 y+=左）timeout {timeout:.0f}s")
    t0 = time.time()
    nav = GalbotNavigation()
    if not nav.init():
        return False, "GalbotNavigation.init 失败"
    # ⚠ 10-14 实录：move_straight_to 不查定位——--nav-suspend 解冻后
    # localization score=-1（未重锁）它照样执行，到位后定位重锁、map 系
    # belief 跳变 → 到达判定失败 → 规划器「回退→再进」危险三段式（日志
    # "executing, localization score: -1" 20Hz 实锤）。发车前必须等定位。
    if loc_wait_s > 0:
        t_loc = time.time()
        loc_ok = False
        while time.time() - t_loc < loc_wait_s:
            try:
                loc_ok = bool(nav.is_localized())
            except Exception:
                loc_ok = False
            if loc_ok:
                break
            time.sleep(2.0)
        log(f"[move] 定位{'就绪' if loc_ok else '未就绪'}"
            f"（等 {time.time() - t_loc:.1f}s）")
        if not loc_ok:
            try:
                nav.stop_navigation()
            except Exception:
                pass
            return False, f"LOCALIZATION_TIMEOUT({loc_wait_s:.0f}s)"
    try:
        st_sw = robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
        log(f"[move] switch_controller(chassis_pose_ctrl) → {st_sw} "
            f"({time.time() - t0:.1f}s)")
        if not str(st_sw).startswith("ControlStatus.SUCCESS"):
            return False, f"switch_controller 失败 {st_sw}"
        ok, st = False, ""
        for attempt in range(max(1, ready_retries)):
            ret = nav.move_straight_to(goal, is_blocking=True, timeout=timeout)
            ok, st = (ret if isinstance(ret, tuple) and len(ret) == 2
                      else (bool(ret), ret))
            if ok or str(st) != "WAIT_INITIALIZED":
                break
            if attempt < ready_retries - 1:
                log(f"[move] 导航栈未就绪（WAIT_INITIALIZED），{ready_gap_s:.0f}s "
                    f"后重试（{attempt + 1}/{ready_retries - 1}）...")
                time.sleep(ready_gap_s)
        log(f"[move] → ok={ok} status={st} 用时 {time.time() - t0:.1f}s")
        return bool(ok), str(st)
    finally:
        try:
            nav.stop_navigation()   # 释放导航对底盘的控制（幂等）
        except Exception:
            pass


# 工位绝对坐标（map 系 x,y_m / yaw_deg）。10-14 用户 --where 实测：
# pick=发车原点，place=横移实到点（较 nav_points 理论差 ~3cm/2°，以实到为准）。
# ⚠ 与 g1_base_move.py 的 STATIONS 同源镜像（改一处同步另一处——那边为保
# --help 零 SDK 导入不便反抓本模块，两份都是 2 行字典，成本可控）。
STATIONS = {
    "pick":  (-0.001, -0.015, 0.47),
    "place": (+0.035, +1.436, 2.84),
}


def _pose7(x, y, yaw_deg):
    """map 系绝对位姿 → navigate_to_goal 的 7 元 Pose（z=0，四元数 [qx,qy,qz,qw]）。"""
    half = math.radians(yaw_deg) / 2.0
    return [x, y, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)]


def nav_goto_station(robot, station, timeout, log=print,
                     ready_retries=15, ready_gap_s=4.0, loc_wait_s=90.0):
    """fold 横移的绝对坐标版（10-14 --goto 同款 API）：map 系工位直达。
    navigate_to_goal(enable_collision_check=False, is_blocking=False)——
    move_straight_to 只收相对位移且无碰撞参数（SDK 桩 1295-1313）；同事
    infer_policy/robot_controller.py:393 同款 + 0.5s 轮询 check_goal_arrival。
    定位闸（绝对坐标没有 map 锁=盲走）与 WAIT_INITIALIZED 限时重试与
    nav_straight_move 同源（--nav-suspend 解冻后 pns 新鲜度恢复窗 60-90s）。
    返回 (ok, detail)。"""
    gx, gy, gyaw = STATIONS[station]
    t0 = time.time()
    nav = GalbotNavigation()
    if not nav.init():
        return False, "GalbotNavigation.init 失败"
    try:
        if loc_wait_s > 0:      # 定位闸：同 nav_straight_move（10-14 实录）
            t_loc = time.time()
            loc_ok = False
            while time.time() - t_loc < loc_wait_s:
                try:
                    loc_ok = bool(nav.is_localized())
                except Exception:
                    loc_ok = False
                if loc_ok:
                    break
                time.sleep(2.0)
            log(f"[goto] 定位{'就绪' if loc_ok else '未就绪'}"
                f"（等 {time.time() - t_loc:.1f}s）")
            if not loc_ok:
                return False, f"LOCALIZATION_TIMEOUT({loc_wait_s:.0f}s)"
        st_sw = robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
        log(f"[goto] switch_controller(chassis_pose_ctrl) → {st_sw} "
            f"({time.time() - t0:.1f}s)")
        if not str(st_sw).startswith("ControlStatus.SUCCESS"):
            return False, f"switch_controller 失败 {st_sw}"
        ret = nav.navigate_to_goal(_pose7(gx, gy, gyaw),
                                   enable_collision_check=False,
                                   is_blocking=False, timeout=timeout)
        ok, st = (ret if isinstance(ret, tuple) and len(ret) == 2
                  else (bool(ret), ret))
        if str(st) == "WAIT_INITIALIZED":
            # 非阻塞接单也可能撞栈未醒透：限时重发（同 nav_straight_move 窗）
            for attempt in range(max(1, ready_retries) - 1):
                log(f"[goto] 导航栈未就绪（WAIT_INITIALIZED），"
                    f"{ready_gap_s:.0f}s 后重试（{attempt + 1}/"
                    f"{ready_retries - 1}）...")
                time.sleep(ready_gap_s)
                ret = nav.navigate_to_goal(_pose7(gx, gy, gyaw),
                                           enable_collision_check=False,
                                           is_blocking=False, timeout=timeout)
                ok, st = (ret if isinstance(ret, tuple) and len(ret) == 2
                          else (bool(ret), ret))
                if str(st) != "WAIT_INITIALIZED":
                    break
            if str(st) == "WAIT_INITIALIZED":
                return False, f"WAIT_INITIALIZED（重试 {ready_retries} 次）"
        log(f"[goto] 工位 {station}（x={gx:+.3f} y={gy:+.3f} "
            f"yaw={gyaw:+.2f}°）接单 → ok={ok} status={st}", flush=True)
        reached, t_g = False, time.time()
        while time.time() - t_g < timeout:
            time.sleep(0.5)
            try:
                if nav.check_goal_arrival():
                    reached = True
                    break
            except Exception:
                pass
        log(f"[goto] {'✅ 到达' if reached else '⛔ 超时未判到达'}"
            f"工位 {station}（{time.time() - t0:.1f}s）")
        return reached, ("ARRIVED" if reached else
                         f"ARRIVAL_TIMEOUT({timeout:.0f}s)")
    finally:
        try:
            nav.stop_navigation()   # 释放导航对底盘的控制（幂等）
        except Exception:
            pass
