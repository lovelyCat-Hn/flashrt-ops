#!/usr/bin/env python
"""S1 底盘移动（相对横移 / map 系绝对直达 / 发初始位姿），自 G1 g1_base_move.py 迁移。

保留 G1 版的进程隔离架构（SDK 原生段错误无法在 Python 侧捕获，2026-10-13 教训：
编排器本进程直连时一次 segfault 无声拖死整条流程；独立进程崩溃只死本段、rc=-11
明码上抛）与干净关闭纪律（不 destroy 的 SDK 进程每个漏 ~21 个 FastDDS
datasharing 段，/dev/shm 实测 422 死段 2.5GB；残留线程末尾 os._exit）。

与 G1 版的差异（对照 docs/s1-joint-api-quickref.md §6 + SDK 示例 complete_example.py）：
  1. SDK: galbot_sdk.s1；控制器名 SWERVE_CHASSIS_POSE_CTRL（G1=CHASSIS_POSE_CTRL，
     S1 是舵轮底盘 swerve）
  2. STATIONS 工位档案为空——S1 尚未建图；建图后按 G1 惯例（--where 实测落点
     填档）补，届时 --goto 可用
  3. 新增 --relocal：S1 SDK 原生 relocalize（G1 时代靠 engine_tools 菜单 2 发
     初始位姿，S1 换 singorix 栈后无此菜单）——发初始位姿并轮询 is_localized
  4. navigate_to_goal 多 omni_plan 参数（默认 False 不动）；其余签名同族同语义
     （move_straight_to=base_link 相对+odom 累计系，无需定位；navigate_to_goal=
     map 系绝对，需定位锁）

用法（LD_LIBRARY_PATH/PYTHONPATH=/data/galbot/lib，flash_pyrt311）：
  # 相对横移（base_link 系，x+=前 y+=左，无需定位）
  s1_base_move.py --shift-y 0.3
  # 发初始位姿（map 系 x,y_m + yaw_deg）并验定位锁；可与 --goal-* 串联
  s1_base_move.py --relocal 0.0 0.0 90
  # 绝对坐标直达（需定位锁在；collision 默认关=工位直达惯例，--collision-check 开）
  s1_base_move.py --goal-x 0.035 --goal-y 1.436 --goal-yaw-deg 2.84
  # 只查当前 map 系位姿 / SDK 健康预检（零运动）
  s1_base_move.py --where
  s1_base_move.py --probe-only
"""
import argparse
import math
import os
import time

# map 系工位档案（x,y_m / yaw_deg）。S1 尚未建图，留空；建图后照 G1 惯例：
# 手推到位 → --where 实测 → 填这里，坐标指挥的意义就是把机器人按回实测点。
STATIONS = {}


def pose7(x, y, yaw_deg):
    """map 系绝对位姿 → navigate_to_goal 的 7 元 Pose（z=0，四元数 [qx,qy,qz,qw]）。"""
    half = math.radians(yaw_deg) / 2.0
    return [x, y, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)]


def read_pose(nav):
    """当前 map 系位姿 → (x, y, yaw_deg)；绑定形态防御同 G1 版。None=未定位/异常。"""
    try:
        p = nav.get_current_pose()
    except Exception:
        return None
    if isinstance(p, (list, tuple)) and len(p) == 7:
        qx, qy, qz, qw = p[3], p[4], p[5], p[6]
        px, py = p[0], p[1]
    else:
        try:
            o = p.orientation
            qx, qy, qz, qw = o.x, o.y, o.z, o.w
            px, py = p.position.x, p.position.y
        except Exception:
            return None
    yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy),
                                  1.0 - 2.0 * (qy * qy + qz * qz)))
    return px, py, yaw


def shutdown_sdk(robot):
    """干净关闭（FastDDS /dev/shm 泄漏教训，同 G1 版）。"""
    try:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        print("SDK 已关闭", flush=True)
    except Exception as e:
        print(f"[move] SDK 关闭异常（忽略）：{e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shift-x", type=float, default=0.0,
                    help="前向位移 m（base_link 系，x+=前）")
    ap.add_argument("--shift-y", type=float, default=0.0,
                    help="左向位移 m（base_link 系，y+=左）")
    ap.add_argument("--shift-yaw-deg", type=float, default=0.0,
                    help="原地转角 deg（正值=逆时针/左转）")
    ap.add_argument("--relocal", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "YAW_DEG"),
                    help="发初始位姿（map 系）并轮询定位锁；可与 --goal-* 串联"
                         "（S1 原生 relocalize，替代 G1 engine_tools 菜单 2）")
    ap.add_argument("--move-timeout", type=float, default=30.0,
                    help="移动/定位轮询超时 s")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="ctor 后 settle 秒（前段刚释放 SDK，栈热留 0.5s）")
    ap.add_argument("--collision-check", action="store_true",
                    help="navigate_to_goal 开碰撞检查（默认关=G1 工位直达惯例；"
                         "依赖部署侧配置时先核 S1 的导航配置）")
    ap.add_argument("--probe-only", action="store_true",
                    help="SDK 健康预检：窄域 start/stop + 干净关闭，零运动")
    ap.add_argument("--where", action="store_true",
                    help="只查当前位姿（map 系；未定位则明确告知）零运动")
    ap.add_argument("--goto", choices=sorted(STATIONS) or None,
                    help="map 系工位直达（STATIONS 档案；S1 建图后填档启用）")
    ap.add_argument("--goal-x", type=float, default=None,
                    help="绝对坐标 x m（map 系；与 --goal-y/--goal-yaw-deg 三件套）")
    ap.add_argument("--goal-y", type=float, default=None,
                    help="绝对坐标 y m（map 系）")
    ap.add_argument("--goal-yaw-deg", type=float, default=None,
                    help="绝对坐标朝向 deg（绝对角，非相对转角）")
    args = ap.parse_args()
    if args.goto and (args.goal_x is not None or args.goal_y is not None
                      or args.goal_yaw_deg is not None):
        ap.error("--goto 与 --goal-* 互斥，二选一")
    if not args.goto and any(v is not None for v in
                             (args.goal_x, args.goal_y, args.goal_yaw_deg)):
        if None in (args.goal_x, args.goal_y, args.goal_yaw_deg):
            ap.error("--goal-x/--goal-y/--goal-yaw-deg 需三个同时给")
    abs_goal = bool(args.goto) or args.goal_x is not None
    if not (abs_goal or args.relocal or args.shift_x or args.shift_y
            or args.shift_yaw_deg or args.where or args.probe_only):
        ap.error("无事可做：给 --shift-*/--relocal/--goal-*/--goto/--where/--probe-only 之一")

    from galbot_sdk.s1 import (GalbotRobot, GalbotNavigation,  # 延迟导入：--help 不碰 SDK
                               S1ControllerName)
    print("[move] 连接 SDK（子进程隔离版）...", flush=True)
    robot = GalbotRobot()
    # ⚠ init 必调（G1 教训：缺它 start_controller = ControllerManager this=NULL
    # 原地 Segfault）；init(set()) 写法照抄 ~/yuqiz/s1_teleop 真机跑通代码
    if not robot.init(set()):
        print("[move] robot.init 失败", flush=True)
        shutdown_sdk(robot)
        os._exit(1)
    time.sleep(args.settle)
    st = robot.start_controller("all")
    print(f"start_controller('all') → {st}", flush=True)
    if not str(st).startswith("ControlStatus.SUCCESS"):
        shutdown_sdk(robot)
        os._exit(1)

    code = 0
    nav = None
    try:
        if args.where:
            # 位姿查询：map 系（需定位）；move_straight_to 走 odom 累计系不依赖
            nav = GalbotNavigation()
            if not nav.init():
                print("[where] GalbotNavigation.init 失败", flush=True)
                shutdown_sdk(robot)
                os._exit(1)
            try:
                if not bool(nav.is_localized()):
                    print("[where] 未定位（map 系位姿不可用）。move_straight_to "
                          "走 odom 累计系不依赖定位，横移不受影响")
                else:
                    p = read_pose(nav)
                    if p is None:
                        print("[where] 已定位但位姿解析失败（绑定形态异常）")
                    else:
                        print(f"[where] map 系位姿 x={p[0]:+.3f} "
                              f"y={p[1]:+.3f} yaw={p[2]:+.2f}°", flush=True)
            finally:
                try:
                    nav.stop_navigation()
                except Exception:
                    pass
            shutdown_sdk(robot)
            os._exit(0)
        elif args.probe_only:
            # 预检信号与真跑同款走 "all"（G1 INVALID_INPUT 案：窄域名不被
            # start_controller 接受）；另验 nav 子系统连通
            nav = GalbotNavigation()
            if not nav.init():
                print("[probe] GalbotNavigation.init 失败", flush=True)
                shutdown_sdk(robot)
                os._exit(1)
            nav.stop_navigation()
            print("[probe] SDK 健康（start_controller('all') + nav.init 通），"
                  "零运动退出", flush=True)
        else:
            t0 = time.time()
            nav = GalbotNavigation()
            if not nav.init():
                raise RuntimeError("GalbotNavigation.init 失败")
            st_sw = robot.switch_controller(S1ControllerName.SWERVE_CHASSIS_POSE_CTRL)
            print(f"switch_controller(swerve_chassis_pose_ctrl) → {st_sw}",
                  flush=True)
            if not str(st_sw).startswith("ControlStatus.SUCCESS"):
                raise RuntimeError(f"switch_controller 失败 {st_sw}")

            if args.relocal is not None:
                # ── 发初始位姿（S1 原生 relocalize，官方示例同款轮询）──
                rx, ry, ryaw = args.relocal
                print(f"[relocal] 发初始位姿 x={rx:+.3f} y={ry:+.3f} "
                      f"yaw={ryaw:+.2f}°，轮询定位锁（≤{args.move_timeout:.0f}s）",
                      flush=True)
                locked, t_r = False, time.time()
                while time.time() - t_r < args.move_timeout:
                    try:
                        nav.relocalize(pose7(rx, ry, ryaw))
                    except Exception as e:
                        print(f"[relocal] relocalize 异常（重试）：{e}", flush=True)
                    time.sleep(0.5)
                    try:
                        locked = bool(nav.is_localized())
                    except Exception:
                        locked = False
                    if locked:
                        break
                cur = read_pose(nav)
                print(f"[relocal] {'✅ 定位锁在' if locked else '⛔ 未锁上'}"
                      f"（{time.time() - t_r:.1f}s；当前 "
                      + (f"x={cur[0]:+.3f} y={cur[1]:+.3f} yaw={cur[2]:+.2f}°"
                         if cur else "位姿未知")
                      + "）", flush=True)
                if not locked:
                    raise RuntimeError("relocal 后仍未定位——核初始位姿数值/地图在位")

            if abs_goal:
                # ── 绝对坐标直达（map 系）：G1 工位直达同款 ──
                if args.goto:
                    gx, gy, gyaw = STATIONS[args.goto]
                    tag = f"工位 {args.goto}"
                else:
                    gx, gy, gyaw = (args.goal_x, args.goal_y, args.goal_yaw_deg)
                    tag = "指定坐标"
                if not args.relocal:      # relocal 分支刚验过锁
                    if not bool(nav.is_localized()):
                        raise RuntimeError("未定位（is_localized=False）——绝对"
                                           "坐标没有 map 锁等于盲走；先 "
                                           "--relocal 发初始位姿再试")
                cur = read_pose(nav)
                print(f"[goto] {tag} map 系绝对目标 x={gx:+.3f} y={gy:+.3f} "
                      f"yaw={gyaw:+.2f}°（当前 "
                      + (f"x={cur[0]:+.3f} y={cur[1]:+.3f} "
                         f"yaw={cur[2]:+.2f}°" if cur else "位姿未知")
                      + "）", flush=True)
                ret = nav.navigate_to_goal(pose7(gx, gy, gyaw),
                                           enable_collision_check=args.collision_check,
                                           is_blocking=False,
                                           timeout=args.move_timeout)
                ok, status = (ret if isinstance(ret, tuple) and len(ret) == 2
                              else (bool(ret), ret))
                print(f"[goto] 接单 → ok={ok} status={status}", flush=True)
                reached, t_g = False, time.time()
                while time.time() - t_g < args.move_timeout:
                    time.sleep(0.5)
                    try:
                        if nav.check_goal_arrival():
                            reached = True
                            break
                    except Exception:
                        pass
                cur = read_pose(nav)
                print(f"[goto] {'✅ 到达' if reached else '⛔ 超时未判到达'}"
                      f"（{time.time() - t0:.1f}s；实到 "
                      + (f"x={cur[0]:+.3f} y={cur[1]:+.3f} "
                         f"yaw={cur[2]:+.2f}°" if cur else "位姿未知")
                      + "）", flush=True)
                if not reached:
                    code = 1
            elif args.relocal is None:
                # ── 相对横移（base_link 系，无需定位）──
                half = math.radians(args.shift_yaw_deg) / 2.0
                goal = [args.shift_x, args.shift_y, 0.0,
                        0.0, 0.0, math.sin(half), math.cos(half)]
                print(f"[move] move_straight_to 相对位移 x={args.shift_x:+.3f} "
                      f"y={args.shift_y:+.3f} m"
                      f" yaw={args.shift_yaw_deg:+.2f}°（base_link 系，y+=左）"
                      f" timeout {args.move_timeout:.0f}s", flush=True)
                ret = nav.move_straight_to(goal, is_blocking=True,
                                           timeout=args.move_timeout)
                # .pyi 只写 tuple：防御性解包，兼容 (bool, status)/单值两种形态
                ok, status = (ret if isinstance(ret, tuple) and len(ret) == 2
                              else (bool(ret), ret))
                print(f"[move] → ok={ok} status={status} "
                      f"用时 {time.time()-t0:.1f}s", flush=True)
                if not ok:
                    code = 1
    except Exception as e:
        print(f"[move] 异常：{e}", flush=True)
        code = 1
    if nav is not None:
        try:
            nav.stop_navigation()   # 释放导航对底盘的控制（幂等）
        except Exception:
            pass
    shutdown_sdk(robot)
    os._exit(code)   # SDK 残留线程：绕过 atexit 干净退（G1 同款）


if __name__ == "__main__":
    main()
