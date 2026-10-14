#!/usr/bin/env python
"""G1 底盘相对横移（编排器 move 段子进程 + SDK 健康预检）。

为什么独立进程（2026-10-13 教训）：SDK 原生段错误无法在 Python 侧捕获，
编排器本进程直连时一次 segfault 无声拖死整条流程（表面症状=「底盘没动」）。
进程隔离后崩溃只死本段、rc=-11 明码上抛；配合 --probe-only 预检，SDK 全局
态中毒（症状=任何新 SDK 进程 start_controller 原地崩，gdb 栈=
ControllerManager::has_controller_group 空引用，10-13 案）在 pick 前红灯，
不再烧掉一整段 pick 才发现。处置=重启服务栈+清 /dev/shm 死段。

底盘相对横移契约（官方样例 docs/g1/zh/python/galbot_navigation/
move_straight_to.py + galbot_navigation.hpp:215-248）：
  nav = GalbotNavigation(); nav.init()
  robot.switch_controller(CHASSIS_POSE_CTRL)   # 只切 chassis 组，不碰臂组
  nav.move_straight_to(7 元 Pose, is_blocking=True, timeout)
  nav.stop_navigation()
  Pose 相对当前 base_link、odom 累计系；无需定位、无碰撞检查、不走全局
  规划；x 前 / y 左 / z 上；阻塞返回 (success, status) =到位才 True。
  ⚠ GalbotRobot 本体绑定无 move_straight_to（10-13 AttributeError 案，API
  挂在独立的 GalbotNavigation 对象上）；GalbotRobot.set_base_pose(
  base_link/odom) 语义近似可作备用但无官方文档、阻塞语义含糊。

用法（一般经 run_g1_task_pickplace.py 派发，手动调试直接跑）：
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/g1/g1_base_move.py --shift-y 0.3
  # 绝对坐标直达（map 系，2026-10-14 用户实测两工位 --where 定点）：
  #     ... g1_base_move.py --goto place    # 或 --goto pick（回程）
  #     ... g1_base_move.py --goal-x 0.035 --goal-y 1.436 --goal-yaw-deg 2.84
  # 走 navigate_to_goal(enable_collision_check=False)——同事 infer_policy/
  # robot_controller.py:393 同款（move_straight_to 无碰撞参数、且只收相对
  # 位移）；is_localized 拒发（绝对坐标没有锁就是盲走）+ 非阻塞轮询
  # check_goal_arrival 判到位。⚠ 依赖部署配置 sdk_navi_config.json 已关
  # check_vaild/check_collision（10-14 改，备份 .bak-1014）。
  # --probe-only：只窄域 start/stop 控制器 + 干净关闭，零运动 ~5s
"""
import argparse
import math
import os
import time

# 工位绝对坐标（map 系 x,y_m / yaw_deg）。10-14 用户 --where 实测：
# pick=发车原点，place=横移实到点（较 nav_points 理论差 ~3cm/2°，
# 以实到为准——坐标指挥的意义就是把机器人按回这两个实测点）。
STATIONS = {
    "pick":  (-0.001, -0.015, 0.47),
    "place": (+0.035, +1.436, 2.84),
}


def pose7(x, y, yaw_deg):
    """map 系绝对位姿 → navigate_to_goal 的 7 元 Pose（z=0，四元数 [qx,qy,qz,qw]）。"""
    half = math.radians(yaw_deg) / 2.0
    return [x, y, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)]


def read_pose(nav):
    """当前 map 系位姿 → (x, y, yaw_deg)；形态防御同 --where 分支。返回 None=未定位/异常。"""
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
    """干净关闭。不 destroy 的 SDK 进程死亡每个漏 ~21 个 FastDDS
    datasharing 段（10-13 /dev/shm 实测 422 死段共 2.5GB，全部来自当天
    异常退场的 SDK 进程）。残留线程仍在（warmup 同款），末尾 os._exit。"""
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
                    help="左向位移 m（base_link 系，y+=左，SDK 头文件明文）")
    ap.add_argument("--shift-yaw-deg", type=float, default=0.0,
                    help="原地转角 deg（正值=逆时针/左转）")
    ap.add_argument("--move-timeout", type=float, default=30.0,
                    help="move_straight_to 内部超时 s")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="ctor 后 settle 秒（前段刚释放 SDK，栈热留 0.5s）")
    ap.add_argument("--probe-only", action="store_true",
                    help="SDK 健康预检：窄域 start/stop + 干净关闭，零运动")
    ap.add_argument("--where", action="store_true",
                    help="只查当前位姿（map 系；未定位则明确告知）——"
                         "move_straight_to TIMEOUT 后看实际走到哪，零运动")
    ap.add_argument("--goto", choices=sorted(STATIONS),
                    help="绝对坐标直达 map 系工位（pick/place=10-14 --where "
                         "实测定点）；走 navigate_to_goal(collision_check="
                         "False)，需定位锁在")
    ap.add_argument("--goal-x", type=float, default=None,
                    help="绝对坐标 x m（map 系；与 --goal-y/--goal-yaw-deg "
                         "三件套同用，和 --goto 互斥）")
    ap.add_argument("--goal-y", type=float, default=None,
                    help="绝对坐标 y m（map 系）")
    ap.add_argument("--goal-yaw-deg", type=float, default=None,
                    help="绝对坐标朝向 deg（绝对角，非相对转角）")
    args = ap.parse_args()
    if args.goto and (args.goal_x is not None or args.goal_y is not None
                      or args.goal_yaw_deg is not None):
        ap.error("--goto 与 --goal-* 互斥，二选一")
    abs_goal = bool(args.goto)
    if not abs_goal and any(v is not None for v in
                            (args.goal_x, args.goal_y, args.goal_yaw_deg)):
        if None in (args.goal_x, args.goal_y, args.goal_yaw_deg):
            ap.error("--goal-x/--goal-y/--goal-yaw-deg 需三个同时给")
        abs_goal = True

    from galbot_sdk.g1 import (GalbotRobot, GalbotNavigation,  # 延迟导入：--help 不碰 SDK
                               G1ControllerName)
    scope = "all"
    print(f"[move] 连接 SDK（子进程隔离版；start_controller('{scope}')）...",
          flush=True)
    robot = GalbotRobot()
    # ⚠ init() 必调（reset_pos/warmup/loop 三样本同款）：缺它的
    # start_controller = ControllerManager this=NULL 原地 Segfault
    # （2026-10-13 案，gdb 栈 has_controller_group+16 ldr x1,[x0]）
    if not robot.init():
        print("[move] robot.init 失败", flush=True)
        shutdown_sdk(robot)
        os._exit(1)
    time.sleep(args.settle)
    st = robot.start_controller(scope)
    print(f"start_controller('{scope}') → {st}", flush=True)
    if not str(st).startswith("ControlStatus.SUCCESS"):
        shutdown_sdk(robot)
        os._exit(1)

    code = 0
    nav = None
    try:
        if args.where:
            # 位姿查询：map 系（需定位）；move_straight_to 走 odom 累计系
            # 不依赖定位，故未定位只影响本查询可读性、不影响横移
            nav = GalbotNavigation()
            if not nav.init():
                print("[where] GalbotNavigation.init 失败", flush=True)
                shutdown_sdk(robot)
                os._exit(1)
            try:
                loc = bool(nav.is_localized())
                if not loc:
                    print("[where] 未定位（map 系位姿不可用）。move_straight_to "
                          "走 odom 累计系不依赖定位，横移不受影响")
                else:
                    p = nav.get_current_pose()
                    # 绑定形态防御（同 move_straight_to goal 解包）：7 元列表
                    # [x,y,z,qx,qy,qz,qw] 或带 position/orientation 属性的对象
                    if isinstance(p, (list, tuple)) and len(p) == 7:
                        qx, qy, qz, qw = p[3], p[4], p[5], p[6]
                        px, py = p[0], p[1]
                    else:
                        o = p.orientation
                        qx, qy, qz, qw = o.x, o.y, o.z, o.w
                        px, py = p.position.x, p.position.y
                    yaw = math.degrees(math.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz)))
                    print(f"[where] map 系位姿 x={px:+.3f} "
                          f"y={py:+.3f} yaw={yaw:+.2f}°", flush=True)
            finally:
                try:
                    nav.stop_navigation()
                except Exception:
                    pass
            shutdown_sdk(robot)
            os._exit(0)
        elif args.probe_only:
            # 预检信号与真跑同款走 "all"（INVALID_INPUT 案：窄域名不被
            # start_controller 接受；接上=保持当前位姿，无运动指令）；
            # 另验 nav 子系统连通（真跑横移的前置，init 0.4s 实测）
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
            st_sw = robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
            print(f"switch_controller(chassis_pose_ctrl) → {st_sw}",
                  flush=True)
            if not str(st_sw).startswith("ControlStatus.SUCCESS"):
                raise RuntimeError(f"switch_controller 失败 {st_sw}")
            if abs_goal:
                # ── 绝对坐标直达（map 系）：navigate_to_goal 同事款 ──
                if args.goto:
                    gx, gy, gyaw = STATIONS[args.goto]
                    tag = f"工位 {args.goto}"
                else:
                    gx, gy, gyaw = (args.goal_x, args.goal_y,
                                    args.goal_yaw_deg)
                    tag = "指定坐标"
                loc = False
                try:
                    loc = bool(nav.is_localized())
                except Exception:
                    pass
                if not loc:
                    raise RuntimeError("未定位（is_localized=False）——绝对"
                                       "坐标没有 map 锁等于盲走；先发初始"
                                       "位姿再试")
                cur = read_pose(nav)
                print(f"[goto] {tag} map 系绝对目标 x={gx:+.3f} y={gy:+.3f} "
                      f"yaw={gyaw:+.2f}°（当前 "
                      + (f"x={cur[0]:+.3f} y={cur[1]:+.3f} "
                         f"yaw={cur[2]:+.2f}°" if cur else "位姿未知")
                      + "）", flush=True)
                ret = nav.navigate_to_goal(pose7(gx, gy, gyaw),
                                           enable_collision_check=False,
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
            else:
                # ── 相对横移（原路径，编排器 fold 段同语义）──
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
    os._exit(code)   # SDK 残留线程：绕过 atexit 干净退（warmup 同款）


if __name__ == "__main__":
    main()
