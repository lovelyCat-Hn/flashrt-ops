#!/usr/bin/env python
"""pns 导航会话探针（官方例程顺序）：

  ctrl-sweep   robot.init → switch_controller(SWERVE_CHASSIS_POSE_CTRL) → 八方向+参考点可达性扫描
  go X Y       同上初始化 → navigate_to_goal(阻塞) → 到达检查

用法: PYTHONPATH=/userdata/update/manual_update/lib python pns_nav_session.py <mode> [dx dy]
安全: 结束必走优雅三连 + os._exit（S1 退出段 segfault 教训）。
"""
import faulthandler
import math
import os
import sys
import time

import numpy as np

# 挂死自检：25s 后自动 dump 全线程栈并退出（定位 SDK 阻塞点）
faulthandler.dump_traceback_later(25, exit=True)

from galbot_sdk.s1 import GalbotRobot, GalbotNavigation, S1ControllerName

mode = sys.argv[1] if len(sys.argv) > 1 else "ctrl-sweep"
robot = None
nav = None


def sweep(nav, x, y):
    start = nav.get_current_pose()
    print("[probe] 八方向扫描（0.5m 与 1m）+ 参考点：")
    targets = []
    for dist in (0.5, 1.0):
        for ang in range(0, 360, 45):
            a = math.radians(ang)
            targets.append((f"{ang:>3}° {dist}m", x + dist * math.cos(a), y + dist * math.sin(a)))
    targets.append(("远点 3m 前方", x + 3.0, y))
    n_ok = 0
    for name, gx, gy in targets:
        goal = np.array([gx, gy, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        try:
            ok = bool(nav.check_path_reachability(goal, start))
        except Exception as e:
            ok = f"异常:{e}"
        n_ok += 1 if ok is True else 0
        print(f"  {name}: {ok}")
    print(f"[probe] 汇总: {n_ok}/{len(targets)} 可达")
    return n_ok


try:
    robot = GalbotRobot()
    print("[init] robot.init →", robot.init())
    nav = GalbotNavigation()
    if not nav.init():
        print("[init] nav.init 失败")
        os._exit(2)
    for _ in range(20):
        if nav.is_localized():
            break
        time.sleep(0.5)
    p = nav.get_current_pose()
    x, y, qz, qw = p[0], p[1], p[5], p[6]
    yaw = math.degrees(math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * qz * qz))
    print(f"[init] 定位 ({x:.4f}, {y:.4f}) yaw≈{yaw:.2f}°")

    st = robot.switch_controller(S1ControllerName.SWERVE_CHASSIS_POSE_CTRL)
    print(f"[init] switch_controller(SWERVE_CHASSIS_POSE_CTRL) → {st}")
    time.sleep(1.5)  # 让控制模式切换落地

    if mode == "ctrl-sweep":
        sweep(nav, x, y)
    elif mode == "go":
        dx = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        dy = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        goal = np.array([x + dx, y + dy, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        ok = bool(nav.check_path_reachability(goal, nav.get_current_pose()))
        print(f"[go] 目标 ({goal[0]:.4f}, {goal[1]:.4f}) 可达性={ok}")
        if not ok:
            print("[go] pns 判不可达，不下发")
            os._exit(3)
        print("[go] navigate_to_goal 下发（避障开、阻塞、timeout 120s）…")
        t0 = time.time()
        st2 = nav.navigate_to_goal(goal, enable_collision_check=True, is_blocking=True,
                                   timeout=120, omni_plan=False)  # S1 铁律 omni_plan=False
        print(f"[go] 返回 {st2}  用时 {time.time()-t0:.1f}s")
        try:
            print(f"[go] check_goal_arrival → {nav.check_goal_arrival()}")
        except Exception as e:
            print(f"[go] arrival 查询异常: {e}")
        nav.stop_navigation()
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    try:
        if robot is not None:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()
            print("[exit] 优雅三连完成")
    except Exception as e:
        print(f"[exit] 关闭异常（忽略）：{e}")
print("[exit] done", flush=True)
os._exit(0)
