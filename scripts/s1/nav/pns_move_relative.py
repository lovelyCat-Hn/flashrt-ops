#!/usr/bin/env python
"""底盘相对/绝对目标导航执行器（官方 navigate_to_goal_v2 流程）。

位置参数:
  dist            相对位移，单位 m：正数 = 前进（沿车头方向），负数 = 倒车（沿车尾方向），默认 1.0

可选参数:
  --omni          提高全向规划优先级 omni_plan=True（默认 False；False 时规划器
                  仍可能在全向直达探测通过后自行直线平移，转身与否以其裁决为准）。
                  注意：雷达对车尾为盲区，全向尤其是倒车时看不到后方动态障碍，
                  仅在后方静态无障碍时使用，详见 local_plan_config.toml 同名注释
  --speed V       本任务速度上限 m/s，同时作用于 x/y，角速度取 1.5V（默认 0.2）
  --acc A         全局加速度上限 m/s^2（可选，调用 set_navigation_kinematics_limits）
  --jerk J        全局加加速度上限 m/s^3（可选，与 --acc 配套使用）
  --pos X Y       改用 map 系绝对目标，忽略 dist
  --yaw DEG       绝对目标模式下指定终点朝向，缺省保持当前朝向
  --dry-run       只做可达性查询，不下发运动
  --timeout S     单次任务超时，默认 40 s

用法:
  PYTHONPATH=/userdata/update/manual_update/lib python pns_move_relative.py 1.0
  PYTHONPATH=/userdata/update/manual_update/lib python pns_move_relative.py -1.0 --omni --speed 0.15
  PYTHONPATH=/userdata/update/manual_update/lib python pns_move_relative.py --pos 1.20 -0.30 --dry-run

退出码: 0 到位 / 2 初始化失败 / 3 不可达未下发 / 4 超时或未到位 / 5 控制器切换失败
安全: 结束必走优雅三连 + os._exit（S1 退出段 segfault 教训）。
"""
import faulthandler
import math
import os
import sys
import time

import numpy as np

# SDK 未安装进 conda 环境，位于 manual_update 目录；未设 PYTHONPATH 时自动补齐，
# 使 `python pns_move_relative.py ...` 与 `PYTHONPATH=... python ...` 两种方式均可运行。
SDK_LIB = "/userdata/update/manual_update/lib"
if os.path.isdir(SDK_LIB) and SDK_LIB not in sys.path:
    sys.path.insert(0, SDK_LIB)

# 挂死自检：任务时长 + 20s 后自动 dump 全线程栈并退出
faulthandler.dump_traceback_later(120, exit=True)

from galbot_sdk.s1 import (ControlStatus, GalbotNavigation, GalbotRobot,
                           S1ControllerName)

DIST = 1.0
OMNI = False
SPEED = 0.2
ACC = None
JERK = None
ABS_XY = None
ABS_YAW = None
DRY = False
TIMEOUT = 40.0

args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--omni":
        OMNI = True
        i += 1
    elif a == "--dry-run":
        DRY = True
        i += 1
    elif a == "--speed":
        SPEED = float(args[i + 1]); i += 2
    elif a == "--acc":
        ACC = float(args[i + 1]); i += 2
    elif a == "--jerk":
        JERK = float(args[i + 1]); i += 2
    elif a == "--timeout":
        TIMEOUT = float(args[i + 1]); i += 2
    elif a == "--pos":
        ABS_XY = (float(args[i + 1]), float(args[i + 2])); i += 3
    elif a == "--yaw":
        ABS_YAW = math.radians(float(args[i + 1])); i += 2
    elif a in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    else:
        DIST = float(a)
        i += 1

robot = None
nav = None


def yaw_of(pose):
    qx, qy, qz, qw = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def pose_str(p):
    return 'x=%+.3f y=%+.3f yaw=%+.2f°' % (float(p[0]), float(p[1]), math.degrees(yaw_of(p)))


def main():
    global robot, nav
    robot = GalbotRobot()
    if not robot.init():
        print('[init] robot.init 失败')
        return 2
    nav = GalbotNavigation()
    if not nav.init():
        print('[init] nav.init 失败')
        return 2
    time.sleep(1.5)

    # 官方顺序：只切 POSE 控制器，不调用 start_controller / acquire_controller（会夺走 WBC 执行权）
    st = robot.switch_controller(S1ControllerName.SWERVE_CHASSIS_POSE_CTRL)
    print('[ctrl] switch_controller(SWERVE_CHASSIS_POSE_CTRL) →', st)
    if st != ControlStatus.SUCCESS:
        print('[ctrl] 控制器切换失败，中止')
        return 5

    ok_loc = False
    for _ in range(20):
        try:
            if nav.is_localized():
                ok_loc = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    print('[loc] is_localized →', ok_loc)
    if not ok_loc:
        print('[loc] 未定位，中止（先在应用侧或 tools 侧完成初始定位）')
        return 2

    p0 = nav.get_current_pose()
    yaw0 = yaw_of(p0)
    print('[pose] 起点 ' + pose_str(p0))

    if ACC is not None or JERK is not None:
        vel = np.array([SPEED, SPEED, min(0.5, SPEED * 1.5)], dtype=np.float64)
        acc = np.array([ACC if ACC is not None else 2.0] * 3, dtype=np.float64)
        jrk = np.array([JERK if JERK is not None else 5.0] * 3, dtype=np.float64)
        r = nav.set_navigation_kinematics_limits(vel, acc, jrk)
        print('[kin] set_navigation_kinematics_limits vel=%s acc=%s jerk=%s → %s'
              % (vel.round(3).tolist(), acc.round(3).tolist(), jrk.round(3).tolist(), r))

    if ABS_XY is not None:
        gx, gy = ABS_XY
        gyaw = ABS_YAW if ABS_YAW is not None else yaw0
        goal = np.array([gx, gy, 0.0, 0.0, 0.0, math.sin(gyaw / 2.0), math.cos(gyaw / 2.0)],
                        dtype=np.float64)
        mode = 'abs'
        print('[goal] 绝对目标 x=%+.3f y=%+.3f yaw=%+.2f°' % (gx, gy, math.degrees(gyaw)))
    else:
        goal = np.array([DIST, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        mode = 'rel'
        gx = float(p0[0]) + DIST * math.cos(yaw0)
        gy = float(p0[1]) + DIST * math.sin(yaw0)
        direction = '前进（车头朝向）' if DIST >= 0 else '倒车（车尾方向）'
        print('[dir] 模式：相对位移  方向：%s  距离 %.2f m'
              % (direction, abs(DIST)))
        print('[goal] base_link +x 方向 %+.3f m → map 约 (%+.3f, %+.3f)'
              % (DIST, gx, gy))

    tgt = np.array([gx, gy, 0.0, 0.0, 0.0, math.sin(yaw0 / 2.0), math.cos(yaw0 / 2.0)],
                   dtype=np.float64)
    try:
        reach = bool(nav.check_path_reachability(tgt, nav.get_current_pose()))
    except Exception as e:
        reach = None
        print('[reach] 查询异常：%s' % e)
    print('[reach] check_path_reachability →', reach)

    if DRY:
        print('[dry-run] 不下发运动')
        return 0
    if reach is False:
        print('[reach] pns 判不可达，不下发')
        return 3

    max_vel = np.array([SPEED, SPEED, min(0.5, SPEED * 1.5)], dtype=np.float64)
    print('[nav] 下发 omni_plan=%s max_vel=%s timeout=%.0fs'
          % (OMNI, max_vel.round(3).tolist(), TIMEOUT))
    t0 = time.time()
    if mode == 'rel':
        res = nav.navigate_to_goal_v2(goal, max_vel, pose_frame='base_link',
                                      enable_collision_check=True, is_blocking=False,
                                      timeout=TIMEOUT, omni_plan=OMNI)
    else:
        res = nav.navigate_to_goal(goal, enable_collision_check=True, is_blocking=False,
                                   timeout=TIMEOUT, omni_plan=OMNI)
    print('[nav] 返回', res)

    # 非阻塞下发后自行监控：pns 到位后停住，check_goal_arrival 在非阻塞模式下不刷新，
    # 故以“位移平台 + 目标残差”为判据，check_goal_arrival 仅作辅助。
    best = 0.0
    still = 0.0
    stopped = False
    arrived_sdk = False
    last_print = 0.0
    while time.time() - t0 < TIMEOUT:
        time.sleep(0.4)
        try:
            p = nav.get_current_pose()
        except Exception:
            continue
        el = time.time() - t0
        d = math.hypot(float(p[0]) - float(p0[0]), float(p[1]) - float(p0[1]))
        if d > best + 0.004:
            best = d
            still = 0.0
        else:
            still += 0.4
        if el - last_print >= 1.0:
            last_print = el
            print('  [%4.1fs] 位移 %.3f m  %s' % (el, d, pose_str(p)), flush=True)
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
        print('[pose] 终点 ' + pose_str(p1))
        print('[pose] 净位移 %.3f m  用时 %.1f s  平均 %.3f m/s'
              % (net, el, net / max(el, 1e-3)))
        if ABS_XY is not None:
            resid = math.hypot(float(p1[0]) - ABS_XY[0], float(p1[1]) - ABS_XY[1])
        else:
            resid = abs(net - abs(DIST))
        print('[nav] 目标残差 %.3f m（到位判据 ≤ 0.06）  平台判据 stopped=%s' % (resid, stopped))
        arrived = arrived_sdk or (stopped and resid <= 0.06)
    else:
        arrived = arrived_sdk
    try:
        print('[nav] check_goal_arrival →', nav.check_goal_arrival(), '（非阻塞模式下可能不刷新）')
    except Exception as e:
        print('[nav] arrival 查询异常：%s' % e)
    print('[nav] 到位' if arrived else '[nav] 未确认到位')
    return 0 if arrived else 4


rc = 4
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
        print('[exit] stop_navigation 异常（忽略）：%s' % e)
    try:
        if robot is not None:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()
            print('[exit] 优雅三连完成')
    except Exception as e:
        print('[exit] 关闭异常（忽略）：%s' % e)
print('[exit] rc=%d' % rc, flush=True)
os._exit(rc)
