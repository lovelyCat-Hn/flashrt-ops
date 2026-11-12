#!/usr/bin/env python
"""start_controller FAULT 归因探针（2026-11-12，零运动）。

背景：编排器干跑中 worker 任务前奏 start_controller('all') → FAULT。
嫌疑=WBCS 控制器优先级（.pyi switch_controller 文档串）：底盘被 pns 的
BYPASS 控制器（swerve_chassis_pose_ctrl）占用，'all' 含 swerve_chassis 组
→ 抢不动 → FAULT。10-29/30 traj 成功时本机尚无 pns 栈=无此冲突。

本探针：单 SDK 会话，快照逐组活动控制器 → start_controller('all') 复现 →
若 FAULT 逐组 start 定位挡路组 → 只 stop 自己成功 start 过的组（绝不碰
swerve_chassis，保 pns 通路）→ 优雅三连。全程无任何运动指令下发。

用法：
  cd ~/holy/scripts/s1/mission
  PYTHONPATH=/userdata/update/manual_update/lib \\
      ~/miniforge3/envs/flash_pyrt311/bin/python probe_ctrl_state.py
"""
import sys

sys.path.insert(0, "/userdata/update/manual_update/lib")
from galbot_sdk.s1 import (ControlStatus, GalbotRobot, S1ControllerName,
                           S1JointGroup)

GROUPS = [S1JointGroup.head, S1JointGroup.torso,
          S1JointGroup.left_arm, S1JointGroup.right_arm,
          S1JointGroup.left_gripper, S1JointGroup.right_gripper,
          S1JointGroup.swerve_chassis]
CHASSIS = S1JointGroup.swerve_chassis

robot = GalbotRobot()
started = []   # 本会话成功 start 过的组（还原时逐个 stop）
try:
    if not robot.init():
        print("[probe] robot.init 失败")
        sys.exit(2)
    print("── 快照①（本 SDK 实例已知活动控制器）──")
    for g in GROUPS:
        print(f"  {g:15s} → {robot.get_active_controller(g)}")

    print("── start_controller('all') ──")
    st = robot.start_controller("all")
    print(f"  → {st}")
    if st == ControlStatus.SUCCESS:
        started.append("all")
    else:
        print("── 'all' FAULT 复现 → 逐组 start 定位挡路组 ──")
        for g in GROUPS:
            st2 = robot.start_controller(g)
            print(f"  {g:15s} → {st2}")
            if st2 == ControlStatus.SUCCESS:
                started.append(g)

    print("── 快照② ──")
    for g in GROUPS:
        print(f"  {g:15s} → {robot.get_active_controller(g)}")
finally:
    print("── 还原（只动本会话 start 过的组；swerve_chassis 永不碰）──")
    for g in started:
        try:
            print(f"  stop({g}) → {robot.stop_controller(g)}")
        except Exception as e:   # noqa: BLE001
            print(f"  stop({g}) 异常: {e}")
    if CHASSIS in started:
        try:
            st3 = robot.switch_controller(S1ControllerName.SWERVE_CHASSIS_POSE_CTRL)
            print(f"  switch(POSE) → {st3}")
        except Exception as e:   # noqa: BLE001
            print(f"  switch 异常: {e}")
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("── 优雅三连完成 ──")
