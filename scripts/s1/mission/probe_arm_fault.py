#!/usr/bin/env python
"""臂组 start_controller FAULT 归因探针 v2（2026-11-12，零运动）。

v1 结论：'all' FAULT 挡路组=left_arm/right_arm（head/torso/双爪/底盘全
SUCCESS）；WBCS 文件日志 0 字节。v2 改走 SDK 日志通道
get_log_information（文件日志走 SDK 服务不落盘）+ 机体读数 + acquire 阶梯。

用法：
  cd ~/holy/scripts/s1/mission
  PYTHONPATH=/userdata/update/manual_update/lib \\
      ~/miniforge3/envs/flash_pyrt311/bin/python probe_arm_fault.py
"""
import sys

sys.path.insert(0, "/userdata/update/manual_update/lib")
from galbot_sdk.s1 import (ControlStatus, GalbotRobot, LogLevel,
                           S1ControllerName, S1JointGroup)

ARM_JOINTS = [f"{side}_arm_joint{i}" for side in ("left", "right")
              for i in range(1, 8)]
GRIP_JOINTS = ["left_gripper_joint1", "right_gripper_joint1"]

robot = GalbotRobot()
started = []
try:
    if not robot.init():
        print("[probe] robot.init 失败")
        sys.exit(2)

    # ── ① WBCS 日志回看（FAULT 真因主通道）──
    for lv, name in ((LogLevel.ERROR, "ERROR"), (LogLevel.WARN, "WARN")):
        try:
            info = robot.get_log_information(3600, lv)
            msg = info.get("message") or ""
            if isinstance(msg, list):
                msg = "\n".join(str(m) for m in msg)
            lines = [ln for ln in str(msg).splitlines() if ln.strip()]
            print(f"── 日志[{name}] 近 1h：{len(lines)} 行 ──")
            for ln in lines[-30:]:
                print(f"  {ln}")
        except Exception as e:   # noqa: BLE001
            print(f"  日志[{name}] 拉取异常: {e}")

    # ── ② 双臂关节态（current=力矩代理：≈0=未被持有/掉臂风险面）──
    try:
        sts = robot.get_joint_states(joint_names=ARM_JOINTS)
        print(f"── 双臂关节态：{len(sts)} 个 ──")
        for js in sts:
            print(f"  {js.joint_name:22s} pos={js.position:+.3f} "
                  f"cur={js.current:+.3f} eff={js.effort:+.3f}")
    except Exception as e:   # noqa: BLE001
        print(f"  双臂读数异常: {e}")

    # ── ③ 双爪回读位（行程 0.007-0.11m；0/None/异常=疑似不在位）──
    try:
        gp = robot.get_joint_positions(joint_names=GRIP_JOINTS)
        print(f"── 双爪回读：{gp} ──")
    except Exception as e:   # noqa: BLE001
        print(f"  双爪读数异常: {e}")

    # ── ④ acquire 阶梯（start 基线 → acquire PVT → start 重试）──
    for side, grp, pvt in (("left", S1JointGroup.left_arm,
                            S1ControllerName.LEFT_ARM_PVT_CTRL),
                           ("right", S1JointGroup.right_arm,
                            S1ControllerName.RIGHT_ARM_PVT_CTRL)):
        st1 = robot.start_controller(grp)
        print(f"[{side}] start({grp}) → {st1}")
        if st1 == ControlStatus.SUCCESS:
            started.append(grp)
            continue
        st2 = robot.acquire_controller(pvt)
        print(f"[{side}] acquire({pvt}) → {st2}")
        st3 = robot.start_controller(grp)
        print(f"[{side}] start 重试({grp}) → {st3}")
        if st3 == ControlStatus.SUCCESS:
            started.append(grp)

    print("── 快照 ──")
    for g in (S1JointGroup.left_arm, S1JointGroup.right_arm):
        print(f"  {g:15s} → {robot.get_active_controller(g)}")
finally:
    print("── 还原（只 stop 本会话 start 成功过的组）──")
    for g in started:
        try:
            print(f"  stop({g}) → {robot.stop_controller(g)}")
        except Exception as e:   # noqa: BLE001
            print(f"  stop({g}) 异常: {e}")
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("── 优雅三连完成 ──")
