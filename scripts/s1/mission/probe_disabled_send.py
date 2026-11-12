#!/usr/bin/env python
"""DISABLED 组 set_joint_commands 行为实证（2026-11-12，微目标零风险）。

问题：臂组控制器被软急停锁存（wbcs_test ca=DISABLED/error_code 0），
用户问「PVT 本来就行，直接把推理结果 set_joint_commands 下发不就好了？」
——若 DISABLED 不拦下发路径，则无需 soft_resumec；若拦，则解锁是唯一路。

方法：往 DISABLED 的左臂发一包微目标（j1 +2 mrad，噪声底之上、肉眼不可
见），三判据=回执 ControlStatus / 2s 后 state 真值 / 轨迹执行状态查询。
本机前科：13:26 底盘「回执 SUCCESS 位移≈0」——判据必须用 state 真值。

用法：
  cd ~/holy/scripts/s1/mission
  PYTHONPATH=/userdata/update/manual_update/lib \\
      ~/miniforge3/envs/flash_pyrt311/bin/python probe_disabled_send.py
"""
import sys
import time

sys.path.insert(0, "/userdata/update/manual_update/lib")
from galbot_sdk.s1 import GalbotRobot, JointCommand

ARM = [f"left_arm_joint{i}" for i in range(1, 8)]
DELTA_J1 = 0.002   # rad，微目标

robot = GalbotRobot()
try:
    if not robot.init():
        print("[probe] robot.init 失败")
        sys.exit(2)

    def read_arm(tag):
        p = robot.get_joint_positions(joint_names=ARM)
        print(f"  [{tag}] " + " ".join(f"{v:+.4f}" for v in (p or [])))
        return p

    print("── 基线（连读 3 次看噪声底）──")
    for k in range(3):
        read_arm(f"基线{k}")
        time.sleep(0.4)

    before = robot.get_joint_positions(joint_names=ARM)
    if not before or len(before) != 7:
        print("[probe] 左臂读数异常")
        sys.exit(2)

    cmds = []
    for i, pos in enumerate(before):
        c = JointCommand()   # 无 joint_name 字段（pyi 残缺）：身份靠
        c.position = float(pos) + (DELTA_J1 if i == 0 else 0.0)
        cmds.append(c)       # joint_names 参数位置对应（引擎同款）
    print(f"── 下发微目标（{ARM[0]} +{DELTA_J1*1000:.0f} mrad，tfs=0.033s）──")
    st = robot.set_joint_commands(cmds, joint_names=ARM, time_from_start_s=0.033)
    print(f"  回执 → {st}")

    time.sleep(2.0)
    print("── 下发后 state ──")
    after = read_arm("2s后")
    if after and len(after) == 7:
        d = abs(after[0] - before[0])
        print(f"  j1 位移={d*1000:.2f} mrad（目标 2.0）"
              f" → {'✅ 执行了' if d > 0.0015 else '⛔ 未执行（DISABLED 拦住下发面）'}")

    try:
        ts = robot.check_trajectory_execution_status(["left_arm"])
        print(f"── 轨迹执行状态：{ts} ──")
    except Exception as e:   # noqa: BLE001
        print(f"  执行状态查询异常: {e}")
finally:
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("── 优雅三连完成 ──")
