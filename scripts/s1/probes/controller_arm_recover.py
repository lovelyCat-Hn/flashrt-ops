"""双臂 WBCS 权限恢复序（零运动，2026-11-10；用户已确认无人遥控）。

背景：controller_fault_probe.py 实证 torso/head/双爪 start SUCCESS、
双臂 start FAULT——符合 WBCS 高优先级 BYPASS 残留（遥控/HMI 会话占用后
未还权）特征。恢复序 = switch_controller 官方文档原文：
  stop_controller(group) → release_controller(group)
  → acquire_controller(<pvt>) → start_controller(group)

安全面：全序无任何运动指令；前后 get_joint_positions 零运动断言
（|Δ|<0.01 rad，违=外部在动，立即停手报告）。

用法：
  PYTHONPATH=/data/galbot/lib flash_pyrt311 python3 controller_arm_recover.py
"""
from __future__ import annotations

import sys

from galbot_sdk.s1 import GalbotRobot

GROUPS = ["left_arm", "right_arm"]
PVT = {"left_arm": "left_arm_pvt_ctrl", "right_arm": "right_arm_pvt_ctrl"}
ARM = ([f"right_arm_joint{i}" for i in range(1, 8)]
       + [f"left_arm_joint{i}" for i in range(1, 8)])
TOL = 0.01   # rad——零运动断言阈（权限切换不该动一根手指）


def read_arm(robot):
    v = robot.get_joint_positions(joint_names=ARM)
    return [round(x, 4) for x in v] if v else None


def main():
    robot = GalbotRobot()
    try:
        if not robot.init(set()):
            print("init False → 终止")
            return 2
        before = read_arm(robot)
        print(f"臂位 before: {before}")

        for g in GROUPS:
            print(f"\n== {g} 恢复序（stop→release→acquire({PVT[g]})→start）==")
            for op, st in [
                ("stop", robot.stop_controller(g)),
                ("release", robot.release_controller(g)),
                ("acquire", robot.acquire_controller(PVT[g])),
                ("start", robot.start_controller(g)),
            ]:
                print(f"  {op:8s} → {st}")
                if "FAULT" in str(st) and op == "acquire":
                    print("  ⚠ acquire FAULT——继续观察 start（可能硬件层 fault）")

        after = read_arm(robot)
        print(f"\n臂位 after:  {after}")
        if before and after:
            d = max(abs(a - b) for a, b in zip(before, after))
            print(f"零运动断言: max|Δ|={d:.4f} rad（阈 {TOL}）→ "
                  + ("PASS 无运动" if d < TOL else "⛔ 违例！有外部运动，停手排查"))

        print("\n== 终验：start_controller('all') ==")
        st = robot.start_controller("all")
        print(f"  all → {st}")
        print("\nSUCCESS=破案（BYPASS 残留实锤）；仍 FAULT=臂组硬件/更深状态，"
              "升级控制柜重启或供应商工具")
        return 0 if "SUCCESS" in str(st) else 1
    finally:
        try:
            robot.request_shutdown()
            robot.wait_for_shutdown()
        except Exception as e:   # noqa: BLE001
            print(f"shutdown 异常: {e}")
        robot.destroy()


if __name__ == "__main__":
    sys.exit(main())
