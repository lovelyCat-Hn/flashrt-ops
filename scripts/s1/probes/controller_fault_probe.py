"""控制器 FAULT 定位探针（零运动，2026-11-10）。

背景：run_s1_loop_traj 干跑 `start_controller('all')` → ControlStatus.FAULT
两连复现（11-02/03 同机全 SUCCESS）。急停已由用户现场排除。主嫌疑=WBCS
权限状态：switch_controller 官方文档——高优先级 BYPASS 激活时 PVT 控制器
无法直接 start（返回即 FAULT）；恢复序 = stop_controller → release_controller
→ acquire_controller(pvt) → start_controller（该恢复序不在本探针内——可能
剥他人遥控权限，须用户确认无人遥控后再执行）。

动作面 = start/stop_controller（只切控制器激活态，无任何运动指令，与执行器
每次启动的调用同类）；读面 = get_joint_positions / get_gripper_state /
get_active_controller（注意：get_active_controller 返回的是本 SDK 实例
最后已知值，非 WBCS 实时查询——弱证据，仅参考）。

用法：
  PYTHONPATH=/data/galbot/lib flash_pyrt311 python3 controller_fault_probe.py
"""
from __future__ import annotations

import sys
import time

from galbot_sdk.s1 import (
    GalbotRobot, RgbOutputFormat, SensorType)

GROUPS = ["left_arm", "right_arm", "torso", "head",
          "right_gripper", "left_gripper"]
ARM = ([f"right_arm_joint{i}" for i in range(1, 8)]
       + [f"left_arm_joint{i}" for i in range(1, 8)])
TB_HEAD = ["torso_lift_joint1", "head_joint1", "head_joint2"]


def fmt(st) -> str:
    return str(st)


def main():
    robot = GalbotRobot()
    ok = False
    try:
        print("== init（空传感器集，最小化）==")
        if not robot.init(set()):
            print("init False → 总线/中间件层就挂了，别往下查控制器")
            return 2
        print("init OK")

        print("\n== 读面：总线活性 ==")
        names = robot.get_joint_names()
        print(f"joint_names({len(names) if names else 0}): {names}")
        vals = robot.get_joint_positions(joint_names=ARM)
        print(f"臂位({len(vals) if vals else 0} 维): "
              f"{[round(v, 3) for v in vals] if vals else vals}")
        vals2 = robot.get_joint_positions(joint_names=TB_HEAD)
        print(f"torso/head: "
              f"{[round(v, 3) for v in vals2] if vals2 else vals2}")
        for g in ("right_gripper", "left_gripper"):
            try:
                gs = robot.get_gripper_state(g)
                print(f"{g}: {gs}")
            except Exception as e:   # noqa: BLE001
                print(f"{g}: 读异常 {type(e).__name__}: {e}")

        print("\n== get_active_controller（SDK 实例本地已知值，弱证据）==")
        for g in GROUPS:
            try:
                print(f"  {g}: {robot.get_active_controller(g)!r}")
            except Exception as e:   # noqa: BLE001
                print(f"  {g}: 异常 {type(e).__name__}: {e}")

        print("\n== 基线：start_controller('all')（本实例首次）==")
        st = robot.start_controller("all")
        print(f"  all → {fmt(st)}")

        print("\n== 逐组 start（定位哪个组 FAULT；INVALID_INPUT=组名不存在）==")
        for g in GROUPS:
            st = robot.start_controller(g)
            print(f"  {g} → {fmt(st)}")

        print("\n== 二次 all（看 FAULT 是否粘滞/部分组已活）==")
        st = robot.start_controller("all")
        print(f"  all(2nd) → {fmt(st)}")

        print("\n== 结论判读 ==")
        print("  全组 FAULT → 全局权限状态（WBCS BYPASS 占用/控制柜状态）")
        print("  个别组 FAULT → 该组专属问题（该组控制器/该组硬件）")
        print("  INVALID_INPUT → 组名面不对（看 get_joint_names 实名）")
        return 0
    finally:
        # SDK 优雅三连（10-28 教训：缺=退出段 Segmentation fault）
        try:
            robot.request_shutdown()
            robot.wait_for_shutdown()
        except Exception as e:   # noqa: BLE001
            print(f"shutdown 异常: {e}")
        robot.destroy()


if __name__ == "__main__":
    sys.exit(main())
