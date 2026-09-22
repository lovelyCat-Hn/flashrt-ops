#!/usr/bin/env python
"""PVT 轨迹接口形态探针——⚠⚠ 已封存，勿在真机重跑（2026-09-22 事故）⚠⚠

实测后果：变体 C（velocity=0.5 前馈）与 E/F（致 PVT 控制器 FAULT）引发
机械臂剧烈抖动一次。"零运动目标"不等于零风险——PVT 的速度前馈和故障
转换都会真实作用于关节。轨迹路线（主脚本 --chunk-mode traj / 本探针）
在拿到厂商接口文档或台架验证条件前，一律不要再上真机。

留档结论：A-D 变体（joint_names14/groups + position，±velocity）返回
SUCCESS 但 2 点轨迹；主脚本 4 点轨迹同形态却 INVALID_INPUT，原因未明；
E 单组、F 单点 → FAULT。

用法（封存，仅留档）:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/g1_traj_probe.py
"""
import os
import sys
import time

sys.path.insert(0, "/home/galbot/holy/FlashRT")

from galbot_sdk.g1 import (  # noqa: E402
    GalbotRobot, Trajectory, TrajectoryPoint, JointCommand)

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT   # 与主脚本同序


def mk_point(positions, t):
    tp = TrajectoryPoint()
    tp.time_from_start_second = t
    vec = []
    for v in positions:
        c = JointCommand()
        c.position = float(v)
        vec.append(c)
    tp.joint_command_vec = vec
    return tp


def main():
    robot = GalbotRobot()
    if not robot.init():
        raise SystemExit("robot.init 失败")
    time.sleep(5)
    print(f"start_controller('all') → {robot.start_controller('all')}")
    for g in ("left_arm", "right_arm"):
        try:
            print(f"活跃控制器[{g}] → {robot.get_active_controller(g)}")
        except Exception as e:
            print(f"活跃控制器[{g}] 查询失败: {e}")

    cur = robot.get_joint_positions([], ARM_NAMES)
    if not cur or len(cur) != 14:
        raise SystemExit(f"关节读取失败: {len(cur) if cur else 0}")
    cur = [float(v) for v in cur]
    print(f"当前读数（所有变体目标=当前，零运动）: "
          f"{[round(v, 3) for v in cur]}\n")

    def traj_full(joint_names, groups, positions, times, vel=None, acc=0.0):
        tj = Trajectory()
        if joint_names:
            tj.joint_names = joint_names
        if groups:
            tj.joint_groups = groups
        pts = []
        for i, t in enumerate(times):
            tp = TrajectoryPoint()
            tp.time_from_start_second = t
            vec = []
            for v in positions:
                c = JointCommand()
                c.position = float(v)
                if vel is not None:
                    c.velocity = float(vel)
                c.acceleration = float(acc)
                vec.append(c)
            tp.joint_command_vec = vec
            pts.append(tp)
        tj.points = pts
        return tj

    variants = [
        ("A 复现: joint_names14 + 仅position（主脚本现状）",
         lambda: traj_full(ARM_NAMES, [], cur, [0.1, 0.2])),
        ("B A+显式velocity=0/acceleration=0",
         lambda: traj_full(ARM_NAMES, [], cur, [0.1, 0.2], vel=0.0)),
        ("C A+velocity=0.5（非零前馈）",
         lambda: traj_full(ARM_NAMES, [], cur, [0.1, 0.2], vel=0.5)),
        ("D joint_groups=[left,right] 14关节",
         lambda: traj_full([], ["left_arm", "right_arm"], cur, [0.1, 0.2])),
        ("E 单组 joint_groups=[left_arm] 7关节",
         lambda: traj_full([], ["left_arm"], cur[:7], [0.1, 0.2])),
        ("F 单点轨迹 joint_names14",
         lambda: traj_full(ARM_NAMES, [], cur, [0.1])),
    ]
    for desc, build in variants:
        try:
            tj = build()
            st = robot.execute_joint_trajectory(tj, is_blocking=False)
            time.sleep(0.4)
            ss = robot.check_trajectory_execution_status([])
            print(f"{desc}\n  → {st} | PVT 状态 "
                  f"{[s.name for s in ss] or '未上报'}")
        except Exception as e:
            print(f"{desc}\n  → 异常: {type(e).__name__}: {e}")

    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("\n探针完成（全程零运动）")


if __name__ == "__main__":
    main()
    os._exit(0)   # SDK 残留线程
