#!/usr/bin/env python
"""右臂控制器故障恢复：stop→start 断电重启清故障 → 小步试动 → 回工作位。

背景（2026-09-23 真机）：零位 TIMEOUT 后臂停中途，控制器使能保持状态下
被手动掰动 → 关节驱动层进位置误差/过流保护，新指令被拒（读数通道正常、
get_active_controller 仍显示 right_arm_pvt_ctrl——RT 层无告警，典型驱动
层故障）。恢复三步：
  ① stop_controller("right_arm")——臂会【失力变软】，先扶住！
  ② start_controller("right_arm")——重新使能，保持当前位
  ③ 小步试动（±0.1 rad 限幅）确认能动 → 全程回 state.mean 工作位
⚠ 全程急停就绪；③ 的直控无碰撞检查，臂前空间清空。

用法:
  ~/holy/run.sh ~/holy/scripts/inference/right_arm_recover.py
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, "/data/galbot/lib")
from galbot_sdk.g1 import GalbotRobot  # noqa: E402

CKPT = pathlib.Path("/home/galbot/holy/models/pi05_g1_ft")
ns = json.loads((CKPT / "norm_stats.json").read_text())
TARGET = ns["state"]["mean"][:7]        # 数据集序右臂 dim0-6
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]


def read_arm(robot):
    v = robot.get_joint_positions([], RIGHT)
    assert v and len(v) == 7, f"右臂读取失败: {v}"
    return [float(x) for x in v]


print("== 右臂恢复（急停就绪！）==", flush=True)
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(5)
cur = read_arm(robot)
print(f"右臂当前: {[round(v, 3) for v in cur]}", flush=True)
print(f"目标(工作位): {[round(v, 3) for v in TARGET]}", flush=True)
print(f"active: {robot.get_active_controller('right_arm')}", flush=True)

input("\n⚠ ① 即将 stop_controller('right_arm')——臂会【失力变软】，"
      "先伸手扶住右臂！就绪回车，Ctrl-C 退出...", flush=True)
st = robot.stop_controller("right_arm")
print(f"stop_controller → {st}", flush=True)
time.sleep(2)
st = robot.start_controller("right_arm")
print(f"start_controller → {st}", flush=True)
time.sleep(2)
print(f"active: {robot.get_active_controller('right_arm')}", flush=True)

cur = read_arm(robot)
test_t = [c + max(-0.1, min(0.1, t - c)) for c, t in zip(cur, TARGET)]
input(f"\n⚠ ② 小步试动到 {[round(v, 3) for v in test_t]}（±0.1 rad，0.1 rad/s）。"
      "臂前空间清空，就绪回车...", flush=True)
st = robot.set_joint_positions(test_t, joint_names=RIGHT, is_blocking=True,
                               speed_rad_s=0.1, timeout_s=15.0)
print(f"试动 → {st}", flush=True)
time.sleep(0.5)
now = read_arm(robot)
moved = max(abs(n - c) for n, c in zip(now, cur))
print(f"实际移动 {moved * 1000:.0f} mrad → "
      + ("✅ 臂活了" if moved > 0.01 else "❌ 仍拒动（查急停/RT，或整机重启控制）"),
      flush=True)
if moved <= 0.01:
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    sys.stdout.flush(); os._exit(1)

input("\n⚠ ③ 回工作位（0.15 rad/s）。就绪回车...", flush=True)
st = robot.set_joint_positions(TARGET, joint_names=RIGHT, is_blocking=True,
                               speed_rad_s=0.15, timeout_s=45.0)
print(f"回位 → {st}", flush=True)
time.sleep(0.5)
fin = read_arm(robot)
err = max(abs(f - t) for f, t in zip(fin, TARGET))
print(f"最终偏差 {err * 1000:.1f} mrad（<20 视为到位）→ "
      + ("✅ 恢复完成，可重跑预热（建议 --skip-zero）" if err < 0.02
         else "⚠ 未到位，查上方状态"), flush=True)
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
sys.stdout.flush(); os._exit(0)
