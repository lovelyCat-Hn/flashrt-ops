#!/usr/bin/env python
"""夹爪真机标定：全闭/全开行程 → width_min / width_max（喂 g1_ckpt_prep --grip-wmin/--grip-wmax）。

流程（左右各一遍，先左后右）：
  ① 只读 2s：确认 SDK 数据在流（width/velocity/effort/is_moving）
  ② 命令闭合到 0（机构限位兜底）→ 到位稳定后采样 width → width_min（零开度）
  ③ 命令张开到超程 0.25（夹爪自身限位封顶）→ 稳定后采样 → width_max（满开度）
保守参数：速度 0.02 m/s、力矩 10 N；非阻塞命令+轮询，Ctrl-C 随时中断。
结束时夹爪停在张开位。急停/RT 异常表现为读取 None（处置见 DEPLOY.md #11 同源问题）。

用法（现场安全确认后）:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
  ~/holy/scripts/inference/gripper_calib.py [--side both|left|right] [--read-only]
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, "/data/galbot/lib")
from galbot_sdk.g1 import GalbotRobot, G1JointGroup, ControlStatus  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--side", choices=("both", "left", "right"), default="both")
ap.add_argument("--read-only", action="store_true", help="只读不动作")
ap.add_argument("--speed", type=float, default=0.02)
ap.add_argument("--effort", type=float, default=10.0)
ap.add_argument("--open-target", type=float, default=0.25, help="满开探测目标（超程由限位封顶）")
args = ap.parse_args()

SIDES = {"left": ["left_gripper"], "right": ["right_gripper"],
         "both": ["left_gripper", "right_gripper"]}[args.side]


def read(robot, side):
    st = robot.get_gripper_state(getattr(G1JointGroup, side))
    if st is None:
        print(f"  ❌ {side}: get_gripper_state 返回 None（查 RT/急停/使能）", flush=True)
    return st


def sample(robot, side, n=5, dt=0.2):
    """连采 n 次，返回 (中位 width, 极差)。任一次 None 返回 None。"""
    ws = []
    for _ in range(n):
        st = read(robot, side)
        if st is None:
            return None, None
        ws.append(st.width)
        time.sleep(dt)
    return statistics.median(ws), max(ws) - min(ws)


def goto(robot, side, target, timeout=30.0, w_start=None):
    """非阻塞命令 + 轮询到位；返回稳定采样。w_start 用于反馈活性校验。"""
    status = robot.set_gripper_command(getattr(G1JointGroup, side),
                                       target, args.speed, args.effort, False)
    if status != ControlStatus.SUCCESS:
        print(f"  ❌ {side}: set_gripper_command({target}) → {status}", flush=True)
        return None, None
    t0 = time.time()
    moved = False
    while time.time() - t0 < timeout:
        st = read(robot, side)
        if st is None:
            time.sleep(0.1)
            continue
        if w_start is not None and abs(st.width - w_start) > 2e-3:
            moved = True                 # 反馈在动——活性确认
        if not st.is_moving and moved:
            break
        if not st.is_moving and not moved and time.time() - t0 > 8.0:
            # 实测：反馈冷启动/滞后可达 ~6.3s（2026-09-23 标定实锤），8s 内不动才算真冻结
            print(f"  ❌ {side}: 反馈冻结 >8s（width 恒 {st.width:.4f}，is_moving=False）——"
                  f"查 RT/急停或重启本进程重试", flush=True)
            return None, None
        time.sleep(0.1)
    else:
        print(f"  ⚠ {side}: {timeout}s 内 is_moving 未清零，按当前位采样", flush=True)
    time.sleep(0.5)                      # 机械稳态
    return sample(robot, side)


robot = GalbotRobot()
robot.init()
time.sleep(2)
print("✓ SDK 初始化完成，① 只读 2s 数据探针", flush=True)

for side in SIDES:
    print(f"\n== {side} ==", flush=True)
    st = read(robot, side)
    if st is None:
        continue
    print(f"  当前: width={st.width:.4f} velocity={st.velocity:.4f} "
          f"effort={st.effort:.2f} is_moving={st.is_moving}", flush=True)

    if args.read_only:
        w, spread = sample(robot, side)
        print(f"  静置采样: median={w:.4f} 极差={spread:.4f}", flush=True)
        continue

    print(f"  → 闭合探测（目标 0，速度 {args.speed}，力矩 {args.effort}N）...", flush=True)
    w_min, sp = goto(robot, side, 0.0, w_start=st.width)
    if w_min is None:
        continue
    print(f"  width_min = {w_min:.4f} m（采样极差 {sp:.4f}）", flush=True)

    print(f"  → 张开探测（目标 {args.open_target}）...", flush=True)
    w_max, sp = goto(robot, side, args.open_target, w_start=w_min)
    if w_max is None:
        continue
    print(f"  width_max = {w_max:.4f} m（采样极差 {sp:.4f}）", flush=True)

    if w_max - w_min < 0.005:
        print("  ⚠ 行程 <5mm，数值可疑——目视夹爪确认是否真的动了", flush=True)
    print(f"\n  >> prep 命令参数: --grip-wmin {w_min:.4f} --grip-wmax {w_max:.4f}", flush=True)

robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
sys.stdout.flush()
os._exit(0)                              # SDK 残留线程不退，强杀（终端行缓冲已刷）
