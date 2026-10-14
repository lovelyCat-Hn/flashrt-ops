#!/usr/bin/env python
"""夹爪物理速度标定台（零臂运动，只动单爪）——判别「爪开的慢」三选一：

  A. SDK 调用阻塞      → set_gripper_command 返回耗时（生产 is_blocking=False，
                          正常应 <10ms；阻塞会拖死控制环）
  B. 闭环命令斜坡慢    → 与本台无关（本台单步指令），但对照闭环日志可判：
                          182005 跑命令侧 2.5s 拉满 1→103mm，不是它
  C. 伺服实际速度不足  → 单指令 0→100% 行程的 T90 实测 vs width_span/speed 理论值

顺带直测回读滞后：GripperState.timestamp_ns 与墙钟差（10-05 间接测得 6.3-8s，
本次带时间戳直接量）。产物落 evidence/grip_speed_bench/。

用法（真机；确认爪行程内无障碍、人不扶爪）:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \\
  ~/miniforge3/envs/flash_pyrt311/bin/python \\
      ~/holy/scripts/test/grip_speed_bench.py --side left --speed 0.15

可选: --side right | --speed 0.03（SDK 默认速对照）| --cycles 2 | --effort 30
安全: 只发夹爪指令，臂/腿/头维度永不触碰；Ctrl-C 随时中止。
"""
import argparse
import atexit
import csv
import json
import pathlib
import time

from galbot_sdk.g1 import GalbotRobot, G1JointGroup

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_onlypick_deploy")
ap.add_argument("--side", choices=["left", "right"], default="left")
ap.add_argument("--speed", type=float, default=0.15, help="m/s，生产值 0.15（SDK 默认 0.03）")
ap.add_argument("--effort", type=float, default=30)
ap.add_argument("--cycles", type=int, default=2, help="开+合 循环次数")
ap.add_argument("--poll-hz", type=float, default=20.0)
ap.add_argument("--stroke-timeout", type=float, default=6.0, help="单程观测窗 s")
args = ap.parse_args()

mf_p = pathlib.Path(args.ckpt) / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
_g = mf.get("gripper", {})
WMIN, WMAX = _g.get("width_min"), _g.get("width_max")
if WMIN is None or WMAX is None:
    raise SystemExit("manifest 无夹爪标定（width_min/width_max）——先跑 g1_ckpt_prep.py")
NAME = f"{args.side}_gripper"
GROUP = getattr(G1JointGroup, NAME)

print("== 夹爪速度标定台（零臂运动）==")
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(3)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")
if not str(st).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止")

gs0 = robot.get_gripper_state(GROUP)
w0 = gs0.width if gs0 is not None else float("nan")
print(f"起始宽度: {w0*1000:.1f} mm | 目标: 开 {WMAX*1000:.0f} / 合 {WMIN*1000:.0f} mm"
      f" | 速度帽 {args.speed} m/s → 理论全程 {abs(WMAX-WMIN)/args.speed:.2f} s")
input("确认爪行程内无障碍、急停可及，回车开始（Ctrl-C 中止）> ")

out_d = pathlib.Path("/home/galbot/holy/evidence/grip_speed_bench")
out_d.mkdir(parents=True, exist_ok=True)
tag = time.strftime("%Y%m%d-%H%M%S")
rows = []
t_first: dict = {}
input("再次确认，回车开始 > ")


def poll_until(target, label):
    """发单指令并 20Hz 轮询回读，直到到位/超窗。返回 (call_ms, 样本数)。"""
    t0 = time.perf_counter()
    tc = time.perf_counter()
    stc = robot.set_gripper_command(GROUP, target, args.speed, args.effort, False)
    call_ms = (time.perf_counter() - tc) * 1000.0
    n = 0
    while True:
        tel = time.perf_counter() - t0
        gs = robot.get_gripper_state(GROUP)
        if gs is not None:
            ts_age = (time.time_ns() - gs.timestamp_ns) / 1e9 if getattr(gs, "timestamp_ns", 0) else float("nan")
            rows.append([label, round(tel, 3), round(gs.width * 1000, 2),
                         round(gs.velocity, 4), int(gs.is_moving), round(ts_age, 3),
                         str(stc).replace("ControlStatus.", "")])
            n += 1
        if tel >= args.stroke_timeout:
            break
        time.sleep(1.0 / args.poll_hz)
    return call_ms, n


print("\n─ 每程结果 ─")
for c in range(args.cycles):
    for label, target in (("open", WMAX), ("close", WMIN)):
        call_ms, n = poll_until(target, f"c{c}_{label}")
        mine = [r for r in rows if r[0] == f"c{c}_{label}"]
        span = abs(target - w0) if c == 0 and label == "open" else abs(WMAX - WMIN)
        t90 = next((r[1] for r in mine if abs(r[2] - target * 1000) <= 0.1 * span * 1000), None)
        v_pk = max((abs(r[3]) for r in mine), default=0.0)
        ts_max = max((r[5] for r in mine), default=float("nan"))
        t_first[f"c{c}_{label}"] = t90
        print(f"  {label:>5}: 指令调用 {call_ms:5.1f} ms（>10=疑似阻塞）| "
              f"到 90% 行程: {'超窗未到' if t90 is None else f'{t90:.2f} s'} | "
              f"理论 {span/args.speed:.2f} s | 回读峰值速 {v_pk:.3f} m/s | "
              f"回读时戳滞后 max {ts_max:.2f} s")

out_p = out_d / f"{tag}_{args.side}_spd{args.speed}.csv"
with out_p.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stroke", "t_rel_s", "width_mm", "velocity_mps", "is_moving",
                "state_ts_age_s", "cmd_status"])
    w.writerows(rows)
print(f"\n曲线已落盘: {out_p}")
print("判读: T90≈理论值→伺服守约，「慢」=闭环命令斜坡（模型侧）；"
      "T90≫理论值→伺服实际速度不足（硬件/底层层）；指令调用>10ms→SDK 阻塞。")
atexit.register(lambda: None)
