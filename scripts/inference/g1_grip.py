#!/usr/bin/env python
"""夹爪独立控制：手动摆放物品后夹上/松开（真机小工具，不动臂不进推理）。

背景：only_place 起步位=双臂 33% 持物开度——warmup 后夹爪是"空撑着"的，
手动把物品放进掌心夹不住。闭环里夹爪由模型 dim7/dim15 下发，起步前没人管；
本工具补这个空档：先 --close 夹上物品，再跑 run_g1_loop。

用法（现场安全确认后，走 run.sh 同一套环境）:
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py --close              # 双臂夹紧（力矩限位兜底，夹住即停）
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py --pct 33             # 回 33% 持物开度（松开换物/摆放）
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py --side left --pct 10 # 单臂微调
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py                      # 不给目标=只读，打印当前开度

标定宽度从部署目录 flashrt_deploy.json 读（与闭环 send_grip 同源，
缺 manifest 回退 2026-09-23 实测 0.0005/0.1200 m）。
⚠ SDK 夹爪反馈滞后可达 ~6.3s 且期间 is_moving 恒 False（标定实录）——
轮询按"宽度动过再停"判稳，8s 纹丝不动才报疑似夹住/冻结，超时 20s。
易碎品用 --effort 10 降夹力（默认取 config [gripper].effort=30N）。
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, "/home/galbot/holy/scripts/inference")
import g1_config  # noqa: E402  同目录共享配置

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--side", choices=("both", "left", "right"), default="both")
tgt = ap.add_mutually_exclusive_group()
tgt.add_argument("--close", action="store_true",
                 help="夹紧到 0%%（有物品时力矩限位兜底，夹住即停）")
tgt.add_argument("--pct", type=float, help="目标开度 %%（0~100；33=only_place 持物位）")
tgt.add_argument("--width", type=float, help="直接给 SDK 宽度（米，高级）")
ap.add_argument("--speed", type=float, default=None, help="夹爪速度 m/s（config [gripper].speed）")
ap.add_argument("--effort", type=float, default=None, help="夹爪力矩 N（config [gripper].effort）")
ap.add_argument("--timeout", type=float, default=20.0, help="到位轮询超时 s（反馈滞后 ~6.3s）")
args = ap.parse_args()

# ── 标定与默认值（CLI > config/g1.toml > BUILTIN/回退）──
cfg, _ = g1_config.load()                        # load 返回 (dict, path)
g_sec = cfg.get("gripper", {})
SPEED = args.speed if args.speed is not None else float(
    g_sec.get("speed", g1_config.BUILTIN["gripper"]["speed"]))
EFFORT = args.effort if args.effort is not None else float(
    g_sec.get("effort", g1_config.BUILTIN["gripper"]["effort"]))
ckpt = pathlib.Path(cfg.get("run", {}).get("ckpt") or g1_config.BUILTIN["run"]["ckpt"])
mf_p = ckpt / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
g_mf = mf.get("gripper", {})
WMIN = float(g_mf.get("width_min", 0.0005))
WMAX = float(g_mf.get("width_max", 0.1200))
SPAN = WMAX - WMIN
if SPAN <= 0:
    raise SystemExit(f"标定可疑: wmin={WMIN} >= wmax={WMAX}")

if args.close:
    PCT = 0.0
elif args.pct is not None:
    if not 0.0 <= args.pct <= 100.0:
        raise SystemExit(f"--pct 须在 0~100，收到 {args.pct}")
    PCT = args.pct
elif args.width is not None:
    PCT = (args.width - WMIN) / SPAN * 100.0
else:
    PCT = None                                  # 只读模式

def pct_of(w):
    return max(0.0, min(100.0, (w - WMIN) / SPAN * 100.0))

print(f"标定: 0%→{WMIN:.4f} m | 100%→{WMAX:.4f} m（{mf_p if g_mf else '回退默认'}）")
print(f"参数: 速度 {SPEED} m/s | 力矩 {EFFORT} N | 轮询超时 {args.timeout}s")

# ── SDK 在 parse 后才碰（--help 不初始化任何硬件）──
from galbot_sdk.g1 import GalbotRobot, G1JointGroup, ControlStatus  # noqa: E402

SIDES = {"left": ["left_gripper"], "right": ["right_gripper"],
         "both": ["left_gripper", "right_gripper"]}[args.side]

robot = GalbotRobot()
robot.init()
time.sleep(2)

for side in SIDES:
    grp = getattr(G1JointGroup, side)
    print(f"\n== {side} ==", flush=True)
    st = robot.get_gripper_state(grp)
    if st is None:
        print("  ❌ get_gripper_state 返回 None（查 RT/急停/使能），跳过", flush=True)
        continue
    print(f"  当前: width={st.width:.4f} m（{pct_of(st.width):.1f}%）"
          f" velocity={st.velocity:.4f} effort={st.effort:.2f}", flush=True)

    if PCT is None:
        continue                                 # 只读

    w_tgt = args.width if args.width is not None else WMIN + PCT / 100.0 * SPAN
    status = robot.set_gripper_command(grp, w_tgt, SPEED, EFFORT, False)
    if status != ControlStatus.SUCCESS:
        print(f"  ❌ set_gripper_command({w_tgt:.4f}) → {status}", flush=True)
        continue
    print(f"  → 目标 {w_tgt:.4f} m（{pct_of(w_tgt):.1f}%）已下发，轮询到位...", flush=True)

    # 判稳：宽度动过（活性确认）且 is_moving 清零；8s 纹丝不动=已夹住或反馈冻结
    t0 = time.time()
    moved = False
    last_w = st.width
    while time.time() - t0 < args.timeout:
        time.sleep(0.1)
        s = robot.get_gripper_state(grp)
        if s is None:
            continue
        if abs(s.width - last_w) > 5e-4:         # 0.1s 内变 0.5mm → 在动
            moved = True
        last_w = s.width
        if moved and not s.is_moving:
            break
        if not moved and time.time() - t0 > 8.0:
            print("  ⚠ 8s 反馈纹丝不动——已夹住（力矩兜底）或反馈冻结；目视确认", flush=True)
            break
    else:
        print(f"  ⚠ {args.timeout}s 未判稳，按当前位收尾", flush=True)

    time.sleep(0.5)                              # 机械稳态
    ws = []
    for _ in range(5):
        s = robot.get_gripper_state(grp)
        if s is not None:
            ws.append(s.width)
        time.sleep(0.2)
    if ws:
        w_fin = statistics.median(ws)
        print(f"  终态: width={w_fin:.4f} m（{pct_of(w_fin):.1f}%）"
              f" 目标 {w_tgt:.4f} m（{pct_of(w_tgt):.1f}%）"
              f" 偏差 {(w_fin - w_tgt) * 1000:+.1f} mm", flush=True)
        if abs(w_fin - w_tgt) > 5e-3 and PCT < 5:
            print("  ⓘ 终态比目标宽 → 大概率被物品挡住（力矩限位夹持中），正常", flush=True)
    else:
        print("  ❌ 终态采样全 None（查 RT/急停）", flush=True)

robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
sys.stdout.flush()
os._exit(0)                                      # SDK 残留线程不退，强杀
