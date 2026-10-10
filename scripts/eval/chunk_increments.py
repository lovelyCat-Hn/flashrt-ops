#!/usr/bin/env python
"""chunk 增量谱分析——模型输出的 50 步增量范围定性（run_g1_loop_traj --dump-chunks 的离线伴生工具）

回答的问题：模型输出的逐行增量 |Δ| 分布到底多大 → 校准 --delta-max 取值
（限幅阈值应落在分布的哪里：切尾部 vs 切主体）；同时给出 30Hz 行增量
与 240Hz 帧增量（行增量/frames_per_row，样条近似均匀再分）两套口径，
以及方向翻转率（模型抖动签名，与真机遥测 flips 对照）。

输入：logs/chunks_traj_<ts>/chunk_*.npz（--dump-chunks 落盘，含 chunk(50,16)
+base_arm(14)+infer_ms+fps）。纯离线只读，不碰 SDK、不下发。

维度约定（16 = R 臂0-6 | R 爪7 | L 臂8-14 | L 爪15）；臂维单位 rad，
行增量 ×fps = 隐含速度 rad/s。

用法：
  python3 scripts/eval/chunk_increments.py --chunks-dir logs/chunks_traj_XXXX
  python3 scripts/eval/chunk_increments.py --chunks-dir ... --frames-per-row 8
"""
import argparse
import glob
import os
import re

import numpy as np

DIM_LABELS = ([f"R{j+1}" for j in range(7)] + ["R爪"]
              + [f"L{j+1}" for j in range(7)] + ["L爪"])
ARM_DIMS = list(range(0, 7)) + list(range(8, 15))   # 14 臂维
GRIP_DIMS = [7, 15]

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--chunks-dir", required=True, help="--dump-chunks 落盘目录")
ap.add_argument("--frames-per-row", type=int, default=8,
                help="帧倍频（帧增量=行增量/该值，默认 8）")
ap.add_argument("--thresholds", type=float, nargs="+",
                default=[0.02, 0.03, 0.04, 0.05],
                help="限幅候选阈值（rad/行），报告每个阈值的截断占比")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.npz")))
if not files:
    raise SystemExit(f"目录无 chunk_*.npz：{args.chunks_dir}")

chunks, bases, infers = [], [], []
for f in files:
    d = np.load(f)
    chunks.append(d["chunk"])
    bases.append(d["base_arm"])
    infers.append(float(d["infer_ms"]))
C = np.stack(chunks)                     # (B,50,16)
B, T, D = C.shape
fps = float(np.median([float(np.asarray(np.load(f)["fps"])) for f in files])) or 30.0

# ── 行增量谱（模型绝对目标的逐行差）──
dC = np.diff(C, axis=1)                  # (B,49,16)
arm = np.abs(dC[:, :, ARM_DIMS]).ravel()  # 全部臂维行增量
arm_signed = dC[:, :, ARM_DIMS].reshape(-1, len(ARM_DIMS))

print(f"== chunk 增量谱 == 输入 {args.chunks_dir}")
print(f"块数 {B} | 每块 {T} 行 | 臂维 14 | 样本 {arm.size} 行增量")
print(f"推理时延: p50 {np.median(infers):.0f} | max {max(infers):.0f} ms")
print()
print("[臂维行增量 |Δ| 全体分布]（rad/行；×30=rad/s）")
for q in (50, 75, 90, 95, 99, 100):
    v = np.percentile(arm, q)
    print(f"  p{q:<3} {v*1000:7.1f} mrad  ({v*fps:5.2f} rad/s, "
          f"帧增量 {v/args.frames_per_row*1000:5.2f} mrad@240Hz)")
print()
print("[限幅候选阈值截断占比]（>阈值的行增量比例；臂维全体）")
for th in args.thresholds:
    frac = float(np.mean(arm > th))
    per_block = float(np.mean(np.any(
        np.abs(dC[:, :, ARM_DIMS]) > th, axis=2)))   # 每块至少一行超阈的块比例
    print(f"  delta-max {th:.3f} ({th*fps:.2f} rad/s): 步数占比 {frac*100:5.2f}% | "
          f"涉及块比例 {per_block*100:5.1f}%")
print()
print("[逐维分布]（|Δ| mrad：p50 / p95 / p99 / max）")
hdr = "  维    p50    p95    p99     max   翻转率"
print(hdr)
for dd in range(D):
    x = np.abs(dC[:, :, dd]).ravel()
    s = dC[:, :, dd].ravel()
    nz = s[s != 0]
    flip = float(np.mean(nz[1:] * nz[:-1] < 0)) if nz.size > 2 else 0.0
    print(f"  {DIM_LABELS[dd]:<4} {np.percentile(x,50)*1000:6.1f} "
          f"{np.percentile(x,95)*1000:6.1f} {np.percentile(x,99)*1000:6.1f} "
          f"{x.max()*1000:6.1f}   {flip*100:5.1f}%")
print()
print("[帧增量口径]（240Hz = 行增量/8 均分，样条近似）")
print(f"  臂维帧增量 p50 {np.percentile(arm,50)/args.frames_per_row*1000:.2f} | "
      f"p99 {np.percentile(arm,99)/args.frames_per_row*1000:.2f} | "
      f"max {arm.max()/args.frames_per_row*1000:.2f} mrad/帧")
print(f"  每帧隐含速度 = 帧增量×240Hz：p95 {np.percentile(arm,95)*fps:.2f} rad/s（与行口径同速）")
print()
print("[换块失配]（下一块行0目标 vs 上一块水位行的模型目标差，臂维 max；"
      "模型自身重规划跳变，未含任何执行器重锚）")
j0 = np.abs(C[:, 1, :] - C[:, 0, :])[:, ARM_DIMS]     # 块内行0→1 作基线参照
print(f"  基线·块内行0→1 |Δ| max 逐块: " +
      " ".join(f"{v*1000:.0f}" for v in j0.max(axis=1)))
if B >= 2:
    mism = np.abs(C[1:, 0, ARM_DIMS] - C[:-1, 25, ARM_DIMS]).max(axis=1)
    print(f"  C[b,0]−C[b−1,25] 模型重规划失配逐块: " +
          " ".join(f"{v*1000:.0f}" for v in mism) +
          f"  (p50 {np.median(mism)*1000:.0f} / max {mism.max()*1000:.0f} mrad)")
