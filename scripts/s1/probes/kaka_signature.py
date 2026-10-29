#!/usr/bin/env python
"""kaka 电流签名分析器——kaka_*.npz → 分窗口电流/速度/误差特征表。

判读（电流=力矩代理，静态基线≈各关节重力保持电流）：
  A 齿隙扫过   ：kaka 窗电流快速过零/大幅 |Δcur| 尖峰（反向瞬间）
  B 加速度饱和 ：kaka 窗电流顶棚平台（|cur| 持续高位）+ acc p95≈limit
  C 速度饱和急停：|vel| p95→limit_vel（3 rad/s）后电流骤降
  D hunting     ：静止段电流固定频率振荡（FFT 峰）
  E stick-slip  ：低速段速度微脉冲
用法： python kaka_signature.py <kaka_*.npz> [--mark-win 1.5]
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--mark-win", type=float, default=1.5,
                help="标记窗口半径 s（kaka 窗=标记±此值）")
A = ap.parse_args()
z = np.load(A.npz, allow_pickle=False)
names = [str(s) for s in z["names"]]
t, pos, vel, acc, cur = z["t"], z["pos"], z["vel"], z["acc"], z["cur"]
tick, ticks_t, ticks_tgt, marks = z["tick"], z["ticks_t"], z["ticks_tgt"], z["marks"]
cfg = json.loads(str(z["config"]))
print(f"配置: ctrl={cfg['ctrl_hz']} R={cfg.get('rows_per_cmd')} fr={cfg['frames_per_row']} "
      f"r/c={cfg['execute_actions_per_chunk']} | 样本 {len(t)} | 标记 {len(marks)}")
dur = t[-1] - t[0]
tgt_at = ticks_tgt[tick]                      # 每样本所属 tick 的末行目标
# ⚠ get_joint_states.position 与指令系符号相反（10-29 三跑实锤：同列相关
# −0.97~−0.98、负镜像 +0.98、每关节偏移 c 跨跑稳定≈常量，臂关节全体）。
# err 必须先镜像归位再取偏移：err = |(−pos−c)−tgt|，c=逐关节 med(−pos−tgt)。
# 直接 |tgt−pos| 会抬到行程量级（②安静跑也"3.9 rad"，纯符号伪影）。
neg = -pos
_c = np.median(neg - tgt_at, axis=0)
err = np.abs(neg - _c - tgt_at)               # (N, J) 目标滞后（含 LP 尾）

in_mark = np.zeros(len(t), bool)
for m in marks:
    in_mark |= (t > m - A.mark_win) & (t < m + A.mark_win)

def fliprate(x):                              # 电流符号翻转率 /s（去基线）
    seg = np.diff(x)
    return float(np.sum((seg[:-1] * seg[1:]) < 0)) / max(dur, 1e-9) if len(x) > 2 else 0.0

def spike_rate(x, thr):                       # |Δcur| 尖峰率 /s
    d = np.abs(np.diff(x))
    return float(np.sum(d > thr)) / max(dur, 1e-9)

hdr = f"{'关节':<18} {'窗':>4} {'cur中位':>8} {'cur p95':>8} {'|Δcur|>1A/s':>11} {'|vel|p95':>9} {'acc p95':>8} {'err p95 mrad':>12}"
print(hdr)
for j, nm in enumerate(names):
    for tag, msk in (("响", in_mark), ("静", ~in_mark)):
        if msk.sum() < 10:
            continue
        c, v, a, e = cur[msk, j], vel[msk, j], acc[msk, j], err[msk, j]
        print(f"{nm:<18} {tag:>4} {np.median(c):8.2f} {np.percentile(c,95):8.2f} "
              f"{spike_rate(c, 1.0):11.1f} {np.percentile(np.abs(v),95):9.3f} "
              f"{np.percentile(np.abs(a),95):8.2f} {np.percentile(e,95)*1000:12.1f}")

# hunting 检查：全程每关节电流去基线后的主频（最高峰）
print("\n电流主频（去中位基线，FFT 峰；显著=峰高>8×谱底中位，>5Hz 才报）:")
dt = float(np.median(np.diff(t)))
for j, nm in enumerate(names):
    x = cur[:, j] - np.median(cur[:, j])
    if len(x) < 64:
        continue
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), dt)
    k = int(np.argmax(sp[1:]) + 1)
    med = float(np.median(sp[10:])) + 1e-12   # 谱底中位（跳过近直流）；无门限则噪声关节全是假峰
    if fr[k] > 5 and sp[k] > 8 * med:
        print(f"  {nm}: {fr[k]:.1f} Hz（显著 {sp[k]/med:.0f}×）")
