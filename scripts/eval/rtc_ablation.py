#!/usr/bin/env python
"""RTC 引导（guided RTC）质量+延迟消融：off vs guided(h=10)，同噪声教师强制。

背景（2026-10-09，计划 stateful-whistling-zebra Part C）：
  FlashRT 引擎新增推理侧前缀引导（lerobot policies/rtc guided 模式移植），
  严格 opt-in（rtc=False 默认）。本探针量化两件事：
  1. 质量：同噪声下 off 与 guided 的输出差异——
     - 引导区移动：行 < horizon 被拉动（无移动=引导未生效，硬门槛）
     - 尾段传播：行 ≥ horizon 无直接修正（内核级保证），但 decoder
       self-attention 会把头段改动传播到尾段——这是 RTC 的正常语义
       （lerobot 同款），指标化报告传播幅度，不作位等断言
     - cos（臂维）：行 < horizon 被引导拉动的幅度分布
     - 接缝跳变（RTC 存在的唯一理由，主指标）：模拟换块——旧块 A 消费
       25 步后换新块 B，接缝=|B[0] − A[24]|（臂维，数据空间）。guided 的 B
       以 A 未消费尾段（重锚后）为前缀，接缝应显著小于 off
     - 夹爪哨兵：guided vs off 夹爪维(7/15)最大差（0.8 线惯例，超=异常）
  2. 延迟：同模型同噪声 infer 耗时 off vs guided（引导在图内 elementwise，
     预期 ≈相等）

方法（对齐 ab_compare_pi05.py 惯例）：
  - 同一 rtc=True 模型实例内两态对照（off=rtc_prefix=None；rtc=None 前缀走
    enable=0 短路，与 rtc=False 实例逐位等——tests/test_pi05_rtc_guided.py
    已证，此处不再重复加载第二实例）
  - 每样本：obs_A（随机图×3 + state）→ A 块；obs_B = 图 1px 平移 + state
    小扰动（模拟 0.33s 后场景，A/B 相关）；BASE_B = BASE_A + ε（臂 state
    同步移 ε，前缀按脚本侧同款公式重锚）
  - 噪声对齐：torch.manual_seed 后 predict（内部默认生成器），A/B 各自独立
    种子；off 与 guided 跑同一 B 种子 → 差异纯来自引导
  - cache_frames=1 全量帧，无图模式（探针惯例，与 09-21/10-06 批口径同）

用法:
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/eval/rtc_ablation.py [N] [--out 证据目录]
输出:
  <out>/README.md（条件+文件清单）、summary.txt（stdout 全文）、
  sample_i_{A,B_off,B_on}.npy（未归一化动作块）
"""
import argparse
import os
import pathlib
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=12, help="样本数（默认 12）")
ap.add_argument("--out", default=None, help="证据目录（默认 evidence/20261009_rtc_guided_ablation）")
ap.add_argument("--horizon", type=int, default=10, help="RTC 引导窗口（默认 10=lerobot 默认）")
ap.add_argument("--leftover-from", type=int, default=25, help="A 块消费步数=换块位置（默认 25=水位）")
args_cli = ap.parse_args()

N = args_cli.n
SEED_BASE = 20261009
HORIZON = args_cli.horizon
K_SWAP = args_cli.leftover_from          # A 消费到第 K 步换块（水位语义）
EPS = 0.01                               # 合成 BASE 漂移 rad（重锚公式用）
SENTINEL_GRIP = 0.8                      # 夹爪维异常线（惯例）

OUT = pathlib.Path(args_cli.out) if args_cli.out else (
    pathlib.Path(__file__).resolve().parents[2] / "evidence"
    / "20261009_rtc_guided_ablation")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 前端模块导入时读此 env（v3 同款坑）：必须在 import flash_rt 前定死 chunk
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
os.environ["FLASH_RT_PI05_ACTION_CHUNK_SIZE"] = "50"

import flash_rt  # noqa: E402

CKPT = os.path.expanduser("~/holy/models/pi05_g1_onlypick_deploy")
VIEWS, ACTION_DIM, CHUNK = 3, 16, 50
PROMPT = "Left arm pick up A. Right arm pick up A."   # 部署原句（config [run].prompt）


def _make_obs_A(rng):
    return {
        "image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "wrist_image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "wrist_image_right": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "state": rng.uniform(-1.5, 1.5, (23,)).astype(np.float32),
    }


def _make_obs_B(obs_A, rng):
    """0.33s 后的观测近似：图 1px 平移 + state 小扰动（含臂维 BASE 漂移 ε）。"""
    b = {}
    for k in ("image", "wrist_image", "wrist_image_right"):
        b[k] = np.roll(obs_A[k], 1, axis=1).copy()
    st = obs_A["state"] + rng.normal(0, 0.02, obs_A["state"].shape).astype(np.float32)
    st[:14] += EPS               # 臂维（0-13=右臂7+左臂7）合成 BASE 漂移
    b["state"] = st.astype(np.float32)
    return b


def _reanchor(leftover, eps):
    """脚本侧 _rtc_prefix 同款：臂维 (OLD_BASE+delta)−NEW_BASE = delta−eps；
    夹爪维（7/15）绝对原值携带。"""
    pf = leftover.copy()
    pf[:, :7] -= eps
    pf[:, 8:15] -= eps
    return pf


def _cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


ARM = list(range(0, 7)) + list(range(8, 15))   # 臂维（右臂在前）
GRIP = [7, 15]

print(f"ckpt={CKPT} N={N} horizon={HORIZON} K_swap={K_SWAP} eps={EPS}")
print(f"mode=bf16 cache_frames=1 no-graph（探针口径）prompt={PROMPT!r}")

t0 = time.time()
model = flash_rt.load_model(CKPT, config="pi05", num_views=VIEWS,
                            cache_frames=1, action_dim=ACTION_DIM,
                            rtc=True, rtc_horizon=HORIZON, rtc_max_weight=10.0)
print(f"load {time.time() - t0:.1f}s")

rows = []
seam_off_all, seam_on_all, cos_head_all = [], [], []
moved_ok, grip_sent = True, 0.0
tail_prop_all = []
lat_off, lat_on = [], []

for i in range(N):
    rng = np.random.RandomState(SEED_BASE + i)
    obs_A = _make_obs_A(rng)
    obs_B = _make_obs_B(obs_A, rng)

    torch.manual_seed(SEED_BASE + 1000 + i)      # A 块独立噪声
    t = time.perf_counter()
    A = np.asarray(model.predict(obs_A, prompt=PROMPT, state=obs_A["state"]))
    lat_off.append((time.perf_counter() - t) * 1000)

    torch.manual_seed(SEED_BASE + 2000 + i)      # B 块噪声（off/guided 共用）
    t = time.perf_counter()
    B_off = np.asarray(model.predict(obs_B, prompt=PROMPT, state=obs_B["state"]))
    lat_off.append((time.perf_counter() - t) * 1000)

    prefix = _reanchor(A[K_SWAP:], EPS)          # 旧块未消费尾段，重锚进 B 坐标系
    torch.manual_seed(SEED_BASE + 2000 + i)      # 同噪声
    t = time.perf_counter()
    B_on = np.asarray(model.predict(obs_B, prompt=PROMPT, state=obs_B["state"],
                                    rtc_prefix=prefix))
    lat_on.append((time.perf_counter() - t) * 1000)

    np.save(OUT / f"sample_{i}_A.npy", A)
    np.save(OUT / f"sample_{i}_B_off.npy", B_off)
    np.save(OUT / f"sample_{i}_B_on.npy", B_on)

    # ① 引导区移动门槛：行 < horizon 必须真被拉动（逐位相同=引导未生效）
    head_bits_same = np.array_equal(
        B_on[:HORIZON].view(np.uint16), B_off[:HORIZON].view(np.uint16))
    moved_ok &= not head_bits_same

    # ② cos（臂维）：行 < horizon 引导拉动幅度
    cos_h = _cos(B_on[:HORIZON, ARM], B_off[:HORIZON, ARM])
    cos_head_all.append(cos_h)

    # ②' 尾段传播幅度（行 ≥ horizon，预期 ≪ 头段移动量，报告不作门槛）
    tail_prop_all.append(float(np.max(np.abs(B_on[HORIZON:] - B_off[HORIZON:]))))

    # ③ 接缝跳变（主指标）：|B[0] − A[K-1]|，臂维数据空间
    seam_off = float(np.max(np.abs(B_off[0, ARM] - A[K_SWAP - 1, ARM])))
    seam_on = float(np.max(np.abs(B_on[0, ARM] - A[K_SWAP - 1, ARM])))
    seam_off_all.append(seam_off)
    seam_on_all.append(seam_on)

    # ④ 夹爪哨兵
    g = float(np.max(np.abs(B_on[:, GRIP] - B_off[:, GRIP])))
    grip_sent = max(grip_sent, g)

    rows.append((i, cos_h, seam_off, seam_on, head_bits_same, g))
    print(f"[{i:>2}] cos_head {cos_h:.4f} | seam off {seam_off*1000:7.1f} → "
          f"on {seam_on*1000:7.1f} mrad | 头段移动 {not head_bits_same} | "
          f"尾段传播 {tail_prop_all[-1]*1000:.1f} mrad | grip Δmax {g:.3f}")

n_on_win = sum(1 for r in rows if r[3] < r[2])
print("\n===== 汇总（同噪声教师强制，臂维数据空间） =====")
print(f"引导区移动门槛（行<{HORIZON} 逐位有变化）: {'PASS' if moved_ok else 'FAIL'}"
      f"（{N}/{N} 样本）")
print(f"尾段传播（行≥{HORIZON}，attention 语义）: max "
      f"{np.max(tail_prop_all)*1000:.1f} | mean {np.mean(tail_prop_all)*1000:.1f} mrad"
      f"（无直接修正；头段移动量见逐行 cos/传播列）")
print(f"cos 行<{HORIZON}（臂维）: mean {np.mean(cos_head_all):.4f} | "
      f"min {np.min(cos_head_all):.4f} | max {np.max(cos_head_all):.4f}")
print(f"接缝跳变 |B[0]−A[{K_SWAP-1}]|: off p50 {np.percentile(seam_off_all, 50)*1000:.1f} / "
      f"mean {np.mean(seam_off_all)*1000:.1f} mrad → guided p50 "
      f"{np.percentile(seam_on_all, 50)*1000:.1f} / mean {np.mean(seam_on_all)*1000:.1f} mrad"
      f"（guided 更小 {n_on_win}/{N} 样本）")
print(f"夹爪哨兵: max |Δ| {grip_sent:.3f}（{SENTINEL_GRIP} 线 "
      f"{'PASS' if grip_sent < SENTINEL_GRIP else 'FAIL'}）")
print(f"延迟（predict 含建 pipeline 开销口径）: off p50 {np.percentile(lat_off, 50):.1f} / "
      f"p95 {np.percentile(lat_off, 95):.1f} ms → guided p50 "
      f"{np.percentile(lat_on, 50):.1f} / p95 {np.percentile(lat_on, 95):.1f} ms"
      f"（图内 elementwise，预期 ≈相等）")
print(f"\n证据目录: {OUT}")
