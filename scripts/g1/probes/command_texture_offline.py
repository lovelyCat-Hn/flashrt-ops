#!/usr/bin/env python
"""指令阶梯谱（离线，无机器人）：复刻 replay-bench 的确定性指令管线
（数据行 → 链钳 ±delta-max → EMA → 合步采样 → 样条 → 突发下发时刻表），
在**帧粒度**上量指令路径自身的纹理——只回答"抖动轴"（指令间阶梯）；
跟随轴（残差）请看真机日志，两轴独立判读。

用法：
  LD_LIBRARY_PATH=/data/galbot/lib ~/miniforge3/envs/flashrt/bin/python \
    command_texture_offline.py --dataset ~/holy/datasets/pick_place_balence \
    --episode 0 --delta-max 0.14
  --cells "4:1:0.1:linear,4:2:0.1:linear"   # per:merge:alpha:interp，逗号分隔
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "inference"))
from g1_traj_interp import create_interpolator  # noqa: E402

ap = __import__("argparse").ArgumentParser(description=__doc__)
ap.add_argument("--dataset", required=True)
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--delta-max", type=float, default=0.14)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--blocks", type=int, default=3, help="取前 N 个 50 行伪块（=真机 --rounds）")
ap.add_argument("--burst-reserve-ms", type=float, default=6.0)
ap.add_argument("--cells", type=str, default=(
    "4:1:0.1:linear,4:2:0.1:linear,3:2:0.1:linear,6:3:0.1:linear,"
    "6:4:0.3:linear,4:4:0.1:linear,2:4:0.3:linear,4:1:0.1:cubic"))
args = ap.parse_args()

TICK = 1.0 / args.fps
EPS_MRAD = 0.02          # zigzag 计号的最小步长（低于=停滞噪声不计方向）
ACTIVE_TRAVEL_RAD = 0.05  # 全程位移低于此的关节视为惰性，不进纹理统计


def load_blocks(ds_dir, episode):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pfiles = sorted(glob.glob(os.path.join(ds_dir, "data", "chunk-*", "file-*.parquet")))
    for pf in pfiles:
        t = pq.read_table(pf, columns=["action", "episode_index"])
        e = np.asarray(t.column("episode_index").to_pylist())
        idx = np.nonzero(e == episode)[0]
        if len(idx):
            acts = np.asarray(t.take(pa.array(idx)).column("action").to_pylist(),
                              dtype=np.float32)
            break
    else:
        raise SystemExit(f"数据集无 episode {episode}")
    arm = np.concatenate([acts[:, :7], acts[:, 8:15]], axis=1)
    return [arm[i:i + 50].copy() for i in range(0, len(arm), 50)]


def build_rows(cur, tgt_blk, chain_state, alpha, merge):
    """bench build_rows 同构：链钳 → EMA（携带态含块首行）→ 合步 keep-last。"""
    rows = np.empty_like(tgt_blk)
    prev = chain_state.copy() if chain_state is not None else np.asarray(cur, np.float32)
    for kk in range(len(tgt_blk)):
        prev = prev + np.clip(tgt_blk[kk] - prev, -args.delta_max, args.delta_max)
        rows[kk] = prev
    if alpha > 0:
        carry = chain_state is not None
        acc = chain_state.copy() if carry else rows[0]
        for kk in range(0 if carry else 1, len(rows)):
            acc = alpha * rows[kk] + (1.0 - alpha) * acc
            rows[kk] = acc
    if merge > 1:
        idx = list(range(0, len(rows), merge))
        if idx[-1] != len(rows) - 1:
            idx[-1] = len(rows) - 1
        rows = rows[idx]
    return rows, rows[-1].copy()


def dispatch_schedule(n_macros, per, merge):
    """帧下发时刻表（秒）：拍 k 内第 i 帧在 k*TICK + i*fdt，预算内匀速铺满。"""
    times = []
    for _m in range(n_macros):
        groups = np.array_split(np.arange(_m * per, (_m + 1) * per), merge)
        for grp in groups:
            tick_idx = _tick_counter[0]
            if len(grp):    # 空突发不发令，但空拍也占节拍（bench 流式循环同构）
                fdt = (TICK - args.burst_reserve_ms / 1000.0) / len(grp)
                for i, fi in enumerate(grp):
                    times.append((tick_idx * TICK + i * fdt, int(fi)))
            _tick_counter[0] += 1
    return times


_tick_counter = [0]


def texture(per, merge, alpha, interp):
    INTERP = create_interpolator(interp, dim=14, input_hz=args.fps,
                                 output_hz=args.fps * per)
    chain = None
    joints_v, joints_a, joints_zz = [], [], []
    for bi, tgt in enumerate(BLOCKS[:args.blocks]):
        cur = BLOCKS[0][0] if bi == 0 else _last_end  # 块0假想就位；后续块锚=前块末
        rows, chain = build_rows(cur, tgt, chain, alpha, merge)
        _last_end = chain
        frames = INTERP.interpolate(np.vstack([cur[None].astype(np.float32), rows]))
        _tick_counter[0] = 0
        sched = dispatch_schedule(len(rows), per, merge)
        fi = np.asarray([s[1] for s in sched])
        ft = np.asarray([s[0] for s in sched])
        pos = frames[fi]                       # (F,14) 实际下发帧位姿
        dt = np.diff(ft)
        d = np.diff(pos, axis=0)               # (F-1,14) 帧间位移（有符号）
        v = d / dt[:, None]                    # 帧速度 rad/s
        a = np.diff(v, axis=0) / dt[1:, None]  # 帧加速度 rad/s²
        for j in range(14):
            joints_v.append(v[:, j]); joints_a.append(a[:, j])
            dj = d[:, j]
            ok = np.abs(dj[1:]) > EPS_MRAD * 1e-3
            ok &= np.abs(dj[:-1]) > EPS_MRAD * 1e-3
            if ok.sum() >= 10:
                flip = np.sum(np.sign(dj[1:][ok]) != np.sign(dj[:-1][ok]))
                joints_zz.append(flip / ok.sum())
    travel = np.sum(np.abs(np.diff(np.vstack(BLOCKS[:args.blocks]), axis=0)), axis=0)
    active = travel > ACTIVE_TRAVEL_RAD
    med = lambda lst: float(np.median(lst)) if lst else float("nan")
    acc_p = {q: med([np.percentile(np.abs(x), q) * 1e3
                     for j, x in enumerate(joints_a) if active[j % 14]]) for q in (50, 90)}
    zz = med([z for j, z in enumerate(joints_zz) if active[j % 14]])
    return acc_p[50], acc_p[90], zz * 100


BLOCKS = load_blocks(args.dataset, args.episode)
print(f"[指令阶梯谱] {args.dataset} ep{args.episode} 前 {args.blocks} 块 | "
      f"链钳 ±{args.delta_max} | 帧粒纹理（中位 across 活性关节 "
      f"travel>{ACTIVE_TRAVEL_RAD*1e3:.0f}mrad）")
print(f"{'配方 per:merge:α:interp':26s} {'指令加速度p50':>12s} {'p90':>8s} "
      f"{'zigzag%':>8s}   （单位 mrad/s²、方向翻转率）")
for cell in args.cells.split(","):
    per, merge, alpha, interp = cell.split(":")
    a50, a90, zz = texture(int(per), int(merge), float(alpha), interp)
    print(f"  {cell:24s} {a50:12.0f} {a90:8.0f} {zz:8.1f}")
