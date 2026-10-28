#!/usr/bin/env python
"""lerobot 数据集（S1 遥测）→ 回放台 traj .npz 转换器。

输入：lerobot v3.0 数据集（如 ~/datasets/clean_ready_224_1028，galbot_one_s1，
action 16 维右臂在前：right_arm×7+right_gripper+left_arm×7+left_gripper，
爪=0–100 百分比，fps=30）。列映射按 info.json 的 names **按名对号**（不依赖
列序）；episode 切分按 data parquet 的 episode_index 列（本数据集 file↔集
1:1，但转换器不依赖该巧合）。运行环境=lerobot_cpu（pyarrow 所在）：
  ~/miniforge3/envs/lerobot_cpu/bin/python lerobot_to_traj.py \
      --dataset ~/datasets/clean_ready_224_1028 --episodes 0,1,2
输出：traj npz（replay_stream_bench.py --data 直接吃）：
  chunks [C,L,14] 臂绝对目标 rad + joint_names 14 + grip [C,L,2] 宽度 m
  （gripper_pct_to_width_m 同执行器发令路径换算）+ episode_ids [C] + hz。
每集按 --chunk-rows（默认 50，=执行器块长）切伪块，尾块 hold-last 补齐。

⚠ 本数据集 meta/stats.json 的臂值范围（±0.5）与 data 实测（±2.0 rad）矛盾
（全库 28 file 量纲一致，stats 疑清洗前旧口径）——训练 norm 判读时勿引这份。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "s1" / "inference"))
from policies.base import GRIP_WIDTH_MAX_M, GRIP_WIDTH_MIN_M  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", required=True, help="lerobot 数据集根目录")
ap.add_argument("--episodes", default="0",
                help="集号，如 '0,1,2' 或 '0-8'（默认第 0 集；'all'=全部）")
ap.add_argument("--chunk-rows", type=int, default=50,
                help="伪块行数（默认 50=执行器块长口径）")
ap.add_argument("--out", default=None,
                help="输出 npz（默认 <数据集同级>/<名>_traj_ep<X>.npz）")
args = ap.parse_args()


def parse_episodes(spec: str, total: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(total))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [e for e in out if not 0 <= e < total]
    if bad:
        raise SystemExit(f"集号越界 {bad}（共 {total} 集）")
    return sorted(set(out))


root = pathlib.Path(args.dataset).expanduser()
info = json.loads((root / "meta" / "info.json").read_text())
fps = int(info["fps"])
act_names = list(info["features"]["action"]["names"])       # 16 维按名对号
files = sorted((root / "data").glob("chunk-*/*.parquet"))
print(f"数据集 {root.name}：fps={fps}，action {len(act_names)} 维，"
      f"{len(files)} 个 data file")

# episode → data file（读 episodes meta parquet 拿 from/to + file_index；
# 本数据集 file↔集 1:1，但按 from/to 切是通用口径）
ep_meta = {}
for mf in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
    import pyarrow.parquet as pq
    d = pq.ParquetFile(mf).read(columns=[
        "episode_index", "length", "data/chunk_index",
        "data/file_index", "dataset_from_index", "dataset_to_index"]).to_pydict()
    for i, ep in enumerate(d["episode_index"]):
        ep_meta[ep] = {"chunk": d["data/chunk_index"][i],
                       "file": d["data/file_index"][i],
                       "frm": d["dataset_from_index"][i],
                       "to": d["dataset_to_index"][i],
                       "len": d["length"][i]}
eps = parse_episodes(args.episodes, max(ep_meta) + 1 if ep_meta else 0)
print(f"回放集：{eps}")

import pyarrow.parquet as pq  # noqa: E402

chunks, grip_all, ep_ids = [], [], []
for ep in eps:
    m = ep_meta[ep]
    f = root / "data" / f"chunk-{m['chunk']:03d}" / f"file-{m['file']:03d}.parquet"
    # dataset_from/to_index 是全库全局索引（非文件内偏移）→ 直接按 episode_index 列过滤
    t = pq.ParquetFile(f).read(columns=["action", "episode_index"])
    mask = np.array(t.column("episode_index").to_pylist()) == ep
    A = np.array(t.column("action").to_numpy().tolist())[mask]
    assert len(A) == m["len"], (ep, len(A), m["len"])
    idx = {nm: i for i, nm in enumerate(act_names)}
    # 臂：按名取 7×2；爪：百分比 → 宽度 m（执行器发令路径同款换算）
    la = np.stack([A[:, idx[n]] for n in (f"left_arm_joint{i}" for i in range(1, 8))], 1)
    ra = np.stack([A[:, idx[n]] for n in (f"right_arm_joint{i}" for i in range(1, 8))], 1)
    rows = np.concatenate([ra, la], 1)                    # 关节名顺序=RIGHT+LEFT（执行器口径）
    grip_pct = np.stack([A[:, idx["right_arm_gripper"]],
                         A[:, idx["left_arm_gripper"]]], 1)
    # 向量化 = gripper_pct_to_width_m 同公式（常量自 base 导入，不另起口径）
    grip_m = GRIP_WIDTH_MIN_M + np.clip(grip_pct, 0.0, 100.0) / 100.0 * (
        GRIP_WIDTH_MAX_M - GRIP_WIDTH_MIN_M)
    # 切伪块，尾块 hold-last 补齐
    L = args.chunk_rows
    n0 = len(chunks)
    for s in range(0, len(rows), L):
        blk = rows[s:s + L]
        g = grip_m[s:s + L]
        pad = L - len(blk)
        if pad > 0:
            blk = np.concatenate([blk, np.repeat(blk[-1:], pad, 0)], 0)
            g = np.concatenate([g, np.repeat(g[-1:], pad, 0)], 0)
        chunks.append(blk)
        grip_all.append(g)
        ep_ids.append(ep)
    print(f"  ep{ep}: {len(rows)} 帧 → {len(chunks) - n0} 伪块，"
          f"爪宽 m [{grip_m.min():.3f},{grip_m.max():.3f}]")

names = ([f"right_arm_joint{i}" for i in range(1, 8)]
         + [f"left_arm_joint{i}" for i in range(1, 8)])
C = np.array(chunks)
out = pathlib.Path(args.out) if args.out else root.parent / f"{root.name}_traj_ep{args.episodes.replace(',', '_').replace('-', 'to')}.npz"
np.savez(out, chunks=C, joint_names=np.array(names),
         grip=np.array(grip_all), episode_ids=np.array(ep_ids), hz=fps)
print(f"落盘 {out}：chunks {C.shape}（ep_ids {sorted(set(ep_ids))}，"
      f"hz={fps}）→ replay --data {out}")
