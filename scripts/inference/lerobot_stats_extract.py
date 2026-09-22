#!/usr/bin/env python
"""lerobot 微调输出 → openpi 形状 norm_stats.json（喂 g1_ckpt_prep --stats）。

微调输出（lerobot train 产物）没有裸 stats json，归一化统计序列化在
processor safetensors 里：
  policy_preprocessor_step_*_normalizer_processor.safetensors
  policy_postprocessor_step_*_unnormalizer_processor.safetensors
两份应同源一致（本脚本强制校验）。统计语义看训练 config 的
normalization_mapping：pi0.5 G1 微调实测 QUANTILES（对应 --mode q01_q99）。

用法:
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/lerobot_stats_extract.py \
      --ckpt <pretrained_model 目录> [--out <json 路径>]
输出 json 形状（prep 的 openpi schema）:
  {"actions": {mean,std,q01,q99: [...]}, "state": {...}}
"""
import argparse
import json
import pathlib

import numpy as np
from safetensors.torch import load_file
import torch

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", required=True,
                help="lerobot pretrained_model 目录（含 policy_*processor*.safetensors）")
ap.add_argument("--out", help="输出 json（默认 <ckpt 上级>/norm_stats_openpi.json）")
args = ap.parse_args()

d = pathlib.Path(args.ckpt)
norm = sorted(d.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
unnorm = sorted(d.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
if not norm or not unnorm:
    raise SystemExit(f"processor safetensors 未找到: {[str(p) for p in norm + unnorm]}")

sd_n = load_file(str(norm[-1]))
sd_u = load_file(str(unnorm[-1]))

# 消费的两块统计必须前后逐位一致（同源）；image 等条目是 IDENTITY 的琐碎
# 统计，前后 processor 本就可能不同，不校验
for feat in ("action", "observation.state"):
    for suf in ("mean", "std", "q01", "q99", "min", "max", "q50", "q10", "q90"):
        k = f"{feat}.{suf}"
        if k in sd_n and k in sd_u and not torch.equal(sd_n[k], sd_u[k]):
            raise SystemExit(f"前后 processor 统计不一致: {k}")

BLOCKS = {"actions": "action", "state": "observation.state"}
FIELDS = ("mean", "std", "q01", "q99")
out = {}
for blk, feat in BLOCKS.items():
    have = {f: sd_n.get(f"{feat}.{f}") for f in FIELDS}
    missing = [f for f, v in have.items() if v is None]
    if missing:
        raise SystemExit(f"{feat} 缺字段 {missing}（processor 里有哪些: "
                         f"{sorted(k for k in sd_n if k.startswith(feat + '.'))}）")
    out[blk] = {f: have[f].float().numpy().tolist() for f in FIELDS}
    print(f"{blk}: {len(out[blk]['q01'])} 维 | q01[0:3]={out[blk]['q01'][:3]} "
          f"q99[-3:]={out[blk]['q99'][-3:]}")

# 夹爪维 sanity：数据集语义 0~100%（第 7/15 维）
for i in (7, 15):
    lo, hi = out["actions"]["q01"][i], out["actions"]["q99"][i]
    print(f"actions.q01/q99 夹爪维[{i}]: {lo:.2f}/{hi:.2f}"
          + ("" if (0 <= lo and hi <= 100.0001) else "  ⚠ 非 0~100 量级，核对语义!"))

dest = pathlib.Path(args.out) if args.out else d.parent / "norm_stats_openpi.json"
dest.write_text(json.dumps(out, indent=1))
print(f"✓ 写 {dest}（喂 g1_ckpt_prep --stats {dest}，--mode 按训练 "
      f"normalization_mapping 定，G1 实测 QUANTILES → q01_q99）")

# 顺带产出 lerobot 数据集 schema 的 meta/stats.json（含 min/max），
# 供 norm_align_check --dataset 直接使用——源机无需拉 857M 数据集
lg = {}
for blk, feat in BLOCKS.items():
    lg[feat] = {}
    for suf in ("mean", "std", "q01", "q99", "min", "max"):
        t = sd_n.get(f"{feat}.{suf}")
        if t is None:
            raise SystemExit(f"{feat}.{suf} 缺失，无法产出 lerobot schema")
        lg[feat][suf] = t.float().numpy().tolist()
meta_dir = d.parent / "meta"
meta_dir.mkdir(exist_ok=True)
(meta_dir / "stats.json").write_text(json.dumps(lg, indent=1))
print(f"✓ 写 {meta_dir / 'stats.json'}（norm_align_check --dataset {d.parent}）")
