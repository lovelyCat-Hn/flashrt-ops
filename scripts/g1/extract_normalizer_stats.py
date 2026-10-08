#!/usr/bin/env python
"""lerobot 训练输出 normalizer safetensors → g1_ckpt_prep 可吃的 stats json。

背景：lerobot 微调输出（policy_*_unnormalizer_processor.safetensors）里存的是
数据集统计张量（min/max/mean/std/q01..q99/count，F32），而 g1_ckpt_prep.py
只认 stats json（openpi 形状）。本工具零依赖手解 safetensors（struct 读头 +
按偏移取 F32），把 action / observation.state 两族张量抽成：
    {"action": {...}, "observation.state": {...}}   # lerobot 形状，prep 自动转 openpi

用法（系统 python3 即可，无需 torch）:
  python3 extract_normalizer_stats.py \
      --safetensors <.../policy_postprocessor_step_0_unnormalizer_processor.safetensors> \
      --out <stats.json>
"""
import argparse
import json
import pathlib
import struct

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--safetensors", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--features", default="action,observation.state",
                help="逗号分隔的张量名前缀（默认 action,observation.state）")
args = ap.parse_args()

DTYPES = {"F32": ("<f", 4), "F64": ("<d", 8), "F16": ("<e", 2),
          "I64": ("<q", 8), "I32": ("<i", 4)}

path = pathlib.Path(args.safetensors)
with path.open("rb") as f:
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen))
    blobs = {}
    for name, meta in header.items():
        if name == "__metadata__" or not name.startswith(tuple(args.features.split(","))):
            continue
        fmt, width = DTYPES[meta["dtype"]]
        f.seek(8 + hlen + meta["data_offsets"][0])
        count = 1
        for d in meta["shape"]:
            count *= d
        vals = struct.unpack(f"<{count}{fmt.lstrip('<')}", f.read(count * width))
        # 嵌回原形状
        out, idx = vals, 0
        shape = list(meta["shape"])
        for dim in reversed(shape[1:]):
            out = [out[i:i + dim] for i in range(0, len(out), dim)]
        blobs[name] = out

stats = {}
for feat in args.features.split(","):
    block = {}
    for name, val in blobs.items():
        if name.startswith(feat + "."):
            block[name[len(feat) + 1:]] = val
    stats[feat] = block

out = pathlib.Path(args.out)
out.write_text(json.dumps(stats, indent=1))
print(f"✓ {path.name} → {out}")
for feat, block in stats.items():
    print(f"  {feat}: {sorted(block)}")
    for k in ("q01", "q99"):
        v = block.get(k)
        if v is not None:
            while isinstance(v, list) and isinstance(v[0], list):
                v = v[0]
            print(f"    {k}[0:4] = {[round(x, 6) for x in v[:4]]}")
