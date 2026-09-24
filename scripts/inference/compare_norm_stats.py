#!/usr/bin/env python
"""三方 norm stats 对比：数据集 meta/stats.json vs checkpoint normalizer safetensors。

用法:
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/compare_norm_stats.py \
      [--dataset ~/holy/datasets/pick_place_balence] \
      [--ckpt ~/holy/models/pi05_g1_pretrain/pi05_g1_040000/pretrained_model] \
      [--npz ~/holy/datasets/replay_input.npz]

判读：state q01/q99 出现 ←不相交 = checkpoint 训练数据与该数据集不是同一代
（该维位姿整个在盒子外，模型必然当 OOD 处理、输出无条件先验）。
"""
import argparse
import json
import pathlib
import struct

import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", default="~/holy/datasets/pick_place_balence")
ap.add_argument("--ckpt",
                default="~/holy/models/pi05_g1_pretrain/pi05_g1_040000/pretrained_model")
ap.add_argument("--npz", default="~/holy/datasets/replay_input.npz")
args = ap.parse_args()
ds = pathlib.Path(args.dataset).expanduser()
ckpt = pathlib.Path(args.ckpt).expanduser()


def load_safetensors(p: pathlib.Path) -> dict:
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        off = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            npdt = np.float32 if v["dtype"] == "F32" else np.float64
            fh.seek(off + v["data_offsets"][0])
            out[k] = np.frombuffer(
                fh.read(int(np.prod(v["shape"])) * np.dtype(npdt).itemsize),
                npdt).copy()
        return out


dsl = json.loads((ds / "meta" / "stats.json").read_text())
dsq = np.asarray(dsl["observation.state"]["q01"], np.float32)
dsQ = np.asarray(dsl["observation.state"]["q99"], np.float32)

pre = load_safetensors(ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors")
pq, pQ = pre["observation.state.q01"], pre["observation.state.q99"]

z = np.load(pathlib.Path(args.npz).expanduser(), allow_pickle=False)
snames = json.loads(str(z["state_names_json"]))

print("=== state q01/q99：checkpoint normalizer vs 数据集自身 ===")
bad = 0
for i, nm in enumerate(snames):
    hit = pq[i] > dsQ[i] + 1e-4 or pQ[i] < dsq[i] - 1e-4
    bad += hit
    print(f"{i:2d} {nm:20s} ckpt[{pq[i]:7.3f},{pQ[i]:7.3f}] "
          f"ds[{dsq[i]:7.3f},{dsQ[i]:7.3f}]" + (" ←不相交" if hit else ""))
print(f"\n不相交 {bad}/{len(snames)} 维 → " +
      ("checkpoint 训练数据 ≠ 该数据集（或统计按子集算的）" if bad else "统计同源，正常"))

daq = np.asarray(dsl["action"]["q01"], np.float32)
daQ = np.asarray(dsl["action"]["q99"], np.float32)
print(f"action q01 最大差 {float(np.abs(pre['action.q01']-daq).max()):.4f} | "
      f"q99 最大差 {float(np.abs(pre['action.q99']-daQ).max()):.4f}"
      "（≈0 = 动作统计同源；大 = 训练动作分布不同）")
