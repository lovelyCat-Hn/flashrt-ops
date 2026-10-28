#!/usr/bin/env python
"""SigLIP 塔逐层走查·引擎克隆在 lerobot_cpu CPU fp32 下重跑（同 torch 构建）。

消除 GPU/SDPA/backend 变量：若仍 l00 崩 → 权重加载或克隆代码 bug；
若恢复 → 引擎 GPU 路径问题。落盘 <dir>/tower_walk_engine_cpu.npz。
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

CHECKS = (0, 1, 2, 4, 8, 16, 26)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/holy/models/fineart_vla_midtrained"))
    ap.add_argument("--dir", default=os.path.expanduser("~/holy/evidence/fineart_subtask_parity"))
    args = ap.parse_args()

    sys.path.insert(0, "/home/galbot/holy/FlashRT")
    import torch
    from flash_rt.models.fineart_vla.text_decode_rtx import SubtaskDecoder

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    dec = SubtaskDecoder(args.ckpt, device="cpu", lm_head=None)
    with torch.no_grad():
        pixel, _ = dec._preprocess_images(imgs)
        v = dec.vision
        x = v.patch_embedding(pixel).flatten(2).transpose(1, 2)
        x = x + v.position_embedding(v.position_ids)
        out = {"emb0": x.float().cpu().numpy()}
        for i, layer in enumerate(v.layers):
            x = layer(x)
            if i in CHECKS:
                out[f"l{i:02d}"] = x.float().cpu().numpy()
        out["post_ln"] = v.post_layernorm(x).float().cpu().numpy()

    np.savez(args.dir + "/tower_walk_engine_cpu.npz", **out)
    print("saved:", {k: v.shape for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
