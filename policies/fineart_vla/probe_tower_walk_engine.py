#!/usr/bin/env python
"""SigLIP 塔逐层走查·引擎侧：embeddings 输出 → 每层后 → post_layernorm。

落盘 <dir>/tower_walk_engine.npz：emb0, l00,l01,l02,l04,l08,l16,l26, post_ln
（形状 (3,256,1152)）。
"""
from __future__ import annotations

import argparse
import os

import numpy as np

DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")
CHECKS = (0, 1, 2, 4, 8, 16, 26)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/holy/models/fineart_vla_midtrained"))
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    import torch
    from flash_rt.frontends.torch.fineart_vla_rtx import FineARTVLATorchFrontendRtx

    dec = FineARTVLATorchFrontendRtx(args.ckpt, num_views=3, action_dim=14,
                                     num_steps=10)._ensure_subtask_decoder()
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

    np.savez(args.dir + "/tower_walk_engine.npz", **out)
    print("saved:", {k: v.shape for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
