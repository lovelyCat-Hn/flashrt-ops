#!/usr/bin/env python
"""单进程合并复走：引擎 SubtaskDecoder 塔 vs HF 塔，同一 pixel 逐层对拍。

同时对照旧落盘 npz（排除跨进程状态污染）。lerobot_cpu 环境。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")


def cos(a, b) -> float:
    a = a.detach().float().cpu().numpy().astype(np.float64).ravel()
    b = b.detach().float().cpu().numpy().astype(np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def main() -> None:
    sys.path.insert(0, "/home/galbot/holy/FlashRT")
    from flash_rt.models.fineart_vla.text_decode_rtx import SubtaskDecoder
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    dec = SubtaskDecoder(CKPT, device="cpu")
    policy = FineARTVLAPolicy.from_pretrained(CKPT)
    policy.eval()
    vm = policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model

    with torch.no_grad():
        # 引擎侧 pixel 作为两侧共同输入
        pixel, _ = dec._preprocess_images(imgs)
        ve = dec.vision
        xe = ve.patch_embedding(pixel).flatten(2).transpose(1, 2)
        xe = xe + ve.position_embedding(ve.position_ids)
        xr = vm.embeddings(pixel.float())
        print(f"emb0 cos={cos(xe, xr):.10f}")

        for i in range(3):
            xe = ve.layers[i](xe)
            o = vm.encoder.layers[i](xr.float(), None)
            xr = o.last_hidden_state if hasattr(o, "last_hidden_state") else o
            print(f"layer{i} cos={cos(xe, xr):.10f}")

        xe = ve.post_layernorm(xe)
        xr = vm.post_layernorm(xr)
        print(f"post_ln cos={cos(xe, xr):.10f}")

    # 对照旧落盘
    old_e = np.load(DIR + "/tower_walk_engine_cpu.npz")
    old_r = np.load(DIR + "/tower_walk_ref.npz")
    print("旧文件 emb0 cos =", cos(torch.from_numpy(old_e["emb0"]),
                                     torch.from_numpy(old_r["emb0"])))
    print("旧文件 l00 cos  =", cos(torch.from_numpy(old_e["l00"]),
                                     torch.from_numpy(old_r["l00"])))


if __name__ == "__main__":
    main()
