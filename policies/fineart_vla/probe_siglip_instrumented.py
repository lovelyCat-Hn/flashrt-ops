#!/usr/bin/env python
"""C 调用全程仪表化：hooks 抓 _SiglipLayer.forward 每个子模块中间量 vs 手动分解。

定位「引擎整层 ≠ 引擎手动」的具体子步骤。lerobot_cpu 环境。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")


def cos(a, b) -> float:
    a = a.detach().float().cpu().numpy().astype(np.float64).ravel()
    b = b.detach().float().cpu().numpy().astype(np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def main() -> None:
    sys.path.insert(0, "/home/galbot/holy/FlashRT")
    from flash_rt.models.fineart_vla.text_decode_rtx import SubtaskDecoder

    dec = SubtaskDecoder(CKPT, device="cpu")
    my = dec.vision.layers[0]

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    with torch.no_grad():
        pixel, _ = dec._preprocess_images(imgs)
        ve = dec.vision
        xe = ve.patch_embedding(pixel).flatten(2).transpose(1, 2)
        xe = xe + ve.position_embedding(ve.position_ids)

        # ── hooks 抓整层调用内部量 ──
        cap: dict[str, torch.Tensor] = {}
        hooks = []
        for name in ("layer_norm1", "q_proj", "k_proj", "v_proj", "out_proj",
                     "layer_norm2", "fc1", "fc2"):
            hooks.append(getattr(my, name).register_forward_hook(
                lambda m, i, o, n=name: cap.__setitem__(n + "_out", o.detach())))

        C = my(xe)
        for h in hooks:
            h.remove()

        # ── 手动分解（同对象同输入） ──
        m = {}
        m["layer_norm1"] = my.layer_norm1(xe)
        b, s, _ = m["layer_norm1"].shape
        hd = 1152 // 16
        q = my.q_proj(m["layer_norm1"]).view(b, s, 16, hd).transpose(1, 2)
        k = my.k_proj(m["layer_norm1"]).view(b, s, 16, hd).transpose(1, 2)
        v = my.v_proj(m["layer_norm1"]).view(b, s, 16, hd).transpose(1, 2)
        att = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
        m["out_proj"] = my.out_proj(att.transpose(1, 2).reshape(b, s, -1))
        res = xe + m["out_proj"]
        m["layer_norm2"] = my.layer_norm2(res)
        m["fc1"] = my.fc1(m["layer_norm2"])
        m["fc2"] = my.fc2(torch.nn.functional.gelu(m["fc1"], approximate="tanh"))
        manual = res + m["fc2"]

        print(f"C vs manual      : {cos(C, manual):.12f}")
        for k2 in ("layer_norm1", "out_proj", "layer_norm2", "fc1", "fc2"):
            if k2 + "_out" in cap:
                print(f"{k2:12s} hook vs manual : {cos(cap[k2 + '_out'], m[k2]):.12f}")

        # 层对象运行时状态
        print("layer dtype:", next(my.parameters()).dtype)
        print("layer training:", my.training)
        print("layer extra attrs:", [n for n, _ in my.named_modules()
                                     if n and n not in
                                     ("layer_norm1", "self_attn", "q_proj", "k_proj",
                                      "v_proj", "out_proj", "layer_norm2", "mlp",
                                      "fc1", "fc2", "")])
        print("full named_modules:", [n for n, _ in my.named_modules()])
        print("forward src:", my.forward.__code__.co_filename,
              "lines", my.forward.__code__.co_firstlineno)


if __name__ == "__main__":
    main()
