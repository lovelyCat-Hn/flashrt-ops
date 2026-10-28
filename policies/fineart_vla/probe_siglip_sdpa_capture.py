#!/usr/bin/env python
"""全捕获：forward 内 q/k/v/out_proj 的输入输出全抓，逐步对照手动 + 确定性检查。"""
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

        cap: dict[str, torch.Tensor] = {}

        def mk(n):  # 抓输入+输出，显式 return None
            def h(m, i, o):
                cap[n + "_in"] = i[0].detach()
                cap[n + "_out"] = o.detach()
                return None
            return h

        hooks = [getattr(my, n).register_forward_hook(mk(n))
                 for n in ("layer_norm1", "q_proj", "k_proj", "v_proj", "out_proj")]
        C1 = my(xe)
        C2 = my(xe)                      # 确定性检查
        for h in hooks:
            h.remove()
        print(f"C1 vs C2（同调用两遍）: {cos(C1, C2):.12f}")

        # 手动分解（hook 已摘）
        h1 = my.layer_norm1(xe)
        b, s, _ = h1.shape
        hd = 1152 // 16
        q = my.q_proj(h1).view(b, s, 16, hd).transpose(1, 2)
        k = my.k_proj(h1).view(b, s, 16, hd).transpose(1, 2)
        v = my.v_proj(h1).view(b, s, 16, hd).transpose(1, 2)
        att = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
        att_flat = att.transpose(1, 2).reshape(b, s, -1)
        res = xe + my.out_proj(att_flat)

        print(f"h1 vs hook.layer_norm1_out : {cos(h1, cap['layer_norm1_out']):.12f}")
        print(f"q  vs hook.q_proj_out      : {cos(q, cap['q_proj_out']):.12f}")
        print(f"k  vs hook.k_proj_out      : {cos(k, cap['k_proj_out']):.12f}")
        print(f"v  vs hook.v_proj_out      : {cos(v, cap['v_proj_out']):.12f}")
        print(f"att_flat vs hook.out_proj_IN: {cos(att_flat, cap['out_proj_in']):.12f}")
        print(f"att_flat shape={tuple(att_flat.shape)} hook_in shape={tuple(cap['out_proj_in'].shape)}")
        print(f"my.scale={my.scale!r}  hd**-0.5={hd ** -0.5!r}")
        print(f"C1 vs (xe+out_proj(att)) manual : {cos(C1, res + my.fc2(torch.nn.functional.gelu(my.fc1(my.layer_norm2(res)), approximate='tanh'))):.12f}")


if __name__ == "__main__":
    main()
