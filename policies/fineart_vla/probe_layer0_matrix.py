#!/usr/bin/env python
"""layer0 六方对拍矩阵：引擎整层 / HF 整层 / HF 手动 / 引擎手动 + 权重三方位等 + 类源。

单进程同输入。一次划掉全部剩余可能性。
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
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy

    dec = SubtaskDecoder(CKPT, device="cpu")
    policy = FineARTVLAPolicy.from_pretrained(CKPT)
    policy.eval()
    vm = policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    my, el = dec.vision.layers[0], vm.encoder.layers[0]

    print("engine layer cls :", type(my).__module__ + "." + type(my).__name__)
    print("engine fwd source:", my.forward.__code__.co_filename)
    print("hf     layer cls :", type(el).__module__ + "." + type(el).__name__)
    print("hf     fwd source:", el.forward.__code__.co_filename)
    print("hf     attn cls  :", type(el.self_attn).__module__ + "." + type(el.self_attn).__name__)
    print("hf     attn src  :", el.self_attn.forward.__code__.co_filename)

    # 权重三方位等：引擎模块 vs HF 模块 vs ckpt 原始张量
    from safetensors.torch import safe_open
    with safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt") as f:
        keys = {k: f.get_tensor(k) for k in f.keys()
                if "encoder.layers.0" in k and ("q_proj" in k or "layer_norm1" in k)}
    wcheck = {
        "eng_q_w_eq_hf": bool(torch.equal(my.q_proj.weight.data, el.self_attn.q_proj.weight.data)),
        "eng_q_w_eq_raw": bool(torch.equal(my.q_proj.weight.data,
                                           keys.get("paligemma_with_expert.paligemma.model."
                                                    "vision_tower.vision_model.encoder.layers.0"
                                                    ".self_attn.q_proj.weight",
                                                    torch.empty(0)))),
        "hf_ln1_w_eq_raw": bool(torch.equal(el.layer_norm1.weight.data,
                                            keys.get("paligemma_with_expert.paligemma.model."
                                                     "vision_tower.vision_model.encoder.layers.0"
                                                     ".layer_norm1.weight",
                                                     torch.empty(0)))),
    }
    print("权重三方：", wcheck)

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    with torch.no_grad():
        pixel, _ = dec._preprocess_images(imgs)
        ve = dec.vision
        xe = ve.patch_embedding(pixel).flatten(2).transpose(1, 2)
        xe = xe + ve.position_embedding(ve.position_ids)
        xr = vm.embeddings(pixel.float())
        print(f"emb0       cos(eng,hf)={cos(xe, xr):.12f}")

        C = ve.layers[0](xe)                       # 引擎整层
        D_o = el(xr.float(), None)                 # HF 整层
        D = D_o.last_hidden_state if hasattr(D_o, "last_hidden_state") else D_o

        # 手动分解（同一输入 xr 基底）
        h = el.layer_norm1(xr.float())
        att, _ = el.self_attn(h, None)
        E = xr.float() + att
        E = E + el.mlp(el.layer_norm2(E))
        F = xe + my.fc2(torch.nn.functional.gelu(my.fc1(my.layer_norm2(xe))))

    print(f"C(eng_full)  vs D(hf_full)   : {cos(C, D):.12f}")
    print(f"C(eng_full)  vs E(hf_manual) : {cos(C, E):.12f}")
    print(f"C(eng_full)  vs F(eng_manual): {cos(C, F):.12f}")
    print(f"D(hf_full)   vs E(hf_manual) : {cos(D, E):.12f}")
    print(f"D(hf_full)   vs F(eng_manual): {cos(D, F):.12f}")


if __name__ == "__main__":
    main()
