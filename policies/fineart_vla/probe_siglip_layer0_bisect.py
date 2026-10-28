#!/usr/bin/env python
"""SigLIP layer0 终极二分：同进程同输入，引擎克隆 vs HF 官方层，逐子模块对比。

lerobot_cpu 环境跑（两侧同 torch）。先位等比对 layer0 权重，再逐步对：
ln1 → qkv → attn → out_proj → residual → ln2 → fc1 → gelu → fc2。
落盘 <dir>/siglip_layer0_bisect.json。
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
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
    el = vm.encoder.layers[0]
    my = dec.vision.layers[0]

    # ── 权重位等比对 ──
    pairs = [
        ("ln1.w", my.layer_norm1.weight, el.layer_norm1.weight),
        ("ln1.b", my.layer_norm1.bias, el.layer_norm1.bias),
        ("q.w", my.q_proj.weight, el.self_attn.q_proj.weight),
        ("q.b", my.q_proj.bias, el.self_attn.q_proj.bias),
        ("k.w", my.k_proj.weight, el.self_attn.k_proj.weight),
        ("k.b", my.k_proj.bias, el.self_attn.k_proj.bias),
        ("v.w", my.v_proj.weight, el.self_attn.v_proj.weight),
        ("v.b", my.v_proj.bias, el.self_attn.v_proj.bias),
        ("o.w", my.out_proj.weight, el.self_attn.out_proj.weight),
        ("o.b", my.out_proj.bias, el.self_attn.out_proj.bias),
        ("fc1.w", my.fc1.weight, el.mlp.fc1.weight),
        ("fc1.b", my.fc1.bias, el.mlp.fc1.bias),
        ("fc2.w", my.fc2.weight, el.mlp.fc2.weight),
        ("fc2.b", my.fc2.bias, el.mlp.fc2.bias),
        ("ln2.w", my.layer_norm2.weight, el.layer_norm2.weight),
        ("ln2.b", my.layer_norm2.bias, el.layer_norm2.bias),
    ]
    weights = {}
    for name, a, b in pairs:
        eq = bool(torch.equal(a.data, b.data))
        weights[name] = {"bit_equal": eq,
                         "shape_a": list(a.shape), "shape_b": list(b.shape),
                         "max_abs_diff": float((a.data - b.data).abs().max())}
    bad = [k for k, v in weights.items() if not v["bit_equal"]]
    print("权重位等：全部一致" if not bad else f"权重不一致：{bad}")

    # ── 同输入逐步前向 ──
    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    with torch.no_grad():
        pixel, _ = dec._preprocess_images(imgs)
        v = dec.vision
        x_in = v.patch_embedding(pixel).flatten(2).transpose(1, 2)
        x_in = x_in + v.position_embedding(v.position_ids)
        x_ref = vm.embeddings(pixel)
        x_in = x_in.float()
        x_ref = x_ref.float()

        trace = {"emb0": {"cos": cos(x_in, x_ref)}}
        with torch.no_grad():
            h_my = my.layer_norm1(x_in)
            h_rf = el.layer_norm1(x_ref)
            trace["ln1"] = {"cos": cos(h_my, h_rf)}

            # 引擎手写 attention vs HF self_attn
            b, s, _ = h_my.shape
            hd = 1152 // 16
            q = my.q_proj(h_my).view(b, s, 16, hd).transpose(1, 2)
            k = my.k_proj(h_my).view(b, s, 16, hd).transpose(1, 2)
            vv = my.v_proj(h_my).view(b, s, 16, hd).transpose(1, 2)
            att_my = torch.nn.functional.scaled_dot_product_attention(
                q, k, vv, scale=hd ** -0.5)
            att_my = my.out_proj(att_my.transpose(1, 2).reshape(b, s, -1))
            att_rf, _ = el.self_attn(h_rf, None)
            trace["attn_out_proj"] = {"cos": cos(att_my, att_rf)}

            r_my = x_in + att_my
            r_rf = x_ref + att_rf
            trace["residual1"] = {"cos": cos(r_my, r_rf)}

            h2_my = my.layer_norm2(r_my)
            h2_rf = el.layer_norm2(r_rf)
            trace["ln2"] = {"cos": cos(h2_my, h2_rf)}

            mlp_my = my.fc2(torch.nn.functional.gelu(my.fc1(h2_my), approximate="tanh"))
            mlp_rf = el.mlp(h2_rf)
            trace["mlp"] = {"cos": cos(mlp_my, mlp_rf)}
            trace["out"] = {"cos": cos(r_my + mlp_my, r_rf + mlp_rf)}

    report = {"weights": weights, "trace": trace}
    out = DIR + "/siglip_layer0_bisect.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(trace, indent=2))
    print("落盘", out)


if __name__ == "__main__":
    main()
