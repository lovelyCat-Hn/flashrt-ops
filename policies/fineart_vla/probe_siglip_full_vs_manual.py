#!/usr/bin/env python
"""对照实验：SiglipEncoderLayer 整调 vs 手动分解（同输入同进程）+ attn 实现名。"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().numpy().astype(np.float64).ravel()
    b = b.detach().float().cpu().numpy().astype(np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def main() -> None:
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    policy = FineARTVLAPolicy.from_pretrained(CKPT)
    policy.eval()
    tok_dir = os.path.expanduser("~/holy/models/paligemma_sp224")
    pre = DataProcessorPipeline.from_pretrained(
        CKPT, "policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"},
                   "fineart_vla_text_tokenizer": {"tokenizer_name": tok_dir},
                   "action_tokenizer_processor": {"paligemma_tokenizer_name": tok_dir}})

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    batch = {"query_kind": "next_subtask", "query_text": "pick up the object"}
    for k, im in zip(("observation.images.cam_high",
                      "observation.images.cam_left_wrist",
                      "observation.images.cam_right_wrist"), imgs):
        batch[k] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    processed = pre(batch)

    vm = policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    print("attn implementation:", vm.config._attn_implementation)
    el = vm.encoder.layers[0]

    with torch.no_grad():
        images, _ = policy._preprocess_images(processed)
        pixel = torch.cat(images, dim=0).float()
        x = vm.embeddings(pixel)

        # 整调
        o = el(x, None)
        full = o.last_hidden_state if hasattr(o, "last_hidden_state") else o

        # 手动分解（bisect 同款）
        h = el.layer_norm1(x)
        att, _ = el.self_attn(h, None)
        man = x + att
        man = man + el.mlp(el.layer_norm2(man))

        print("layer(x,None) vs manual: cos =", cos(full, man))

        # sdpa vs eager 实现切换对照
        impl = vm.config._attn_implementation
        vm.config._attn_implementation = "eager"
        o2 = el(x, None)
        eager = o2.last_hidden_state if hasattr(o2, "last_hidden_state") else o2
        vm.config._attn_implementation = impl
        print("layer(eager) vs manual: cos =", cos(eager, man))


if __name__ == "__main__":
    main()
