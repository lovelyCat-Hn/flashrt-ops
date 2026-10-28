#!/usr/bin/env python
"""一次性实证：这台 ckpt 上 embed_image/投影器/select_message 的真实维度链路。

同进程对照：_embed_images 直出形状 vs select_message 全跑通，锁定 1152/2048
矛盾的机制（2026-10-28 System 2 parity 逐层定位中间步）。
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
TOK = os.path.expanduser("~/holy/models/paligemma_sp224")
VIEWS = ("observation.images.cam_high",
         "observation.images.cam_left_wrist",
         "observation.images.cam_right_wrist")


def main() -> None:
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    policy = FineARTVLAPolicy.from_pretrained(CKPT)
    policy.eval()
    pre = DataProcessorPipeline.from_pretrained(
        CKPT, "policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"},
                   "fineart_vla_text_tokenizer": {"tokenizer_name": TOK},
                   "action_tokenizer_processor": {"paligemma_tokenizer_name": TOK}})

    imgs = [np.random.default_rng(8 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    batch = {"query_kind": "next_subtask", "query_text": "pick up the object"}
    for k, im in zip(VIEWS, imgs):
        batch[k] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    processed = pre(batch)

    pg = policy.model.paligemma_with_expert.paligemma
    proj = pg.model.multi_modal_projector
    print("multi_modal_projector:", type(proj).__name__)
    lin = getattr(proj, "linear", None)
    if lin is not None:
        print("  linear weight:", tuple(lin.weight.shape), lin.weight.dtype)

    backbone = policy.model.paligemma_with_expert
    with torch.no_grad():
        images, img_masks = policy._preprocess_images(processed)
        print("training flag:", policy.model.training)
        embs = backbone.embed_image(torch.cat(images, dim=0))
        print("embed_image(cat) →", tuple(embs.shape), embs.dtype)

        # 与 select_message 完全相同的调用
        tokens = processed["observation.language.tokens"]
        amask = processed["observation.language.attention_mask"]
        pe, pp, pa = policy.model.embed_prefix(images, img_masks, tokens, amask)
        print("embed_prefix embs →", tuple(pe.shape))

        policy.config.text_tokenizer_name = TOK
        text = policy.generate_text(processed)
        print("select_message →", repr(text))


if __name__ == "__main__":
    main()
