#!/usr/bin/env python
"""SigLIP 塔逐层走查·参考侧（官方 HF SiglipVisionModel 直接走子模块）。

落盘 <dir>/tower_walk_ref.npz，键与引擎侧一一对应。
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

DEFAULT_CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")
VIEWS = ("observation.images.cam_high",
         "observation.images.cam_left_wrist",
         "observation.images.cam_right_wrist")
CHECKS = (0, 1, 2, 4, 8, 16, 26)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    policy = FineARTVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    tok_dir = os.path.expanduser("~/holy/models/paligemma_sp224")
    pre = DataProcessorPipeline.from_pretrained(
        args.ckpt, "policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"},
                   "fineart_vla_text_tokenizer": {"tokenizer_name": tok_dir},
                   "action_tokenizer_processor": {"paligemma_tokenizer_name": tok_dir}})

    batch = {"query_kind": "next_subtask", "query_text": "pick up the object"}
    for k, im in zip(VIEWS, imgs):
        batch[k] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    processed = pre(batch)

    vm = policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    with torch.no_grad():
        images, _ = policy._preprocess_images(processed)
        pixel = torch.cat(images, dim=0).float()
        x = vm.embeddings(pixel)
        out = {"emb0": x.detach().float().cpu().numpy()}
        for i, layer in enumerate(vm.encoder.layers):
            o = layer(x, None)  # SigLIP 双向，mask=None
            if hasattr(o, "last_hidden_state"):
                x = o.last_hidden_state
            elif isinstance(o, (tuple, list)):
                x = o[0]
            else:
                x = o
            if i in CHECKS:
                out[f"l{i:02d}"] = x.detach().float().cpu().numpy()
        out["post_ln"] = vm.post_layernorm(x).detach().float().cpu().numpy()

    np.savez(args.dir + "/tower_walk_ref.npz", **out)
    print("saved:", {k: v.shape for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
