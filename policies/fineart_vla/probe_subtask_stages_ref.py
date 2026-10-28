#!/usr/bin/env python
"""fineart_vla System 2 前缀逐层探针·参考侧（lerobot_cpu，官方实现）。

镜像 golden_subtask_ref v2 生产链路，抓四段中间量：预处理像素域 /
视觉塔输出（projector forward hook 输入）/ 投影后嵌入 / 前缀全序列隐状态
（embed_prefix → paligemma forward，同 select_message 前缀路径）。
落盘 <dir>/subtask_stages_ref.npz。
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
LEROBOT_VIEW_KEYS = ("observation.images.cam_high",
                     "observation.images.cam_left_wrist",
                     "observation.images.cam_right_wrist")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--task", default="pick up the object")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    print("[1/3] 加载官方 policy + preprocessor", flush=True)
    policy = FineARTVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    tok_dir = os.path.expanduser("~/holy/models/paligemma_sp224")
    pre = DataProcessorPipeline.from_pretrained(
        args.ckpt, "policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"},
                   "fineart_vla_text_tokenizer": {"tokenizer_name": tok_dir},
                   "action_tokenizer_processor": {"paligemma_tokenizer_name": tok_dir}})

    batch = {"query_kind": "next_subtask", "query_text": args.task}
    for key, img in zip(LEROBOT_VIEW_KEYS, imgs):
        batch[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    processed = pre(batch)

    print("[2/3] 抓四段中间量", flush=True)
    backbone = policy.model.paligemma_with_expert
    proj_linear = backbone.paligemma.model.multi_modal_projector.linear
    captured_input = {}

    def _hook(mod, inp, out):  # noqa: ANN001 — 必须显式 return None（返回值会替换输出）
        captured_input.setdefault("vision_feats", inp[0].detach())
        return None

    hook = proj_linear.register_forward_hook(_hook)

    with torch.no_grad():
        images, img_masks = policy._preprocess_images(processed)      # [(1,3,224,224) [−1,1]]
        pixel = torch.cat(images, dim=0)                              # (3,3,224,224)
        print("      images:", [tuple(i.shape) for i in images],
                  "masks:", [tuple(m.shape) for m in img_masks], flush=True)
        tokens = processed["observation.language.tokens"]
        amask = processed["observation.language.attention_mask"]
        n_valid = int(amask.reshape(-1).sum().item())
        lang_ids = tokens[0, :n_valid].tolist()

        prefix_embs, prefix_pad, prefix_att = policy.model.embed_prefix(
            images, img_masks, tokens, amask)
        backbone_dtype = backbone.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        att_2d = policy.model._prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad, prefix_att), dtype=backbone_dtype)
        position_ids = torch.cumsum(prefix_pad, dim=1) - 1
        (vlm_out, _), _ = backbone.forward(
            attention_mask=att_2d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=False)
        prefix_out = vlm_out[0].float().cpu().numpy()                 # (778,2048) post-norm

        img_emb = prefix_embs[0, :768].float().cpu().numpy()          # (768,2048) 投影后嵌入
        vision_feats = captured_input.get("vision_feats")
        vision_feats = None if vision_feats is None else vision_feats.detach().float().cpu().numpy()
        lang_emb = prefix_embs[0, 768:768 + n_valid].float().cpu().numpy()
        pixel_np = pixel.float().cpu().numpy()
    hook.remove()

    print(f"[3/3] pixel min={pixel_np.min():.3f} max={pixel_np.max():.3f} "
          f"lang_valid={n_valid} ids={lang_ids}", flush=True)
    out = args.dir
    os.makedirs(out, exist_ok=True)
    save = {"pixel": pixel_np, "img_emb": img_emb, "lang_emb": lang_emb,
            "lang_ids": np.array(lang_ids, dtype=np.int64),
            "prefix_vision": prefix_out[:768], "prefix_lang": prefix_out[768:768 + n_valid],
            "prefix_last": prefix_out[767 + n_valid]}
    if vision_feats is not None:
        save["vision_feats"] = vision_feats
    np.savez(os.path.join(out, "subtask_stages_ref.npz"), **save)
    print(f"落盘 {out}/subtask_stages_ref.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
