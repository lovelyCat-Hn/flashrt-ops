#!/usr/bin/env python
"""fineart_vla System 2 前缀逐层探针·引擎侧（flash_pyrt311 + GPU）。

抓 SubtaskDecoder 的四段中间量：预处理像素域 / 视觉塔输出 / 投影后嵌入 /
前缀全序列隐状态（vision 768 + lang 有效位）。镜像 _prefix_forward 前向
（同一批 layers 调用），额外落全序列便于逐位置对比。
落盘 <dir>/subtask_stages_engine.npz。
"""
from __future__ import annotations

import argparse
import os

import numpy as np

DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")
VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")
TASK = "pick up the object"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/holy/models/fineart_vla_midtrained"))
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    import torch
    from flash_rt.frontends.torch.fineart_vla_rtx import FineARTVLATorchFrontendRtx

    model = FineARTVLATorchFrontendRtx(args.ckpt, num_views=3, action_dim=14,
                                       num_steps=10)
    dec = model._ensure_subtask_decoder()
    m = dec.model

    with torch.no_grad():
        pixel, img_masks = dec._preprocess_images(imgs)          # (3,3,224,224) fp32 [−1,1]
        vision_feats = dec.vision(pixel)                          # (3,256,1152)
        img_emb = dec._embed_images(pixel)                        # (3,256,2048) bf16
        lang_ids, lang_mask = dec._prompt_tokens(args.task)       # 128 右补 0
        lang_emb = m.embed_tokens(torch.tensor(lang_ids, device=dec.device)) * m.embed_scale

        # 镜像 _prefix_forward（保留全序列输出）
        img_pad = img_masks[:, None].expand(img_emb.shape[0], 256)
        lang_pad = torch.tensor(lang_mask, device=dec.device)[None, :]
        pad = torch.cat((img_pad.reshape(1, -1), lang_pad), dim=1)
        embs = torch.cat((img_emb.reshape(1, -1, 2048), lang_emb[None]), dim=1)
        bool_2d = pad[:, None, :] & pad[:, :, None]
        mask4d = dec._mask4d(bool_2d[:, None], embs.dtype)
        pos = (torch.cumsum(pad, dim=1) - 1).clamp_min(0)[0]
        cache = __import__("flash_rt.models.fineart_vla.text_decode_rtx",
                           fromlist=["_KVCache"])._KVCache(18)
        x = embs
        for layer in m.layers:
            x = layer(x, mask4d, pos, cache)
        prefix_out = m.norm(x)[0].float().cpu().numpy()           # (778,2048) post-norm

    n_valid = sum(lang_mask)
    out = args.dir
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "subtask_stages_engine.npz"),
             pixel=pixel.float().cpu().numpy(),
             vision_feats=vision_feats.float().cpu().numpy(),
             img_emb=img_emb.float().cpu().numpy(),
             lang_ids=np.array(lang_ids[:n_valid], dtype=np.int64),
             prefix_vision=prefix_out[:768], prefix_lang=prefix_out[768:768 + n_valid],
             prefix_last=prefix_out[767 + n_valid])
    print(f"pixel[{tuple(pixel.shape)}] min={pixel.min():.3f} max={pixel.max():.3f} "
          f"lang_valid={n_valid} ids={lang_ids[:n_valid]}")
    print(f"落盘 {out}/subtask_stages_engine.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
