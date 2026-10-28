#!/usr/bin/env python
"""fineart_vla System 2 逐 token logits 探针·引擎侧（flash_pyrt311 + GPU）。

镜像 golden_subtask_engine 输入（seed 8/9/10 噪声图 + 同 task），包
SubtaskDecoder._sample_next_token 抓每步完整 logits（suppress_loc=False，
与生产 select_message 口径一致，保原始 logit 可比）。
落盘 <dir>/subtask_logits_engine.npz：logits (T,V) + token ids + text。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

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

    from flash_rt.frontends.torch.fineart_vla_rtx import FineARTVLATorchFrontendRtx

    model = FineARTVLATorchFrontendRtx(args.ckpt, num_views=3, action_dim=14,
                                       num_steps=10)
    decoder = model._ensure_subtask_decoder()

    captured: list[np.ndarray] = []
    chosen: list[int] = []
    orig = decoder._sample_next_token

    def spy(logits, temperature, top_p):  # noqa: ANN001 — 签名镜像 generate 调用点
        captured.append(logits.detach().float().cpu().numpy().reshape(-1))
        pick = orig(logits, temperature, top_p)
        chosen.append(int(pick.reshape(-1)[0]))
        return pick

    decoder._sample_next_token = spy
    text = decoder.generate(args.task, imgs, temperature=0.0, top_p=1.0,
                            suppress_loc_tokens=False)
    decoder._sample_next_token = orig

    logits = np.stack(captured) if captured else np.zeros((0, 1), np.float32)
    tok_argmax = np.array([int(np.argmax(l)) for l in logits], dtype=np.int64)
    tok_chosen = np.array(chosen, dtype=np.int64)

    out = pathlib.Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "subtask_logits_engine.npz", logits=logits,
             tok_argmax=tok_argmax, tok_chosen=tok_chosen)
    (out / "subtask_engine_probe.json").write_text(json.dumps(
        {"text": text, "task": args.task, "steps": int(logits.shape[0]),
         "tok_ids": tok_chosen.tolist()}, ensure_ascii=False, indent=2))
    print(f"steps={logits.shape[0]} text={text!r}")
    print(f"落盘 {out}/subtask_logits_engine.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
