#!/usr/bin/env python
"""fineart_vla System 2 逐 token logits 探针·参考侧（lerobot_cpu，官方实现）。

镜像 golden_subtask_ref v2 生产链路（query_kind=next_subtask + generate_text），
包 FineARTVLAPolicy._sample_next_token 抓每步完整 logits。
落盘 <dir>/subtask_logits_ref.npz（--autocast 落 *_bf16.npz）。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from contextlib import nullcontext

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
    ap.add_argument("--autocast", action="store_true")
    args = ap.parse_args()

    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

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

    print("[2/3] select_message（探针包 _sample_next_token）", flush=True)
    policy.config.text_tokenizer_name = tok_dir
    captured: list[np.ndarray] = []
    chosen: list[int] = []
    orig = policy._sample_next_token

    def spy(logits, temperature, top_p):  # noqa: ANN001 — (B,V) 同官方调用点
        captured.append(logits.detach().float().cpu().numpy().reshape(-1))
        pick = orig(logits, temperature, top_p)
        chosen.append(int(pick.reshape(-1)[0]))
        return pick

    policy._sample_next_token = spy
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16)
           if args.autocast else nullcontext())
    with ctx, torch.no_grad():
        text = policy.generate_text(processed)
    policy._sample_next_token = orig

    logits = np.stack(captured) if captured else np.zeros((0, 1), np.float32)
    tok_argmax = np.array([int(np.argmax(l)) for l in logits], dtype=np.int64)
    tok_chosen = np.array(chosen, dtype=np.int64)

    tag = "bf16" if args.autocast else "fp32"
    out = pathlib.Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"subtask_logits_ref_{tag}.npz", logits=logits,
             tok_argmax=tok_argmax, tok_chosen=tok_chosen)
    (out / f"subtask_ref_{tag}_probe.json").write_text(json.dumps(
        {"text": text, "task": args.task, "dtype": tag,
         "steps": int(logits.shape[0]), "tok_ids": tok_chosen.tolist()},
        ensure_ascii=False, indent=2))
    print(f"[3/3] steps={logits.shape[0]} dtype={tag} text={text!r}", flush=True)
    print(f"落盘 {out}/subtask_logits_ref_{tag}.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
