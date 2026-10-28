#!/usr/bin/env python
"""fineart_vla System 2 子任务生成·参考侧（lerobot_cpu + CPU eager，官方生产链路）。

镜像 `PolicyRunner.query(kind="next_subtask")`（inference/policy_runner.py:289-308）：
  batch["query_kind"]="next_subtask" + batch["query_text"]=task + 3 视角图
  → policy_preprocessor（render 步用 ckpt recipe subtask turns，只设
    MESSAGES_RENDERED、无 message_streams → tokenizer 步 :384 追加
    "Assistant:" 尾巴 = 纯 runtime 查询路径）
  → policy.generate_text(processed)（modeling_fineart_vla.py:1315 =
    select_message 全默认参数：贪心 temperature=0、use_kv_cache=True、
    suppress_loc_tokens=False——与引擎 SubtaskDecoder special_ids 口径一致）
落盘 <dir>/subtask_ref.json。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from contextlib import nullcontext

# 本机对 huggingface.co/mirror 均无路由：任何 hub 触碰立即失败，
# 不允许 5 轮指数退避空转（2026-10-28 subtask 首跑教训）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

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
    ap.add_argument("--autocast", action="store_true",
                    help="CPU autocast bf16 跑生成（判 bf16-vs-fp32 argmax 翻转）")
    args = ap.parse_args()

    import torch
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    print("[1/3] 加载官方 policy + preprocessor", flush=True)
    t0 = time.time()
    policy = FineARTVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    # google/paligemma-3b-pt-224 在 hub 被屏蔽且本机无路由；本地 sp 原件转换
    # 的 fast tokenizer 与 HF 版逐 token 等价（2026-10-27 对拍实证）。
    tok_dir = os.path.expanduser("~/holy/models/paligemma_sp224")
    pre = DataProcessorPipeline.from_pretrained(
        args.ckpt, "policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"},
                   "fineart_vla_text_tokenizer": {"tokenizer_name": tok_dir},
                   "action_tokenizer_processor": {"paligemma_tokenizer_name": tok_dir}})
    print(f"      {time.time() - t0:.1f}s", flush=True)

    # 镜像 policy_runner.query：query_kind/query_text + 观测图（无 state，
    # 与引擎 generate_subtask 输入面一致；state 只进 low_level stream）
    batch = {"query_kind": "next_subtask", "query_text": args.task}
    for key, img in zip(LEROBOT_VIEW_KEYS, imgs):
        batch[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    print("[2/3] preprocessor（query 渲染 → User:…\\nAssistant: 尾巴）", flush=True)
    processed = pre(batch)
    tokens = processed["observation.language.tokens"]
    n_valid = int(processed["observation.language.attention_mask"].reshape(-1).sum().item())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    prompt_rendered = tok.decode([int(t) for t in tokens[0, :n_valid]])
    has_tail = prompt_rendered.endswith("Assistant:")
    print(f"      prompt {n_valid} tokens: {prompt_rendered!r}  (Assistant 尾巴: {has_tail})",
          flush=True)

    print("[3/3] generate_text → select_message 全默认（贪心，无 loc 抑制）", flush=True)
    # select_message 内部经 config.text_tokenizer_name 加载 loc tokenizer
    # （modeling_fineart_vla.py:1340，默认 hub id，不吃 pipeline overrides）；
    # 指到本地件（suppress_loc=False 时只用于 eos/decode）。
    policy.config.text_tokenizer_name = tok_dir
    t0 = time.time()
    # CPU autocast bf16：权重留 fp32、matmul 动态转 bf16（与 System 1 同配方；
    # policy.to(bf16) 会因 select_message 内 fp32 时间/噪声张量混 dtype 炸）。
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16)
           if args.autocast else nullcontext())
    with ctx, torch.no_grad():
        text = policy.generate_text(processed)
    print(f"      {time.time() - t0:.1f}s → {text!r}", flush=True)

    out = pathlib.Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "subtask_ref_bf16.json" if args.autocast else "subtask_ref.json"
    (out / name).write_text(json.dumps(
        {"text": text, "task": args.task, "path": "policy_runner.query(next_subtask)",
         "dtype": "bf16-autocast" if args.autocast else "fp32",
         "prompt_rendered": prompt_rendered, "prompt_token_len": n_valid,
         "assistant_tail": has_tail}, ensure_ascii=False, indent=2))
    print(f"落盘 {out}/{name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
