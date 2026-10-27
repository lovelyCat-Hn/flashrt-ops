#!/usr/bin/env python
"""fineart_vla 金标·参考侧（lerobot_cpu conda env + CPU eager，官方 lerobot 实现）。

独立 oracle：跑官方 FineARTVLAPolicy 前向，与 golden_engine.py（FlashRT）对照。
输入与引擎侧同源（同 seed/同公式，见 engine 脚本 build_inputs 注释）。

CPU 注意：
  - checkpoint 权重含 FP32 张量，模型按 FP32 跑一遍（engine=bf16，跨精度
    cos 期望略有降低）；再 cast bf16 跑第二遍（同精度口径对照）；
  - preprocessor 里 device 步骤保存的是 cuda，用 overrides 强制 cpu；
  - text tokenizer 需要 google/paligemma-3b-pt-224——本机 huggingface.co
    不通，脚本内设 HF_ENDPOINT=https://hf-mirror.com。
输出 <dir>/reference.npz：fp32/bf16 两组 actions + 参考侧 token ids。

用法（环境装好后）：
  conda run -n lerobot_cpu python \
      ~/holy/policies/fineart_vla/golden_reference.py --dir ~/holy/evidence/fineart_parity
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

DEFAULT_CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_parity")
# lerobot config 图像特征序：cam_high, cam_left_wrist, cam_right_wrist
LEROBOT_VIEW_KEYS = ("observation.images.cam_high",
                     "observation.images.cam_left_wrist",
                     "observation.images.cam_right_wrist")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--task", default="pick up the object")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-seed", type=int, default=123)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    import torch
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    # ── 输入（与 golden_engine.build_inputs 同公式同 seed）──
    rng = np.random.default_rng(args.seed)
    raw_state = rng.uniform(-2.5, 2.5, 14).astype(np.float32)
    imgs = [np.random.default_rng(args.seed + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    noise = np.random.default_rng(args.noise_seed).standard_normal((50, 32)).astype(np.float32)

    print("[1/4] 加载官方 policy + pre/postprocessor（CPU）")
    t0 = time.time()
    policy = FineARTVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    pre = DataProcessorPipeline.from_pretrained(
        args.ckpt, "policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"}})
    post = DataProcessorPipeline.from_pretrained(args.ckpt, "policy_postprocessor.json")
    print(f"      {time.time() - t0:.1f}s")

    batch = {"observation.state": torch.from_numpy(raw_state), "task": args.task}
    for key, img in zip(LEROBOT_VIEW_KEYS, imgs):
        batch[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # HWC uint8 → CHW [0,1]

    print("[2/4] preprocessor（含 state 归一化 + fineart prompt tokenize）")
    processed = pre(batch)
    tokens = processed["observation.language.tokens"]
    ref_ids = tokens[0].tolist() if tokens.ndim == 2 else tokens.tolist()
    n_valid = int(processed["observation.language.attention_mask"].reshape(-1).sum().item())
    ref_ids_valid = ref_ids[:n_valid]
    print(f"      prompt tokens: {n_valid}/{len(ref_ids)}")

    print("[3/4] predict_action_chunk FP32（固定 noise）")
    t0 = time.time()
    with torch.no_grad():
        chunk32 = policy.predict_action_chunk(
            processed, noise=torch.from_numpy(noise).unsqueeze(0))
    acts32 = post({"action": chunk32})
    acts32 = np.asarray(acts32["action"] if isinstance(acts32, dict) else acts32,
                        dtype=np.float32).reshape(-1, chunk32.shape[-1])
    print(f"      FP32 {time.time() - t0:.1f}s → {acts32.shape}")

    print("[4/4] predict_action_chunk BF16（同 noise，对齐引擎精度口径）")
    t0 = time.time()
    policy_bf16 = policy.to(torch.bfloat16)
    with torch.no_grad():
        chunk16 = policy_bf16.predict_action_chunk(
            processed, noise=torch.from_numpy(noise).unsqueeze(0).to(torch.bfloat16))
    acts16 = post({"action": chunk16})
    acts16 = np.asarray(acts16["action"] if isinstance(acts16, dict) else acts16,
                        dtype=np.float32).reshape(-1, chunk16.shape[-1])
    print(f"      BF16 {time.time() - t0:.1f}s → {acts16.shape}")

    out_dir = pathlib.Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "reference.npz", actions_fp32=acts32, actions_bf16=acts16,
             prompt_ids=np.array(ref_ids_valid, dtype=np.int64),
             raw_state=raw_state, noise=noise)
    (out_dir / "reference_meta.json").write_text(json.dumps({
        "task": args.task, "seed": args.seed, "noise_seed": args.noise_seed,
        "ckpt": args.ckpt, "prompt_token_len": n_valid,
    }, indent=2, ensure_ascii=False))
    print(f"落盘 {out_dir}/reference.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
