#!/usr/bin/env python
"""fineart_vla 金标·引擎侧（flash_pyrt311 + CUDA，FlashRT fineart_vla 前端）。

与 golden_reference.py（lerobot CPU eager）构成 parity 对照：
  - 同 seed 合成输入：14 维 raw state（uniform ±2.5，seed 7）、3 视角不同
    seed 噪声图（224² uint8，seed 8/9/10）、固定 noise (50,32)（seed 123）；
  - prompt = fineart 动作路径口径 `User: {task}, State: {bins};\n`
    （state 归一化公式 2(x−q01)/(q99−q01)−1，q01/q99 取自 checkpoint
    normalizer safetensors，与 lerobot preprocessor 同源）；
  - 引擎 infer(noise=固定) → actions 已 unnormalize（norm_stats）。
落盘 <dir>/engine.npz（actions + prompt_ids + 全部输入），供 golden_compare.py
与参考侧逐维 cos 对照（判据 ≥0.999，跨构造位等不可比）。

用法：
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/policies/fineart_vla/golden_engine.py --dir ~/holy/evidence/fineart_parity
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")

DEFAULT_CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_parity")
VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")  # 引擎视角序 = config 序


def build_inputs(ckpt: str, seed: int, noise_seed: int):
    """Golden 输入定义——golden_reference.py 用同一函数公式复现，勿只改一处。"""
    rng = np.random.default_rng(seed)
    raw_state = rng.uniform(-2.5, 2.5, 14).astype(np.float32)
    from safetensors import safe_open

    with safe_open(os.path.join(ckpt, "policy_preprocessor_step_3_normalizer_processor.safetensors"),
                   framework="pt", device="cpu") as f:
        q01 = f.get_tensor("observation.state.q01").numpy().astype(np.float32)
        q99 = f.get_tensor("observation.state.q99").numpy().astype(np.float32)
    norm_state = (2.0 * (raw_state - q01) / (q99 - q01) - 1.0).astype(np.float32)
    imgs = [np.random.default_rng(seed + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    noise = np.random.default_rng(noise_seed).standard_normal((50, 32)).astype(np.float32)
    return raw_state, norm_state, imgs, noise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--task", default="pick up the object")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-seed", type=int, default=123)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    import torch
    from flash_rt.api import load_model

    raw_state, norm_state, imgs, noise = build_inputs(args.ckpt, args.seed, args.noise_seed)

    print(f"[1/3] load_model(config='fineart_vla') ← {args.ckpt}")
    t0 = time.time()
    model = load_model(args.ckpt, framework="torch", config="fineart_vla",
                       num_views=3, num_steps=10, action_dim=14)
    print(f"      {time.time() - t0:.1f}s")

    obs = dict(zip(VIEW_KEYS, imgs))
    print(f"[2/3] set_prompt({args.task!r}, state=normalized) + calibrate + infer(noise=固定)")
    model.set_prompt(args.task, norm_state)
    model.pipeline.calibrate_with_real_data([obs])
    t0 = time.time()
    out = model.infer(obs, noise=noise)
    acts = out["actions"] if isinstance(out, dict) else out
    acts = np.asarray(acts, dtype=np.float32)
    prompt_ids = list(model.pipeline._fineart_prompt_ids(args.task, norm_state))
    print(f"      infer {time.time() - t0:.2f}s → {acts.shape}")

    out_dir = pathlib.Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "engine.npz", actions=acts,
             prompt_ids=np.array(prompt_ids, dtype=np.int64),
             noise=noise, raw_state=raw_state, norm_state=norm_state)
    (out_dir / "engine_meta.json").write_text(json.dumps({
        "task": args.task, "seed": args.seed, "noise_seed": args.noise_seed,
        "ckpt": args.ckpt, "prompt_token_len": len(prompt_ids),
        "actions_shape": list(acts.shape),
    }, indent=2, ensure_ascii=False))
    print(f"[3/3] 落盘 {out_dir}/engine.npz  (prompt {len(prompt_ids)} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
