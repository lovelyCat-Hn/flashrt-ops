#!/usr/bin/env python
"""fineart parity 逐层定位·参考侧（lerobot_cpu + CPU eager，官方实现）。

与 golden_reference.py 同源同输入，用方法包装抓中间量，落盘 stages_ref.npz：
  prefix_embs   (1, L, 2048)  LLM 输入端前缀嵌入（vision+lang）
  prefix_out    (1, L, 2048)  paligemma LLM 输出（use_cache 首调返回）
  x_traj        (10, 1, 50, 32) 逐步 x_t
  v_traj        (10, 1, 50, 32) 逐步速度场输出
  times         (10,)          逐步 flow time
  chunk         (1, 50, 14)    最终归一化动作
用法（建议 OMP_NUM_THREADS=8）：
  conda run -n lerobot_cpu python -u \
      ~/holy/policies/fineart_vla/probe_stages_ref.py 2>&1 | tee log
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

DEFAULT_CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_parity")
LEROBOT_VIEW_KEYS = ("observation.images.cam_high",
                     "observation.images.cam_left_wrist",
                     "observation.images.cam_right_wrist")
TASK = "pick up the object"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    import torch
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    rng = np.random.default_rng(7)
    raw_state = rng.uniform(-2.5, 2.5, 14).astype(np.float32)
    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]
    noise = np.random.default_rng(123).standard_normal((50, 32)).astype(np.float32)

    print("[1/4] 加载官方 policy + preprocessor", flush=True)
    t0 = time.time()
    policy = FineARTVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    tok_dir = os.path.expanduser("~/holy/models/paligemma_sp224")
    pre = DataProcessorPipeline.from_pretrained(
        args.ckpt, "policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"},
                   "fineart_vla_text_tokenizer": {"tokenizer_name": tok_dir},
                   "action_tokenizer_processor": {"paligemma_tokenizer_name": tok_dir}})
    print(f"      {time.time() - t0:.1f}s", flush=True)

    batch = {"observation.state": torch.from_numpy(raw_state), "task": TASK}
    for key, img in zip(LEROBOT_VIEW_KEYS, imgs):
        batch[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    processed = pre(batch)

    print("[2/4] 挂钩：embed_prefix / paligemma_with_expert.forward / denoise_step", flush=True)
    cap: dict = {}
    m = policy.model
    orig_ep = m.embed_prefix

    def ep(*a, **k):
        out = orig_ep(*a, **k)
        cap["prefix_embs"] = out[0].detach().float().cpu().numpy()
        return out

    m.embed_prefix = ep

    pwe = m.paligemma_with_expert
    orig_fwd = pwe.forward

    def fwd(*a, **k):
        out = orig_fwd(*a, **k)
        try:
            first = out[0]
            if isinstance(first, (list, tuple)):
                if cap.get("prefix_out") is None and first[0] is not None:
                    cap["prefix_out"] = first[0].detach().float().cpu().numpy()
                if len(first) > 1 and first[1] is not None:
                    cap["suffix_out_last"] = first[1].detach().float().cpu().numpy()
        except (TypeError, IndexError, AttributeError):
            pass
        return out

    pwe.forward = fwd

    orig_ds = m.denoise_step

    def ds(*a, **k):
        x_t = k.get("x_t", a[2] if len(a) > 2 else None)
        t = k.get("timestep", a[3] if len(a) > 3 else None)
        v = orig_ds(*a, **k)
        cap.setdefault("x_traj", []).append(x_t.detach().float().cpu().numpy())
        cap.setdefault("v_traj", []).append(v.detach().float().cpu().numpy())
        cap.setdefault("times", []).append(float(t.reshape(-1)[0]))
        return v

    m.denoise_step = ds

    print("[3/4] predict_action_chunk（固定 noise）", flush=True)
    t0 = time.time()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(
            processed, noise=torch.from_numpy(noise).unsqueeze(0))
    print(f"      {time.time() - t0:.1f}s → {tuple(chunk.shape)}", flush=True)

    tokens = processed["observation.language.tokens"]
    n_valid = int(processed["observation.language.attention_mask"].reshape(-1).sum().item())

    print("[4/4] 落盘", flush=True)
    out_dir = pathlib.Path(args.dir)
    np.savez(
        out_dir / "stages_ref.npz",
        prefix_embs=cap["prefix_embs"],
        prefix_out=cap["prefix_out"],
        x_traj=np.stack(cap["x_traj"]),
        v_traj=np.stack(cap["v_traj"]),
        times=np.asarray(cap["times"], dtype=np.float64),
        chunk=chunk.detach().float().cpu().numpy(),
        prompt_ids=tokens[0, :n_valid].numpy().astype(np.int64),
        suffix_out_last=cap.get("suffix_out_last", np.zeros(1, np.float32)))
    print(f"落盘 {out_dir}/stages_ref.npz  "
          f"(prefix_embs {cap['prefix_embs'].shape}, x_traj {len(cap['x_traj'])} 步)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
