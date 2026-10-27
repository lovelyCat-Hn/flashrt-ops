#!/usr/bin/env python
"""fineart parity 逐层定位·引擎侧（flash_pyrt311 + GPU）。

与 golden_engine.py 完全同源的同批输入，构造前端时开 denoise_trace，
infer 后切 _prefix_out 拿全前缀隐状态，落盘 stages_engine.npz：
  prefix_vision  (vseq, ENC_D)  引擎 LLM 输入端视觉 token 段之后的前缀隐状态
  prefix_lang    (plen, ENC_D)  前缀隐状态的语言段
  x / delta / timesteps         逐步去噪轨迹（归一化空间）
  raw_actions    (50, 32)       归一化空间最终输出
用法：
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/policies/fineart_vla/probe_stages_engine.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_engine import DEFAULT_CKPT, DEFAULT_DIR, VIEW_KEYS, build_inputs

import torch  # noqa: E402

from flash_rt.frontends.torch.fineart_vla_rtx import (  # noqa: E402
    FineARTVLATorchFrontendRtx,
)

TASK = "pick up the object"


def main() -> int:
    ckpt = DEFAULT_CKPT
    out_dir = pathlib.Path(DEFAULT_DIR)
    raw_state, norm_state, imgs, noise = build_inputs(ckpt, seed=7, noise_seed=123)

    print("[1/2] 直接构造前端（denoise_trace=True），复刻 load_model 参数")
    model = FineARTVLATorchFrontendRtx(ckpt, num_views=3, action_dim=14,
                                       num_steps=10, denoise_trace=True,
                                       prefix_features=True)
    model.set_prompt(TASK, norm_state)
    obs = dict(zip(VIEW_KEYS, imgs))
    model.calibrate_with_real_data([obs])

    print("[2/2] infer(noise=固定) + 切全前缀隐状态")
    out = model.infer(obs, noise=noise)
    es = int(model.pipeline.encoder_seq_len)
    vseq = int(model.pipeline.vision_seq_enc)
    plen = int(model._last_prompt_len)
    from flash_rt.frontends.torch import pi05_rtx as _p
    enc_d = getattr(_p, "ENC_D", None)
    if enc_d is None:
        enc_d = int(model._prefix_out.shape[-1])
    hidden = model._prefix_out.view(1, es, enc_d)[0]
    prefix_vision = hidden[:vseq].float().cpu().numpy()
    prefix_lang = hidden[vseq:vseq + plen].float().cpu().numpy()

    trace = out.get("denoise_trace") or {}
    # 输入端语言嵌入（fineart 覆写路径；带 state 不走缓存，无副作用）
    embeds, elen, _ = model._embed_prompt_cached(TASK, 128, norm_state)
    lang_embs_in = embeds[:elen].float().cpu().numpy()
    np.savez(
        out_dir / "stages_engine.npz",
        prefix_vision=prefix_vision, prefix_lang=prefix_lang,
        lang_embs_in=lang_embs_in,
        x=np.asarray(trace["x"], dtype=np.float32),
        delta=np.asarray(trace["delta"], dtype=np.float32),
        timesteps=np.asarray(trace["timesteps"], dtype=np.float32),
        raw_actions=np.asarray(out["raw_actions"], dtype=np.float32),
        prompt_len=np.int64(plen), vision_seq=np.int64(vseq),
        noise=np.asarray(noise, dtype=np.float32))
    print(f"    vseq={vseq} plen={plen} enc_d={enc_d} "
          f"x={trace['x'].shape} timesteps={trace['timesteps'][:3]}…")
    print(f"落盘 {out_dir}/stages_engine.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
