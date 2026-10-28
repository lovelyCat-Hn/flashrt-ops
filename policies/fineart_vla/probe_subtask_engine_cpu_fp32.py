#!/usr/bin/env python
"""结构判决探针：引擎 SubtaskDecoder 全 fp32（CPU）vs 参考 fp32（CPU）。

同装置同精度下，若逐步 argmax/chosen 全同且 cos≈1.0 → 引擎语言解码结构
正确，GPU bf16 下 0.9996 的差距=纯精度效应；若仍有差 → 语言路径结构残差。
生成口径与 probe_subtask_engine 相同（suppress_loc_tokens=False，
镜像生产 select_message）。lerobot_cpu 环境。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")
VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")
TASK = "pick up the object"


def _cos_finite(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else float("nan")


def _margin(x: np.ndarray) -> tuple[float, float]:
    f = x[np.isfinite(x)]
    part = np.partition(f, -2)
    return float(part[-1]), float(part[-1] - part[-2])


def _margmax(x: np.ndarray) -> int:
    f = np.flatnonzero(np.isfinite(x))
    return int(f[np.argmax(x[f])]) if f.size else -1


def main() -> None:
    sys.path.insert(0, "/home/galbot/holy/FlashRT")
    from flash_rt.models.fineart_vla.text_decode_rtx import SubtaskDecoder

    dec = SubtaskDecoder(CKPT, device="cpu")
    # 全量 fp32：语言权重从 bf16 升回 fp32；_embed_images 尾部硬编码
    # .to(bf16)（生产配方）也一并转 fp32，保证全链同精度
    dec.model.float()
    dec.lm_head = dec.lm_head.float()   # 裸张量（:308 直赋），.float() 需重赋值
    _orig_embed = dec._embed_images

    def embed_fp32(images):  # noqa: ANN001
        return _orig_embed(images).float()

    dec._embed_images = embed_fp32

    imgs = [np.random.default_rng(7 + 1 + i).integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for i in range(3)]

    captured: list[np.ndarray] = []
    chosen: list[int] = []
    orig = dec._sample_next_token

    def spy(logits, temperature, top_p):  # noqa: ANN001 — 镜像 generate 调用点
        captured.append(logits.detach().float().cpu().numpy().reshape(-1))
        pick = orig(logits, temperature, top_p)
        chosen.append(int(pick.reshape(-1)[0]))
        return pick

    dec._sample_next_token = spy
    with torch.no_grad():
        text = dec.generate(TASK, imgs, temperature=0.0, top_p=1.0,
                            suppress_loc_tokens=False)
    dec._sample_next_token = orig
    print("engine fp32 text:", text)

    le = np.stack(captured)
    r = np.load(DIR + "/subtask_logits_ref_fp32.npz")
    lr = r["logits"]
    n = int(min(le.shape[0], lr.shape[0]))
    print(f"steps: engine={le.shape[0]} ref={lr.shape[0]} compared={n}")
    all_ok = True
    for i in range(n):
        t1e, me = _margin(le[i])
        t1r, mr = _margin(lr[i])
        eq = _margmax(le[i]) == _margmax(lr[i])
        ce = bool(chosen[i] == r["tok_chosen"][i])
        c = _cos_finite(le[i], lr[i])
        all_ok &= eq and ce
        print(f"step {i}: argmax={eq!s:5} chosen={ce!s:5} cos={c:.10f} "
              f"m_eng={me:.3f} m_ref={mr:.3f}")
    np.savez(DIR + "/subtask_logits_engine_cpu_fp32.npz",
             logits=le, tok_argmax=np.array([_margmax(x) for x in le]),
             tok_chosen=np.array(chosen))
    print("chosen_all_equal:", all_ok)


if __name__ == "__main__":
    main()
