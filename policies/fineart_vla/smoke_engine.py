#!/usr/bin/env python
"""fineart_vla × FlashRT 引擎冒烟（无新环境，flash_pyrt311 直跑；零运动，不碰机器人）。

目的（2026-10-27 用户定 scope）：验证推理引擎能否正常加载并推理
lerobot/FineART-VLA-subtask-midtrained 权重（9.35 GB，本地
~/holy/models/fineart_vla_midtrained）——是引擎能力验证，不是部署。

判据：
  1) load_model(config="fineart_vla") 成功（注册链：api 白名单 +
     hardware _PIPELINE_MAP/_SM87_ALLOWED + fineart_vla_rtx 子类）；
  2) set_prompt + infer 输出形状合理（chunk × action_dim 口径，含 batch 维
     也算过，打印即可）；
  3) 输出全程有限值（NaN/Inf 计数 = 0）；
  4) 延迟量级记录（对照 pi05 同机口径，仅记录不设阈值）。
结果落盘 ~/holy/evidence/<ts>_fineart_smoke/（result.json + actions.npz）。

用法：
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/policies/fineart_vla/smoke_engine.py --reps 5

二分口径：--as-pi05（config="pi05" 直喂 fineart 权重，绕过注册链——
转换器按名拉取、两个 lm_head 天然不被读。注册链出问题时用它定位是
"注册问题"还是"权重/前端问题"）。

注意：不设 FVK_PI05_RTX_FORCE_INT8（与 G1 在役一致走 bf16 口径）；
INT8 全量量化判死（记忆 09-30 消融），冒烟不碰量化。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

DEFAULT_CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
EVIDENCE_ROOT = os.path.expanduser("~/holy/evidence")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--as-pi05", action="store_true",
                    help='config="pi05" 直喂 fineart 权重（二分定位用）')
    ap.add_argument("--no-cuda-graph", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    import flash_rt
    from flash_rt.api import load_model

    config = "pi05" if args.as_pi05 else "fineart_vla"
    print(f"[1/4] load_model(config={config!r}, num_views=3) ← {args.ckpt}")
    t0 = time.time()
    model = load_model(args.ckpt, framework="torch", config=config,
                       num_views=3, num_steps=args.num_steps,
                       use_cuda_graph=not args.no_cuda_graph)
    print(f"      加载成功，耗时 {time.time() - t0:.1f}s")

    print(f"[2/4] set_prompt({args.prompt!r}) + 合成 3 视角观测")
    model.set_prompt(args.prompt)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": img, "wrist_image_right": img}

    print(f"[3/4] warmup×{args.warmup} + 计测×{args.reps}")
    for _ in range(args.warmup):
        model.infer(obs)
    lats_ms, outs = [], []
    for _ in range(args.reps):
        t0 = time.time()
        out = model.infer(obs)
        lats_ms.append((time.time() - t0) * 1000.0)
        outs.append(np.asarray(out))

    arr = outs[-1]
    nan = int(np.isnan(arr).sum())
    inf = int(np.isinf(arr).sum())
    lats = sorted(lats_ms)
    p50 = lats[len(lats) // 2]
    print(f"[4/4] 输出 shape={arr.shape} dtype={arr.dtype} "
          f"NaN={nan} Inf={inf} | abs均值={float(np.abs(arr).mean()):.4g} "
          f"幅度范围=[{float(arr.min()):.4g}, {float(arr.max()):.4g}]")
    print(f"      延迟 ms: p50={p50:.1f} min={lats[0]:.1f} max={lats[-1]:.1f}")

    ok = (nan == 0 and inf == 0 and arr.size > 0)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ev = pathlib.Path(EVIDENCE_ROOT) / f"{ts}_fineart_smoke"
    ev.mkdir(parents=True, exist_ok=True)
    np.savez(ev / "actions.npz", actions=arr)
    result = {
        "config": config, "ckpt": str(args.ckpt),
        "flash_rt_version": getattr(flash_rt, "__version__", "unknown"),
        "prompt": args.prompt, "num_steps": args.num_steps,
        "use_cuda_graph": not args.no_cuda_graph,
        "action_shape": list(arr.shape), "nan": nan, "inf": inf,
        "abs_mean": float(np.abs(arr).mean()),
        "latency_ms_p50": p50, "latency_ms_min": lats[0],
        "latency_ms_max": lats[-1], "pass": ok,
        "total_s": time.time() - t_all,
    }
    (ev / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"证据落盘：{ev}/{'result.json + actions.npz'}")

    print("✅ 冒烟通过" if ok else "❌ 冒烟未过（见上面 NaN/Inf/shape 行）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
