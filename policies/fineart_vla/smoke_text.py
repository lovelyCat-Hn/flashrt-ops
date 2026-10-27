#!/usr/bin/env python
"""fineart_vla System 2 文本解码冒烟（Stage 2b；无新环境，flash_pyrt311 直跑；零运动）。

目的：验证 flash_rt.models.fineart_vla.text_decode_rtx（纯 torch PaliGemma
移植，lerobot select_message 贪心口径）能否在引擎环境里出合理文本——
合成噪声图属 OOD，判据是机制不是语义质量：
  1) generate_subtask 全链跑通（SubtaskDecoder 懒构建 + 权重按名加载）；
  2) 贪心确定性：同输入重复跑输出逐字一致；
  3) 文本非空（贪心首 token 即 eos 会给空串，记录并标警）；
  4) 解码器在场时 System 1 回归不破（infer 形状+有限值）。
语义质量（子任务文本是否贴题）留待真图/金标环境判定，这里只记录。
结果落盘 ~/holy/evidence/<ts>_fineart_text_smoke/。

用法：
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/policies/fineart_vla/smoke_text.py --reps 3
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

# 与 smoke_engine 同口径（System 1 回归段用；文本解码不读它）
os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")

DEFAULT_CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
EVIDENCE_ROOT = os.path.expanduser("~/holy/evidence")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--prompt2", default="wipe the table")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-system1", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    import torch
    from flash_rt.api import load_model

    print(f"[1/5] load_model(config='fineart_vla') ← {args.ckpt}")
    t0 = time.time()
    model = load_model(args.ckpt, framework="torch", config="fineart_vla",
                       num_views=3, num_steps=10, action_dim=14)
    print(f"      加载成功，耗时 {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": img, "wrist_image_right": img}
    # System 1 回归段要跑 infer；文本解码本身不依赖图捕获仪式
    model.set_prompt(args.prompt)
    model.pipeline.calibrate_with_real_data([obs])

    print(f"[2/5] generate_subtask({args.prompt!r}) 首调（含 SubtaskDecoder 懒构建）")
    mem_before = torch.cuda.memory_allocated() // 2**20
    t0 = time.time()
    text1 = model.pipeline.generate_subtask(args.prompt, obs,
                                            max_new_tokens=args.max_new_tokens)
    first_s = time.time() - t0
    mem_delta = torch.cuda.memory_allocated() // 2**20 - mem_before
    print(f"      首调 {first_s:.1f}s（懒构建+权重），ΔVRAM {mem_delta} MiB")
    print(f"      生成文本: {text1!r}")

    print(f"[3/5] 贪心确定性 ×{args.reps} + 第二 prompt 敏感性")
    texts = [text1]
    lats = []
    for _ in range(args.reps - 1):
        t0 = time.time()
        texts.append(model.pipeline.generate_subtask(
            args.prompt, obs, max_new_tokens=args.max_new_tokens))
        lats.append((time.time() - t0) * 1000.0)
    deterministic = len(set(texts)) == 1
    t0 = time.time()
    text2 = model.pipeline.generate_subtask(args.prompt2, obs,
                                            max_new_tokens=args.max_new_tokens)
    lats.append((time.time() - t0) * 1000.0)
    lats.sort()
    p50 = lats[len(lats) // 2]
    print(f"      确定性: {'✅' if deterministic else '❌'} 重复输出={texts}")
    print(f"      prompt2={args.prompt2!r} → {text2!r}")
    print(f"      热调延迟 ms: p50={p50:.1f} min={lats[0]:.1f} max={lats[-1]:.1f}")

    s1_ok = True
    if not args.skip_system1:
        print("[4/5] System 1 回归（解码器在场）")
        out = model.infer(obs)
        if isinstance(out, dict):
            out = out.get("actions", next(iter(out.values())))
        if hasattr(out, "cpu"):  # torch 张量兜底（现役前端实际返回 numpy）
            arr = out.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(out, dtype=np.float32)
        nan = int(np.isnan(arr).sum())
        inf = int(np.isinf(arr).sum())
        s1_ok = nan == 0 and inf == 0 and arr.size > 0
        print(f"      infer shape={arr.shape} NaN={nan} Inf={inf} → "
              f"{'✅' if s1_ok else '❌'}")
    else:
        print("[4/5] System 1 回归跳过（--skip-system1）")

    ok = deterministic and s1_ok
    nonempty = bool(text1.strip())
    ts = time.strftime("%Y%m%d_%H%M%S")
    ev = pathlib.Path(EVIDENCE_ROOT) / f"{ts}_fineart_text_smoke"
    ev.mkdir(parents=True, exist_ok=True)
    result = {
        "ckpt": str(args.ckpt), "prompt": args.prompt, "prompt2": args.prompt2,
        "max_new_tokens": args.max_new_tokens,
        "text_first": text1, "text_repeats": texts, "text_prompt2": text2,
        "greedy_deterministic": deterministic,
        "text_nonempty": nonempty,
        "decode_latency_ms_p50": p50, "decode_first_call_s": first_s,
        "subtask_decoder_vram_mib": mem_delta,
        "system1_regression_pass": s1_ok,
        "pass": ok, "total_s": time.time() - t_all,
    }
    (ev / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (ev / "generated_texts.txt").write_text(
        f"prompt1={args.prompt!r}\n{text1}\n\nprompt1 重复 {args.reps} 次:\n"
        + "\n".join(texts) + f"\n\nprompt2={args.prompt2!r}\n{text2}\n")
    print(f"证据落盘：{ev}/")

    warn = "" if nonempty else "（⚠ 贪心首 token 即 eos → 空文本，OOD 合成图的已知可能）"
    print(f"{'✅' if ok else '❌'} 文本解码冒烟{'通过' if ok else '未过'}{warn}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
