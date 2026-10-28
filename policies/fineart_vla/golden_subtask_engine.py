#!/usr/bin/env python
"""fineart_vla System 2 子任务生成·引擎侧（flash_pyrt311 + GPU）。

与 golden_subtask_ref.py（lerobot CPU eager select_message 贪心）构成 parity：
同 task、同 3 视角噪声图（seed 8/9/10），贪心 + <loc> 抑制 + eos 停。
落盘 <dir>/subtask_engine.json：生成文本 + 逐 token ids。
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

    from flash_rt.api import load_model

    print("[1/2] load_model + generate_subtask（贪心）")
    model = load_model(args.ckpt, framework="torch", config="fineart_vla",
                       num_views=3, num_steps=10, action_dim=14)
    # load_model 返回 VLAModel 包装，.pipeline 才是前端本体（smoke_text 同款）
    text = model.pipeline.generate_subtask(args.task, dict(zip(VIEW_KEYS, imgs)))
    print(f"    文本: {text!r}")

    out = pathlib.Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "subtask_engine.json").write_text(json.dumps(
        {"text": text, "task": args.task}, ensure_ascii=False, indent=2))
    print(f"[2/2] 落盘 {out}/subtask_engine.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
