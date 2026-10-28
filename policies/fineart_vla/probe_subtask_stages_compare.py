#!/usr/bin/env python
"""fineart_vla System 2 前缀逐层判定：像素域 → 视觉塔 → 投影 → 前缀隐状态。

每段报逐位置 cos 的 mean/min（跨构造位等不可比，cos 判据，见记忆
flashrt-cross-build-bit-compare-pitfall）；像素域与 lang_ids 应逐位一致。
落盘 <dir>/subtask_stages_compare.json。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")


def cos_rows(a: np.ndarray, b: np.ndarray) -> dict:
    # 按最后一维展平成行（引擎/参考布局可能差一个 batch 维：3×256 vs 768）
    a = np.asarray(a, np.float64).reshape(-1, a.shape[-1])
    b = np.asarray(b, np.float64).reshape(-1, b.shape[-1])
    n = min(len(a), len(b))
    cs = np.sum(a[:n] * b[:n], axis=1) / (
        np.linalg.norm(a[:n], axis=1) * np.linalg.norm(b[:n], axis=1) + 1e-30)
    return {"n": n, "cos_mean": float(cs.mean()), "cos_min": float(cs.min()),
            "cos_p05": float(np.percentile(cs, 5))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    e = np.load(d / "subtask_stages_engine.npz")
    r = np.load(d / "subtask_stages_ref.npz")

    report = {}
    pe, pr = e["pixel"], r["pixel"]
    report["pixel"] = {"shape": list(pe.shape),
                       "max_abs_diff": float(np.abs(pe - pr).max()),
                       "allclose": bool(np.allclose(pe, pr, atol=1e-5))}
    report["lang_ids_equal"] = bool(np.array_equal(e["lang_ids"], r["lang_ids"]))
    for key in ("vision_feats", "img_emb", "prefix_vision", "prefix_lang", "prefix_last"):
        if key in e.files and key in r.files:
            report[key] = cos_rows(e[key], r[key])
    first_bad = next((k for k in ("vision_feats", "img_emb", "prefix_vision",
                                  "prefix_lang", "prefix_last")
                      if k in report and report[k].get("cos_min", 1) < 0.999), None)
    report["first_stage_below_0.999"] = first_bad
    (d / "subtask_stages_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
