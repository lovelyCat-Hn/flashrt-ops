#!/usr/bin/env python
"""fineart_vla System 2 逐 token logits 判定（引擎 vs 官方参考）。

每步：argmax 一致性 + 全词表 logit cos（±inf 感知：对齐双侧有限位掩码）
+ top-1/top-2 margin（双侧各自有限域内）。落盘 <dir>/subtask_logits_compare.json。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")


def _cos_finite(a: np.ndarray, b: np.ndarray) -> float:
    """双侧有限位交集上的 cos（-inf 掩码区不进比对）。"""
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else float("nan")


def _margin(x: np.ndarray) -> tuple[float, float]:
    f = x[np.isfinite(x)]
    if f.size < 2:
        return float("nan"), float("nan")
    part = np.partition(f, -2)
    return float(part[-1]), float(part[-1] - part[-2])  # top1, top1-top2


def _margmax(x: np.ndarray) -> int:
    """有限域内的 argmax 原始 id（掩码区不参选）。"""
    f = np.flatnonzero(np.isfinite(x))
    return int(f[np.argmax(x[f])]) if f.size else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--ref", default="subtask_logits_ref_fp32.npz")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    e = np.load(d / "subtask_logits_engine.npz")
    r = np.load(d / args.ref)
    le, lr = e["logits"], r["logits"]
    n = int(min(le.shape[0], lr.shape[0]))

    steps = []
    for i in range(n):
        t1e, me = _margin(le[i])
        t1r, mr = _margin(lr[i])
        steps.append({
            "step": i,
            "argmax_equal": _margmax(le[i]) == _margmax(lr[i]),
            "chosen_equal": bool(e["tok_chosen"][i] == r["tok_chosen"][i]),
            "cos": _cos_finite(le[i], lr[i]),
            "margin_engine": me,
            "margin_ref": mr,
            "top1_engine": t1e,
            "top1_ref": t1r,
        })

    first_bad = next((s for s in steps if not s["chosen_equal"]), None)
    report = {
        "steps_engine": int(le.shape[0]), "steps_ref": int(lr.shape[0]),
        "compared_steps": n,
        "chosen_all_equal": bool(np.array_equal(e["tok_chosen"][:n], r["tok_chosen"][:n])),
        "argmax_all_equal": all(s["argmax_equal"] for s in steps),
        "cos_mean": float(np.nanmean([s["cos"] for s in steps])),
        "cos_min": float(np.nanmin([s["cos"] for s in steps])),
        "first_diverge": first_bad,
        "per_step": steps,
    }
    report["verdict"] = ("PASS" if report["chosen_all_equal"] and report["cos_min"] >= 0.999
                         else "PASS_structural" if report["cos_min"] >= 0.99 else "FAIL")
    (d / "subtask_logits_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    slim = {k: v for k, v in report.items() if k != "per_step"}
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
