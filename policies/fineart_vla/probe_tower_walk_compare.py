#!/usr/bin/env python
"""SigLIP 塔逐层走查判定：各检查点逐位置 cos（emb0 → 逐层 → post_ln）。"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    e = np.load(d / "tower_walk_engine.npz")
    r = np.load(d / "tower_walk_ref.npz")

    report = {}
    for k in e.files:
        if k not in r.files:
            continue
        a = e[k].astype(np.float64).reshape(-1, e[k].shape[-1])
        b = r[k].astype(np.float64).reshape(-1, r[k].shape[-1])
        n = min(len(a), len(b))
        cs = np.sum(a[:n] * b[:n], axis=1) / (
            np.linalg.norm(a[:n], axis=1) * np.linalg.norm(b[:n], axis=1) + 1e-30)
        report[k] = {"cos_mean": float(cs.mean()), "cos_min": float(cs.min())}
    (d / "tower_walk_compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
