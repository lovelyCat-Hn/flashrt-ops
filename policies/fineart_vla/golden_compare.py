#!/usr/bin/env python
"""fineart_vla 金标·对照（任意 python3，只读两份 npz 判 cos）。

判据（跨构造位等不可比，一律 cos）：token 序列必须逐位一致（prompt 口径
对齐的硬门），随后 actions 逐维 cos ≥ 0.999（bf16 对 bf16 为主判据，
fp32 参考列对照报告——差距主要来自引擎 bf16 内核 vs 参考 fp32 的精度差）。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/holy/evidence/fineart_parity"))
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    eng = np.load(d / "engine.npz")
    ref = np.load(d / "reference.npz")
    e = eng["actions"].astype(np.float64)
    print(f"engine {e.shape} vs reference fp32 {ref['actions_fp32'].shape} "
          f"bf16 {ref['actions_bf16'].shape}")

    ids_e = eng["prompt_ids"].tolist()
    ids_r = ref["prompt_ids"].tolist()
    tokens_equal = ids_e == ids_r
    print(f"[1] prompt tokens: engine {len(ids_e)} vs reference {len(ids_r)} → "
          f"{'✅ 逐位一致' if tokens_equal else '❌ 不一致'}")
    if not tokens_equal:
        for i, (x, y) in enumerate(zip(ids_e, ids_r)):
            if x != y:
                print(f"    首个分歧 idx {i}: engine {x} vs ref {y}")
                break

    report = {"tokens_equal": tokens_equal}
    rf32 = ref["actions_fp32"].reshape(e.shape).astype(np.float64)
    r16 = ref["actions_bf16"].reshape(e.shape).astype(np.float64)
    for name, r in (("bf16", r16), ("fp32", rf32)):
        cos_all = _cos(e, r)
        per_dim = [float(_cos(e[:, i], r[:, i])) for i in range(e.shape[-1])]
        diff = float(np.abs(e - r).max())
        report[f"cos_all_{name}"] = cos_all
        report[f"cos_per_dim_{name}"] = [round(c, 6) for c in per_dim]
        report[f"cos_min_dim_{name}"] = min(per_dim)
        report[f"max_abs_diff_{name}"] = diff
        print(f"[2] engine vs {name}: cos 全体={cos_all:.6f} 逐维 min={min(per_dim):.6f} "
              f"max|Δ|={diff:.4g}")
        print(f"    逐维 cos: {[round(c, 4) for c in per_dim]}")

    ok = tokens_equal and report["cos_min_dim_bf16"] >= 0.999
    report["pass"] = bool(ok)
    (d / "compare.json").write_text(json.dumps(report, indent=2))
    print(f"{'✅ 金标通过' if ok else '❌ 金标未过'}（判据=token 逐位一致 且 逐维 cos(bf16) ≥ 0.999）")
    print(f"落盘 {d}/compare.json")
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    raise SystemExit(main())
