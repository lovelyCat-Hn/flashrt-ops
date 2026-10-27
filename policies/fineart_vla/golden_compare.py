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
    has_bf16 = "actions_bf16" in ref
    print(f"engine {e.shape} vs reference fp32 {ref['actions_fp32'].shape} "
          f"bf16 {ref['actions_bf16'].shape if has_bf16 else '无'}")

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
    r16 = ref["actions_bf16"].reshape(e.shape).astype(np.float64) if has_bf16 else None
    for name, r in (("bf16", r16), ("fp32", rf32)):
        if r is None:
            print(f"[2] engine vs {name}: 参考侧本遍未产出（跳过）")
            continue
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

    # 精度带 = bf16 参考自己 vs fp32 参考的逐维 cos——低幅维（夹爪）上
    # 引擎残差只要与带同量级即精度项，非结构分歧（2026-10-27 判据补充）。
    band = None
    if has_bf16:
        band = [float(_cos(r16[:, i], rf32[:, i])) for i in range(e.shape[-1])]
        report["band_per_dim_bf16_vs_fp32"] = [round(c, 6) for c in band]
        report["band_min_dim"] = min(band)
        print(f"[3] 精度带 bf16参考vs fp32参考: 逐维 min={min(band):.6f}")
        print(f"    逐维 band: {[round(c, 4) for c in band]}")

    judge_key = "cos_min_dim_bf16" if has_bf16 else "cos_min_dim_fp32"
    ok = tokens_equal and report[judge_key] >= 0.999
    report["pass"] = bool(ok)
    report["judge_key"] = judge_key
    # 结构判据：引擎残差落入精度带内（带值 −0.005 容差）
    if band is not None:
        e16 = report["cos_per_dim_bf16"]
        structural = tokens_equal and all(
            ev >= bd - 0.005 for ev, bd in zip(e16, band))
        report["pass_structural"] = bool(structural)
        report["structural_note"] = (
            "引擎残差 ≤ 精度带（bf16参考vs fp32参考）+0.005 容差，逐维成立"
            if structural else "存在超出精度带的结构性分歧维")
    (d / "compare.json").write_text(json.dumps(report, indent=2))
    print(f"{'✅ 金标通过' if ok else '❌ 严格阈值未过'}"
          f"（判据=token 逐位一致 且 逐维 cos({judge_key.split('_')[-1]}) ≥ 0.999）")
    if band is not None:
        print(f"{'✅ 结构判据通过' if report['pass_structural'] else '❌ 结构判据未过'}"
              "（引擎残差落入精度带内）")
    print(f"落盘 {d}/compare.json")
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    raise SystemExit(main())
