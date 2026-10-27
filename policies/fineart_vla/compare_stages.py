#!/usr/bin/env python
"""fineart parity 逐层定位·比对（任意 python3，只读两份 npz）。

三段定位，报告第一个分歧层（两侧布局同为 vision(768) → lang(55 valid)）：
  A  lang 输入端嵌入   engine _embed_prompt_cached vs ref embed_language_tokens
     （vision 输入端引擎在 CUDA 图内不可取，靠 B 段 vision 输出等价定位）
  B  LLM 输出隐状态    vision 段（3 视角分别）+ lang 段
  C  去噪轨迹          x_t 逐步 / euler 增量逐步 / 最终归一化动作
判读：cos 高=该层之前一致；首个塌掉的段=分歧层。
用法：python compare_stages.py [--dir ~/holy/evidence/fineart_parity]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def cos_tokens(a, b):
    n = min(len(a), len(b))
    return np.asarray([cos(a[i], b[i]) for i in range(n)])


def summarize(name, vals):
    v = np.asarray(vals, dtype=np.float64)
    line = (f"  {name}: mean={v.mean():+.4f} min={v.min():+.4f} "
            f"<0.9 计数={int((v < 0.9).sum())}/{len(v)}")
    print(line)
    return {"mean": float(v.mean()), "min": float(v.min()),
            "n_below_0.9": int((v < 0.9).sum()), "n": len(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/holy/evidence/fineart_parity"))
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    eng = np.load(d / "stages_engine.npz")
    ref = np.load(d / "stages_ref.npz")
    report: dict = {}

    ref_embs = ref["prefix_embs"][0]   # (896, 2048) 输入端
    ref_out = ref["prefix_out"][0]     # (896, 2048) 输出端
    e_vis_out = eng["prefix_vision"]   # (768, 2048) 输出端
    e_lang_out = eng["prefix_lang"]    # (55, 2048)  输出端
    e_lang_in = eng["lang_embs_in"]    # (55, 2048)  输入端

    print("[A] lang 输入端嵌入（prompt 嵌入路径）")
    report["A_lang_in"] = summarize("逐 token cos", cos_tokens(e_lang_in, ref_embs[768:768 + len(e_lang_in)]))

    print("[B] LLM 输出隐状态")
    for vi in range(3):
        sl = slice(vi * 256, (vi + 1) * 256)
        report[f"B_vis_view{vi}"] = summarize(
            f"vision view{vi} 逐 token cos", cos_tokens(e_vis_out[sl], ref_out[sl]))
    report["B_lang_out"] = summarize(
        "lang 段逐 token cos", cos_tokens(e_lang_out, ref_out[768:768 + len(e_lang_out)]))

    print("[C] 去噪轨迹")
    e_x, e_delta = eng["x"], eng["delta"]
    r_x, r_v = ref["x_traj"][:, 0], ref["v_traj"][:, 0]
    n = min(len(e_x), len(r_x))
    report["C_x_perstep"] = [float(cos(e_x[k], r_x[k])) for k in range(n)]
    print("    x_t 逐步 cos:     " + " ".join(f"{c:+.4f}" for c in report["C_x_perstep"]))
    r_delta = r_v * (-0.1)  # 引擎 delta = dt·v（dt=-0.1）
    report["C_delta_perstep"] = [float(cos(e_delta[k], r_delta[k])) for k in range(n)]
    print("    增量逐步 cos:     " + " ".join(f"{c:+.4f}" for c in report["C_delta_perstep"]))
    print(f"    times 引擎={eng['timesteps'].tolist()}")
    print(f"    times 参考={ref['times'].tolist()}")
    e_raw = eng["raw_actions"][:, :14]
    r_chunk = ref["chunk"].reshape(-1, e_raw.shape[-1])
    report["C_final_cos"] = float(cos(e_raw, r_chunk))
    print(f"    最终归一化动作 cos={report['C_final_cos']:+.5f}")

    (d / "stages_compare.json").write_text(json.dumps(report, indent=2))
    print(f"落盘 {d}/stages_compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
