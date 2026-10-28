#!/usr/bin/env python
"""fineart_vla System 2 子任务 parity 判定（引擎 vs 官方参考）。

硬门 = 生成文本 strip 后逐字相等（贪心确定性 → 双侧应逐位一致）。
不匹配时：用本地 paligemma_sp224 tokenizer 切 token 序列，报首个分歧位置。
落盘 <dir>/subtask_compare.json。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_subtask_parity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--tok", default=os.path.expanduser("~/holy/models/paligemma_sp224"))
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    eng = json.loads((d / "subtask_engine.json").read_text())
    ref = json.loads((d / "subtask_ref.json").read_text())

    t_eng, t_ref = eng["text"], ref["text"]
    report = {
        "text_engine": t_eng,
        "text_ref": t_ref,
        "exact_match": t_eng == t_ref,
        "match_normalized": t_eng.strip() == t_ref.strip(),
    }

    if not report["exact_match"]:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tok)
        ids_e = tok.encode(t_eng, add_special_tokens=False)
        ids_r = tok.encode(t_ref, add_special_tokens=False)
        k = next((i for i, (a, b) in enumerate(zip(ids_e, ids_r)) if a != b),
                 min(len(ids_e), len(ids_r)))
        report["token_lens"] = {"engine": len(ids_e), "ref": len(ids_r)}
        report["first_diverge_idx"] = k
        report["tokens_engine_head"] = tok.convert_ids_to_tokens(ids_e[:k + 8])
        report["tokens_ref_head"] = tok.convert_ids_to_tokens(ids_r[:k + 8])

    report["verdict"] = "PASS" if report["exact_match"] else (
        "PASS_normalized" if report["match_normalized"] else "FAIL")
    (d / "subtask_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
