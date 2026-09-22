#!/usr/bin/env python
"""归一化语义对齐校验（G1 16 维部署，接入微调权重前/后都跑）。

Part A  纯数学往返：数据集统计两种语义 × {随机样本, q01/q99 角点, min/max 角点}
        normalize → unnormalize 恒等，max err < 1e-5
Part B  state 归一化落域：典型带（q01↔q99）采样须落 [-1,1]；常量窄维风险提示
Part C  引擎语义分发实证：同一模型实例，翻转 norm_stats 重放同一噪声。
        raw 归一化输出逐位一致，两种语义反解出的 raw 应一致、mean_std 输出
        应符合其仿射式——孤立证明"FlashRT 真的按 norm_mode 标记分发"
Part D  端到端形状 (10, action_dim) + 输出相对数据集 q01/q99 的落域

用法（先 unset LD_LIBRARY_PATH PYTHONPATH）:
  ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/eval/norm_align_check.py \
      [--ckpt <用于引擎实证的 ckpt>] [--dataset DIR] [--skip-model]
接入真 G1 微调权重后：--ckpt <微调部署目录>（其 norm_stats 写哪种语义就验证哪种；
两种语义的仿射式对照始终用数据集统计做基准）。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="/home/galbot/datasets/pick_place_balence/pick_place_balence")
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_smoke")
ap.add_argument("--skip-model", action="store_true")
args = ap.parse_args()

sys.path.insert(0, "/home/galbot/holy/FlashRT")
from flash_rt.core.utils.actions import normalize_state, unnormalize_actions  # noqa: E402

fails = 0


def check(ok, msg):
    global fails
    print(f"  {'✓' if ok else '✗'} {msg}")
    if not ok:
        fails += 1


stats = json.loads((pathlib.Path(args.dataset) / "meta" / "stats.json").read_text())
ACT, ST = stats["action"], stats["observation.state"]
ACT_DIM, ST_DIM = len(ACT["mean"]), len(ST["mean"])
rng = np.random.default_rng(0)

# ── Part A 纯数学往返 ──
print(f"== Part A 数学往返（action {ACT_DIM} 维）==")
for mode, lo_k, hi_k in (("q01_q99", "q01", "q99"), ("mean_std", "mean", "std")):
    ns = {"norm_mode": mode, "actions": {k: ACT[k] for k in (lo_k, hi_k)}}
    lo, hi = np.array(ACT[lo_k]), np.array(ACT[hi_k])
    samples = np.vstack([
        rng.uniform(lo, hi, size=(64, ACT_DIM)),   # 域内随机
        lo[None], hi[None],                        # 角点
        np.array(ACT["min"])[None], np.array(ACT["max"])[None],
    ])
    if mode == "mean_std":
        normed = (samples - lo) / (hi + 1e-6)
    else:
        normed = (samples - lo) / (hi - lo + 1e-6) * 2 - 1
    back = unnormalize_actions(normed.astype(np.float32), ns)
    err = float(np.abs(back - samples).max())
    check(err < 1e-5, f"{mode:8s} normalize→unnormalize 往返 max err = {err:.2e}")

# ── Part B state 落域 ──
print(f"== Part B state 归一化落域（state {ST_DIM} 维，q01_q99 语义）==")
ns_q = {"norm_mode": "q01_q99", "state": {"q01": ST["q01"], "q99": ST["q99"]}}
q01a, q99a = np.array(ST["q01"]), np.array(ST["q99"])
st_n_typ = normalize_state(rng.uniform(q01a, q99a, size=(256, ST_DIM)), ns_q)
outside = float((np.abs(st_n_typ) > 1.0).mean())
check(outside < 0.01,
      f"q01..q99 典型带采样 → [-1,1] 外比例 {outside:.1%}（应≈0；高了=分布漂移）")
st_n_full = normalize_state(
    rng.uniform(np.array(ST["min"]), np.array(ST["max"]), size=(256, ST_DIM)), ns_q)
print(f"  [info] min..max 全域采样越界比例 {float((np.abs(st_n_full) > 1).mean()):.1%}"
      f"（尾部饱和属正常，离散化打边界 bin）")
width = q99a - q01a
narrow = [f"dim{i}(宽{w:.1e})" for i, w in enumerate(width) if w < 1e-3]
if narrow:
    print(f"  [warn] 常量窄维 {len(narrow)} 个: {', '.join(narrow[:6])}"
          f"{' ...' if len(narrow) > 6 else ''}；部署读数须与训练值同量级，否则打满 bin")
print(f"  数据集 mean 归一化后: {np.round(normalize_state(np.array(ST['mean'], np.float32), ns_q), 3).tolist()}")

# ── Part C/D 引擎实证 ──
if not args.skip_model:
    import os
    for v in ("LD_LIBRARY_PATH", "PYTHONPATH", "LD_PRELOAD"):
        os.environ.pop(v, None)
    os.environ.setdefault("PI05_NO_GRAPH", "1")

    import functools
    import torch  # noqa: E402
    import flash_rt  # noqa: E402
    import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402
    _orig = _fe.Pi05TorchFrontendRtx.__init__

    @functools.wraps(_orig)
    def _no_graph(self, *a, **kw):
        kw["use_cuda_graph"] = False
        _orig(self, *a, **kw)
    _fe.Pi05TorchFrontendRtx.__init__ = _no_graph

    def _find_stats(ck: str) -> dict:
        """与 FlashRT load_norm_stats 相同的候选顺序取 ckpt 自带 stats。"""
        cands = [pathlib.Path(ck) / "norm_stats.json"]
        cands += sorted(pathlib.Path(ck).glob("assets/**/norm_stats.json"))
        cands += [pathlib.Path(ck) / "meta" / "stats.json"]
        for c in cands:
            if c.exists():
                d = json.loads(c.read_text())
                return d.get("norm_stats", d) if isinstance(d, dict) else d
        raise FileNotFoundError(f"{ck} 下无 norm/stats json")

    if not pathlib.Path(args.ckpt).exists():
        print(f"== Part C/D 跳过：{args.ckpt} 不存在 ==")
    else:
        ns = _find_stats(args.ckpt)
        mode0 = ns.get("norm_mode", "q01_q99")
        print(f"== Part C/D ckpt={args.ckpt} 加载语义={mode0} ==")
        # state 归一化：维数按 ckpt 自己的 state 统计走（smoke=libero 8 维，
        # 真 G1 微调 ckpt=23 维）；smoke 维数不足时截数据集 mean 前 d_st 维
        st_block = ns.get("state") or ns["actions"]
        d_st = len(st_block["q01" if mode0 == "q01_q99" else "mean"])
        state_raw = np.array(ST["mean"], np.float32)[:d_st]
        if d_st != ST_DIM:
            print(f"  [note] smoke 用 libero {d_st} 维 state（截数据集 mean），"
                  f"真 G1 ckpt 将用全 {ST_DIM} 维")
        st_norm = normalize_state(state_raw, ns)
        print(f"  归一化 state({d_st} 维): 前4维 {np.round(st_norm[:4], 3)}，"
              f"[-1,1] 外 {float((np.abs(st_norm) > 1).mean()):.1%}")

        m = flash_rt.load_model(args.ckpt, config="pi05", num_views=3,
                                cache_frames=1, action_dim=ACT_DIM)
        loaded_mode = (m._pipe.norm_stats.get("norm_mode", "q01_q99")
                       if isinstance(m._pipe.norm_stats, dict) else None)
        check(loaded_mode == mode0,
              f"引擎加载路径读到语义标记: norm_stats.norm_mode={loaded_mode!r}")
        PROMPT = "Left arm pick up the block. Right arm pick up the block."
        obs = {k: np.zeros((224, 224, 3), np.uint8)
               for k in ("image", "wrist_image", "wrist_image_right")}
        m.predict(obs, prompt=PROMPT, state=st_norm)     # 建管线暖场

        noise = torch.randn(10, 32)
        r1 = m._pipe.infer(obs, noise=noise)
        out_A = np.asarray(r1["actions"] if isinstance(r1, dict) else r1)
        r1b = m._pipe.infer(obs, noise=noise)
        out_A2 = np.asarray(r1b["actions"] if isinstance(r1b, dict) else r1b)
        err_det = float(np.abs(out_A - out_A2).max())
        check(err_det == 0.0, f"同实例同噪声确定性: max diff = {err_det}")
        print(f"  [{mode0}] 输出 {out_A.shape}，范围 [{out_A.min():.3f}, {out_A.max():.3f}]")

        # Part C：同实例翻转 stats 重放同噪声 → raw 逐位一致，只应差反归一化
        print("== Part C 引擎语义分发实证（单实例 stats 翻转）==")
        ms_stats = {"norm_mode": "mean_std",
                    "actions": {"mean": ACT["mean"], "std": ACT["std"]},
                    "state": ns.get("state", {"mean": ACT["mean"]})}
        m._pipe.norm_stats = ms_stats
        r2 = m._pipe.infer(obs, noise=noise)
        out_B = np.asarray(r2["actions"] if isinstance(r2, dict) else r2)
        print(f"  [mean_std] 输出 {out_B.shape}，范围 [{out_B.min():.3f}, {out_B.max():.3f}]")

        q01 = np.array(ACT["q01"], np.float32)
        q99 = np.array(ACT["q99"], np.float32)
        mean = np.array(ACT["mean"], np.float32)
        std = np.array(ACT["std"], np.float32)

        def _invert(out, blk, mode):
            """按输出实际所用的 stats 反解归一化 raw（不是想当然用数据集的）。"""
            if mode == "q01_q99":
                lo, hi = np.array(blk["q01"], np.float32), np.array(blk["q99"], np.float32)
                return ((out - lo) / (hi - lo + 1e-6) * 2 - 1).astype(np.float32)
            lo, hi = np.array(blk["mean"], np.float32), np.array(blk["std"], np.float32)
            return ((out - lo) / (hi + 1e-6)).astype(np.float32)

        raw_from_A = _invert(out_A, ns["actions"], mode0)   # ckpt 加载的那套
        raw_from_B = _invert(out_B, ms_stats["actions"], "mean_std")
        err_cross = float(np.abs(raw_from_A - raw_from_B).max())
        check(err_cross < 1e-4,
              f"两种语义反解 raw 一致：max err = {err_cross:.2e}（输出只差反归一化）")
        ref = unnormalize_actions(raw_from_A, ms_stats)
        err_aff = float(np.abs(ref - out_B).max())
        check(err_aff < 1e-4, f"mean_std 输出符合其仿射式：max err = {err_aff:.2e}")

        # Part D 落域（相对数据集 q01/q99；信息项）
        print("== Part D 落域（相对数据集 q01/q99）==")
        for tag, o in ((mode0, out_A), ("mean_std", out_B)):
            tol = 0.1 * (q99 - q01 + 1e-6)
            lo_v = float((o < q01 - tol).mean())
            hi_v = float((o > q99 + tol).mean())
            note = "" if tag == "mean_std" else "（smoke 用 libero 占位 stats，仅看形状）"
            print(f"  {tag:8s}: 越下界 {lo_v:.1%} / 越上界 {hi_v:.1%}{note}")

print()
if fails:
    print(f"✗ {fails} 项未过")
    sys.exit(1)
print("✓ 对齐校验全部通过")
