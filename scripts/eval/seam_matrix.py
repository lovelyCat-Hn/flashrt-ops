#!/usr/bin/env python
"""接缝矩阵：在 --dump-chunks 落盘的原始 chunk 流上离线评估执行侧衔接算子。

背景：块内行间抖动可被低通（EMA）滤净，但换块重规划失配（接缝）是参考信号的
不连续——低通只把阶跃抹成斜坡、幅度消不掉。本脚本在纯参考空间把各算子对
接缝的作用算成一张表，供排名后再上真臂回放（scripts/test/chunk_replay_test.py）确认。

算子（全部执行侧，不改模型输出）：
  接入偏移 offset   下块首派行号（0=帧0重锚/不跳帧；≈consume=跳帧安装墙钟对齐）
  blend M           前向搭接：首 M 行换成 last_tgt→rows[M-1] 线性坡（构造性 C0 连续）
  ema α             一阶低通跨边界（阶跃→宽度 ~1/α 拍的斜坡）
  合步 N            首组并 1 条（拍0 派 rows[N-1]），大导程交给臂指数追踪吞掉

用法:
  python3 seam_matrix.py --chunks-dir logs/chunks_traj_<ts>/
  python3 seam_matrix.py --chunks-dir … --consume 13 --offsets 0 5 12 20 --blend 4 8 --ema 0.3 --merge 4
输出: 各算子组合的接缝 p50/p90/max（mrad，仅臂维；爪维是 0-100 百分比不入统计）
"""
import argparse
import math
import pathlib

import numpy as np

ARM = slice(0, 14)          # 16 维 action 前 14 为臂（rad）；末 2 爪=百分比剔除
GRIP = slice(14, 16)


def q(vals, p):
    v = np.sort(np.asarray(vals, float).ravel())   # ravel：2D 进来也按全元素分位
    k = (len(v) - 1) * p
    f, c = int(math.floor(k)), min(int(math.floor(k)) + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def stat(mat):
    """mat: (n_blocks, n_joints) mrad → (p50, p90, max) 全臂合并口径"""
    return q(mat.ravel(), .5), q(mat.ravel(), .9), mat.max()


def load(chunks_dir):
    files = sorted(pathlib.Path(chunks_dir).glob("chunk_*.npz"))
    ch, base, bad = [], None, []
    for f in files:
        try:
            d = np.load(f)
            ch.append(d["chunk"].astype(float))
        except Exception:                     # q 中断写坏的尾块，跳过
            bad.append(f.name)
            continue
        if base is None:
            base = d["base_arm"].astype(float)   # 首块的绝对基准（回放从臂位起步用）
    if bad:
        print(f"⚠ 跳过坏文件 {len(bad)} 个（q 中断截断）: {', '.join(bad[:3])}…")
    if len(ch) < 2:
        raise SystemExit(f"至少要 2 个完好 chunk，{chunks_dir} 只有 {len(ch)}")
    return ch, base


def seam_pairs(ch, last_idx, first_idx):
    """换块接缝：|c[k+1][first_idx] − c[k][last_idx]|，(n_pairs, 14) mrad"""
    out = []
    for a, b in zip(ch[:-1], ch[1:]):
        out.append((b[first_idx] - a[last_idx])[ARM])
    return np.abs(np.array(out)) * 1000.0


def row_deltas(ch):
    """块内行间增量（口径基线）：(n_rows_total, 14) mrad"""
    out = []
    for c in ch:
        out.append(np.abs(np.diff(c[:, ARM], axis=0)))
    return np.concatenate(out) * 1000.0


def blend_ramp_cost(c_prev_last, c_next, M):
    """搭接坡的代价：坡行最大步长（下发节律）+ 坡偏离模型原路的最大量（保真税）"""
    tgt = c_next[M - 1][ARM]
    ramp = np.linspace(c_prev_last[ARM], tgt, M)          # (M,14)
    step = np.abs(np.diff(np.vstack([c_prev_last[ARM][None], ramp]), axis=0))
    dev = np.abs(ramp - c_next[:M, ARM])
    return step.max() * 1000.0, dev.max() * 1000.0        # mrad


def ema_transition(ch, alpha, last_idx, first_idx, win=4):
    """跨边界 EMA 后的最大单行步长（transition 窗内），对比不加滤波的裸接缝"""
    out_raw, out_ema = [], []
    for a, b in zip(ch[:-1], ch[1:]):
        seg = np.vstack([a[last_idx - win + 1:last_idx + 1][:, ARM],
                         b[first_idx:first_idx + win][:, ARM]])   # (2win,14)，⚠2D 取列必须 [:,ARM]
        f = [seg[0]]
        for r in seg[1:]:
            f.append(alpha * r + (1 - alpha) * f[-1])
        d = np.abs(np.diff(np.array(f), axis=0))
        j = win - 1                                            # 边界所在行
        out_raw.append(d[j])                                   # 未滤：接缝步（d[j]=跨界阶跃）
        out_ema.append(d[j:j + 2].max(axis=0))                 # 滤后：过渡窗最大步
    return np.array(out_raw) * 1000.0, np.array(out_ema) * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--consume", type=int, default=13,
                    help="每块消费行数（水位 0 栈≈13 拍；决定 last_idx=consume-1）")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 5, 12, 20],
                    help="下块首派行号扫描（0=不跳帧；≈consume=跳帧安装）")
    ap.add_argument("--blend", type=int, nargs="*", default=[],
                    help="搭接坡长 M 扫描（在最优 offset 上算）")
    ap.add_argument("--ema", type=float, nargs="*", default=[],
                    help="EMA α 扫描（跨边界过渡谱）")
    ap.add_argument("--merge", type=int, nargs="*", default=[],
                    help="合步 N 扫描（拍0 派 rows[N-1] 的单发导程）")
    args = ap.parse_args()

    ch, base = load(args.chunks_dir)
    last_idx = args.consume - 1
    n_arm = 14
    print(f"chunks={len(ch)}  行形状={ch[0].shape}  每块消费 {args.consume} 行"
          f"（last_idx={last_idx}）  臂维统计，mrad\n")

    # ① 口径基线：块内正常行步长
    rd = row_deltas(ch)
    p, p90, mx = stat(rd)
    print(f"① 块内行间增量基线        p50={p:6.1f}  p90={p90:6.1f}  max={mx:7.1f}")

    # ② 接缝 × 接入偏移
    print("② 换块接缝 |c[k+1][off] − c[k][last]| ：")
    print("   offset  p50     p90     max")
    for off in args.offsets:
        p, p90, mx = stat(seam_pairs(ch, last_idx, min(off, 49)))
        tag = "（=跳帧安装）" if abs(off - args.consume) <= 1 else ("（=帧0重锚）" if off == 0 else "")
        print(f"   {off:>4}   {p:6.1f}  {p90:6.1f}  {mx:7.1f} {tag}")

    # ③ blend：在最优 offset（=consume）上算坡代价
    if args.blend:
        print(f"③ blend 坡代价（offset={args.consume}，坡长 M）：")
        print("   M   坡行最大步长   偏离模型原路 max")
        for M in args.blend:
            steps, devs = [], []
            for a, b in zip(ch[:-1], ch[1:]):
                s, d = blend_ramp_cost(a[last_idx], b, min(M, 49))
                steps.append(s)
                devs.append(d)
            print(f"   {M:>3}   {q(steps,.9):9.1f}      {q(devs,.9):9.1f}"
                  f"   （p90；坡行步长=下发节律税，偏离=保真税）")

    # ④ EMA：阶跃→斜坡
    if args.ema:
        print("④ EMA 跨边界过渡（裸接缝 vs 滤后过渡窗最大单行步长，offset=0）：")
        print("   α      裸接缝p90   滤后p90   滤后max")
        raw, _ = ema_transition(ch, 1.0, last_idx, 0)
        for a in args.ema:
            _, em = ema_transition(ch, a, last_idx, 0)
            print(f"   {a:<5}  {q(raw,.9):9.1f}  {q(em,.9):8.1f}  {em.max():8.1f}")

    # ⑤ 合步 N：拍0 单发导程（臂指数追踪 v≈2.6×gap 要吞的量）
    if args.merge:
        print("⑤ 合步 N 首发导程 |c[k+1][N-1] − c[k][last]|（越大越靠追踪吞）：")
        print("   N    p50     p90     max")
        for N in args.merge:
            p, p90, mx = stat(seam_pairs(ch, last_idx, min(N - 1, 49)))
            print(f"   {N:>3}  {p:6.1f}  {p90:6.1f}  {mx:7.1f}")

    print(f"\n注：块数={len(ch) - 1} 对；{args.chunks_dir}")


if __name__ == "__main__":
    main()
