#!/usr/bin/env python
"""往复摆动归因探针——带符号波形记录 + 四嫌疑分层判读（2026-10-10 排查设计）

背景：traj 线真机观察往复摆动，调度覆盖已实证无责（10-10 三跑：丢拍 0、
tick p50 1.2ms、星饿 0）。剩余嫌疑——
  ① 模型输出本身含往复（chunk 行增量方向翻转）
  ② 构建层放大（链式限幅/EMA/样条把小抖变往复）
  ③ 伺服/下发侧（指令单调而臂往复：tfs=0 fastest-arrival、240Hz 流式、
     刚性位置伺服 hunting、恒令关节被耦合带动）
  ④ 换块接缝激励（弃尾重锚每 ~1.2s 注入一次反向阶跃，激发衰减振荡）

与既有工具的分工：chunk_increments.py 只看模型离线谱；replay_stream_bench.py
真机隔离但遥测只有幅值（|Δcmd|/导程/残差）——往复=符号问题，幅值遥测
看不见翻转。本台补上：全程记录带符号指令帧序列 + 关节回读序列（同一
perf_counter 时轴），结束后自动跑归因分析（--analyze 可离线重放）。

判读逻辑（分析器输出 → 嫌疑）：
  A. 指令翻转率分层：chunk 行（=①）→ 链式行（①+②）→ 帧（②）；
     帧翻转 ≈ 行翻转×常数 → 构建层无放大；行翻转高 → 回看模型层
  B. 回读侧：残差(fb−cmd) 主导频率与幅值、回读速度方向翻转率
  C. 恒令关节（synthetic 模式未驱动关节指令全程常数）：回读仍摆 → ③实锤
  D. 相位/过零：cmd 速度 vs fb 速度互相关滞后（伺服带宽画像）、
     残差高频过零（chatter 签名）
  E. 换块锁定：换块后 0.5s 窗残差 RMS vs 块内基线比值 → ④

数据源：
  --source chunks     回放 run_g1_loop_traj --dump-chunks 落盘（replay 台同款，
                      含模型层真数据；--rows-per-block 35 复现水位换块节奏）
  --source synthetic  参数化干净正弦摆动（数学上无往复歧义）：hold→N 个整周期
                      正弦→hold；未驱动关节指令全程恒定——①②被构造性排除，
                      纯测 ③。恒令关节摆没摆是伺服侧最强判据。

安全（replay 台同款）：包络钳位（帧级）、漂移护栏、q/Ctrl-C 随退、
起始慢速就位 0.2 rad/s、干跑默认零下发。

用法:
  # 干跑（零下发，打印计划即退）
  ~/holy/run.sh ~/holy/scripts/probes/swing_attrib_probe.py \
      --source chunks --chunks-dir logs/chunks_traj_20261010_110619
  # 真机回放（①②④判读主实验；换块节奏 35 行 + hold 330ms 复现闭环）
  ~/holy/run.sh ~/holy/scripts/probes/swing_attrib_probe.py \
      --source chunks --chunks-dir logs/chunks_traj_XXXX --exec --rows-per-block 35
  # 伺服纯激励（③判读对照实验：指令数学干净）
  ~/holy/run.sh ~/holy/scripts/probes/swing_attrib_probe.py \
      --source synthetic --exec --wave-freq 1.0 --wave-amp 0.05 --joints 2,9
  # 离线重放分析（不碰 SDK，无需 run.sh）
  python3 ~/holy/scripts/probes/swing_attrib_probe.py \
      --analyze logs/swing_XXXX/waveform.npz
"""
import argparse
import atexit
import glob
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "inference"))

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT          # 数据集维序：右臂在前（0-6 右 / 7-13 左）
GAP_EPS = 1e-4

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--source", choices=["chunks", "synthetic"], default="chunks",
                help="chunks=回放 dump（①②④全链）；synthetic=正弦纯激励"
                     "（构造性排除①②，纯测③）")
ap.add_argument("--chunks-dir", default=None, help="chunks_traj_<ts> 落盘目录")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真机下发（不带=干跑：打印计划即退）")
ap.add_argument("--rounds", type=int, default=0, help="回放块数上限（0=全部）")
ap.add_argument("--fps", type=int, default=30, help="行节拍 Hz")
ap.add_argument("--frames-per-row", type=int, default=8,
                help="每行插值帧数（stream 突发下发）")
ap.add_argument("--interp", choices=("linear", "cubic", "quintic"), default="cubic")
ap.add_argument("--delta-max", type=float, default=0.05, help="链式限幅 rad/行")
ap.add_argument("--ema-alpha", type=float, default=0.0, help="行目标 EMA（0=关）")
ap.add_argument("--tfs-ms", type=float, default=0.0,
                help="time_from_start_s 毫秒：0=fastest arrival（现行为）")
ap.add_argument("--burst-reserve-ms", type=float, default=6.0,
                help="突发预算=tick−该值")
ap.add_argument("--rows-per-block", type=int, default=0,
                help="每块消费行数（0=全部 50；35=复现闭环水位换块节奏）")
ap.add_argument("--swap-gap-ms", type=float, default=330.0,
                help="块间 hold 毫秒（模拟推理落地窗；hold 期照常回读→换块后"
                     "振荡衰减过程完整入波形）")
# synthetic 波形参数
ap.add_argument("--wave-freq", type=float, default=1.0, help="正弦频率 Hz")
ap.add_argument("--wave-amp", type=float, default=0.05, help="正弦幅值 rad")
ap.add_argument("--wave-cycles", type=float, default=4.0, help="整周期数")
ap.add_argument("--wave-hold", type=float, default=1.0,
                help="首尾恒令段秒数（恒令关节/恒令期的伺服静度基线）")
ap.add_argument("--joints", type=str, default="2,9",
                help="驱动的臂关节下标（ARM_NAMES 序 0-13，逗号分隔；"
                     "默认 2,9=right_arm_joint3,left_arm_joint3；其余恒令）")
# 分析参数
ap.add_argument("--cmd-deadband", type=float, default=2e-4,
                help="指令翻转判死区 rad（默认 0.2 mrad，滤数值噪声假翻转）")
ap.add_argument("--fb-deadband", type=float, default=0.02,
                help="回读速度翻转判死区 rad/s（默认 0.02；汇总里会打噪声底"
                     "供校准）")
ap.add_argument("--analyze", default=None,
                help="离线模式：只分析指定 waveform.npz 后退出（不碰 SDK）")
# 安全
ap.add_argument("--envelope", type=str,
                default=str(_ROOT / "models" / "pi05_g1_onlypick_deploy"
                            / "joint_envelope.json"),
                help="包络护栏 json（臂 14 维钳位）")
ap.add_argument("--max-excursion", type=float, default=3.0,
                help="偏离起始位护栏 rad（任一关节超限即停）")
ap.add_argument("--goto-start", dest="goto_start", action="store_true", default=True)
ap.add_argument("--no-goto-start", dest="goto_start", action="store_false",
                help="跳过起始慢速就位（臂已在轨迹起点附近时）")
ap.add_argument("--nav-suspend", action="store_true",
                help="循环期 SIGSTOP 冻结导航栈七进程（同 run_g1_loop_traj；默认关）")
args = ap.parse_args()

TICK = 1.0 / args.fps
PER = args.frames_per_row


# ══════════════════════════════════════════════════════════════════════════
# 归因分析器（离线可重放；exec 结束后对刚录的波形自动跑一遍）
# ══════════════════════════════════════════════════════════════════════════
def layer_flips(series, db, dt):
    """(T,14) 带符号序列 → 每关节方向翻转统计。

    翻转定义：相邻有效增量（|Δ|>db）符号相反。翻转密度=次/s（时间口径，
    30Hz 行与 240Hz 帧直接可比）。
    返回 (每关节翻转/s, 有效增量数/s, 加权幅度均值 mrad)。
    """
    x = np.diff(series, axis=0)                      # (T-1,14)
    dur = max((len(series) - 1) * dt, 1e-9)
    out_flip = np.zeros(x.shape[1])
    out_act = np.zeros(x.shape[1])
    out_amp = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        col = x[:, j]
        act = col[np.abs(col) > db]
        if len(act) < 2:
            continue
        out_flip[j] = int(np.sum(act[1:] * act[:-1] < 0)) / dur
        out_act[j] = len(act) / dur
        out_amp[j] = float(np.mean(np.abs(act))) * 1000
    return out_flip, out_act, out_amp


def print_flip_table(title, series, db, dt, hint):
    fl, act, amp = layer_flips(series, db, dt)
    order = np.argsort(-fl)[:6]
    print(f"\n[{title}]（死区 {db*1000:.2f} mrad；{hint}）")
    top = ", ".join(f"j{i}({ARM_NAMES[i].replace('_arm_joint', '')}):"
                    f"{fl[i]:.1f}/s" for i in order if fl[i] > 0)
    pos = amp[amp > 0]
    print(f"  翻转密度合计 {float(fl.sum()):.1f} /s | 有效增量 "
          f"{float(act.sum()):.0f} /s | 加权幅度 p50 "
          f"{float(np.median(pos)) if pos.size else 0.0:.2f} mrad"
          + (f" | Top: {top}" if top else " | 无翻转"))


def analyze_wave(npz_path):
    """四嫌疑判读主入口。输入 exec 模式落盘的 waveform.npz。"""
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    cmd_t, cmd = d["cmd_t"], d["cmd"]          # 指令帧序列（下发时刻+14 维）
    fb_t, fb = d["fb_t"], d["fb"]              # 回读序列（拍首/拍尾+hold 期）
    rows_all, frames_all = d["rows_all"], d["frames_all"]
    swap_arr = np.atleast_1d(d["swap_t"])
    driven = meta["driven_joints"]
    src = meta["source"]

    print(f"\n══ 归因分析：{npz_path} ══")
    print(f"来源 {src} | 指令帧 {len(cmd_t)} | 回读 {len(fb_t)} "
          f"({fb_t[-1] - fb_t[0]:.1f}s) | 块起点 {len(swap_arr)} 个 | "
          f"驱动关节 {driven}")

    # ── A. 指令侧分层翻转（① vs ②）──
    print_flip_table("A1. 模型层：chunk 绝对目标行增量（30Hz）",
                     d["chunk_abs_all"], args.cmd_deadband, TICK,
                     "模型输出直通")
    print_flip_table("A2. 构建层：链式限幅后行目标（30Hz）",
                     rows_all, args.cmd_deadband, TICK,
                     f"链式±{args.delta_max}+EMA α={args.ema_alpha:g}")
    print_flip_table(f"A3. 流式层：下发帧增量（{args.fps * PER:.0f}Hz）",
                     frames_all, args.cmd_deadband / PER, TICK / PER,
                     "样条后帧序列；死区按帧倍频折算")
    if src == "synthetic":
        print("  ⚠ synthetic 源：A1-A3 按构造无往复（整周期正弦），仅校准"
              "记录链完整性；判读看 B/C/D")

    # ── B. 回读侧：速度翻转 + 残差谱 ──
    if len(fb_t) < 20:
        print("\n[B] 回读样本过少，跳过")
        return
    resid = np.empty_like(fb)
    for j in range(14):
        resid[:, j] = fb[:, j] - np.interp(fb_t, cmd_t, cmd[:, j])
    dt_fb = np.diff(fb_t)
    v_fb = np.diff(fb, axis=0) / dt_fb[:, None]          # 回读速度 (K-1,14)
    noise = float(np.median(np.abs(v_fb)))
    db_note = "OK"
    if args.fb_deadband <= 3 * noise:
        db_note = f"⚠ 死区偏小，建议 ≥ {3 * noise:.3f}"
    print(f"\n[B1. 回读速度噪声底] p50 |v| = {noise*1000:.1f} mrad/s"
          f"（--fb-deadband {args.fb_deadband} {db_note}）")
    dur_s = max(fb_t[-1] - fb_t[0], 1e-9)
    vfl = np.zeros(14)
    for j in range(14):
        col = v_fb[:, j]
        act = col[np.abs(col) > args.fb_deadband]
        if len(act) > 1:
            vfl[j] = int(np.sum(act[1:] * act[:-1] < 0)) / dur_s
    order = np.argsort(-vfl)[:6]
    top = ", ".join(f"j{i}: {vfl[i]:.2f}/s" for i in order if vfl[i] > 0)
    print(f"[B2. 回读速度方向翻转] 合计 {float(vfl.sum()):.1f} /s"
          + (f" | Top: {top}" if top else ""))

    # 残差主导频率（逐关节 FFT，去均值+hann；非均匀采样先重采样到中位 dt）
    dt_u = float(np.median(dt_fb))
    grid = np.arange(fb_t[0], fb_t[-1], dt_u)
    print(f"\n[B3. 残差(fb−cmd) 主导频率]（重采样 {1/dt_u:.0f}Hz · "
          f"Nyquist {0.5/dt_u:.1f}Hz；Top2，幅值 <0.1 mrad 不列）")
    for j in range(14):
        r = np.interp(grid, fb_t, resid[:, j])
        r = (r - r.mean()) * np.hanning(len(r))
        R = np.abs(np.fft.rfft(r))
        fq = np.fft.rfftfreq(len(r), dt_u)
        mask = fq > 0.15                                  # <0.15Hz 视为趋势
        if not mask.any() or len(R) < 8:
            continue
        R = np.where(mask, R, 0.0)
        pk = np.argsort(-R)[:2]
        amp = 2 * R[pk] / len(r) * 2                      # hann 窗增益补偿
        if amp[0] < 1e-4:
            continue
        line = (f"  j{j:>2} {ARM_NAMES[j]:<18} {fq[pk[0]]:5.2f} Hz "
                f"({amp[0]*1000:6.1f} mrad)")
        if amp[1] > 0.3 * amp[0]:
            line += f" | {fq[pk[1]]:5.2f} Hz ({amp[1]*1000:.1f})"
        print(line)

    # ── C. 恒令关节静度（③ 最强判据；synthetic 有恒令关节，chunks 无）──
    if src == "synthetic":
        hold = [j for j in range(14) if j not in driven]
        cmd_std = max(float(np.std(cmd[:, j])) for j in hold) * 1000
        print(f"\n[C. 恒令关节静度]（指令全程常数，指令 std max "
              f"{cmd_std:.2f} mrad）")
        for j in hold:
            pp = float(fb[:, j].max() - fb[:, j].min()) * 1000
            tag = "⚠ 指令恒定仍在摆 → 伺服/耦合侧" if pp > 5.0 else "静"
            print(f"  j{j:>2} {ARM_NAMES[j]:<18} 回读峰峰 {pp:6.1f} mrad "
                  f"{tag}")
        for j in driven:
            pp = float(fb[:, j].max() - fb[:, j].min()) * 1000
            a_cmd = args.wave_amp * 2000
            print(f"  j{j:>2} {ARM_NAMES[j]:<18} 回读峰峰 {pp:6.1f} mrad "
                  f"（指令幅值 {a_cmd:.0f} mrad；比值 {pp/max(a_cmd, 1e-9):.2f}）")

    # ── D. 相位滞后 + 残差过零 chatter ──
    max_lag = max(int(0.3 / dt_u), 1)
    v_cmd_u = np.zeros((len(grid), 14))
    v_fb_u = np.zeros((len(grid), 14))
    for j in range(14):
        v_cmd_u[:, j] = np.gradient(np.interp(grid, cmd_t, cmd[:, j]), dt_u)
        v_fb_u[:, j] = np.interp(grid, fb_t[1:], v_fb[:, j])
    dm = driven if src == "synthetic" else list(range(14))
    lags, corrs = [], []
    for j in dm:
        vc = v_cmd_u[:, j]
        vf = v_fb_u[:, j]
        if float(np.std(vc)) < 1e-6:
            continue
        vc = vc - vc.mean()
        vf = vf - vf.mean()
        den = float(np.std(vc) * np.std(vf) * len(vc)) + 1e-12
        best_c, best_l = -2.0, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                c = float(np.dot(vc[lag:], vf[:len(vf) - lag])) / den
            else:
                c = float(np.dot(vc[:lag], vf[-lag:])) / den
            if c > best_c:
                best_c, best_l = c, lag
        lags.append(-best_l * dt_u * 1000)   # 换算：正=fb 相对 cmd 滞后
        corrs.append(best_c)
    if lags:
        lags_a = np.asarray(lags)
        corrs_a = np.asarray(corrs)
        jb = int(np.argmax(corrs_a))
        good = corrs_a > 0.5
        med = float(np.median(lags_a[good])) if good.any() else float("nan")
        print(f"\n[D1. 速度互相关] 最强 j{dm[jb] if isinstance(dm, list) else jb}"
              f" corr {corrs_a[jb]:.2f} @ 滞后 {lags_a[jb]:+.0f} ms | "
              f"相关>0.5 关节中位滞后 {med:+.0f} ms"
              "（正值=回读落后于指令，伺服带宽画像）")
    thr = 3 * noise
    zc = 0
    for j in dm:
        col = resid[:, j]
        act = col[np.abs(col) > thr]
        zc += int(np.sum(act[1:] * act[:-1] < 0))
    print(f"[D2. 残差过零（|r|>{thr*1000:.1f} mrad）] {zc} 次 / "
          f"{dur_s:.1f}s = {zc/dur_s:.2f} /s（高频过零=chatter 签名，伺服侧）")

    # ── E. 换块锁定（④；块 0 起步不是接缝，post 只统计块 2 起）──
    if src == "chunks" and len(swap_arr) >= 2:
        half = 0.5
        post = []
        for st in swap_arr[1:]:
            m = (fb_t >= st) & (fb_t < st + half)
            if m.sum() > 5:
                post.append(float(np.sqrt(np.mean(resid[m] ** 2))))
        base = []
        for b in range(len(swap_arr)):
            t_b = fb_t[(fb_t >= swap_arr[b])
                       & (fb_t < (swap_arr[b + 1] if b + 1 < len(swap_arr)
                                  else fb_t[-1] + 1))]
            if len(t_b) < 10:
                continue
            lo = t_b[0] + (half if b == 0 else 0.1)
            m = (fb_t >= lo) & (fb_t <= t_b[-1] - 0.1)
            if m.sum() > 10:
                base.append(float(np.sqrt(np.mean(resid[m] ** 2))))
        if post and base:
            r = float(np.median(post) / max(np.median(base), 1e-9))
            print(f"\n[E. 换块锁定] 换块后 {half}s 窗残差 RMS / 块内基线 = "
                  f"{r:.2f}×（{len(post)} 次换块；>1.5× = 接缝激励显著）")
    print("\n══ 判读指引 ══")
    print("  A2/A3 翻转高 → 指令本身往复：对照 A1 定位模型 vs 构建（配合")
    print("    chunk_increments.py 的模型翻转率签名）→ 嫌疑①②")
    print("  指令干净 + B2/B3 回读往复 + C 恒令关节摆 → 伺服/下发侧 → 嫌疑③")
    print("    （配合 replay_stream_bench --tfs-ms 0/4.2/33.3 A/B 定位 tfs 语义）")
    print("  E 比值高 → 每次换块激励一次衰减振荡 → 嫌疑④（接缝）")


# ══════════════════════════════════════════════════════════════════════════
# 波形装载/生成
# ══════════════════════════════════════════════════════════════════════════
def make_sine_block(center):
    """hold → N 个整周期正弦（仅驱动关节）→ hold；其余维恒=center。"""
    n_hold = max(int(round(args.wave_hold * args.fps)), 1)
    n_sine = int(round(args.wave_cycles / args.wave_freq * args.fps))
    t_sine = np.arange(n_sine, dtype=np.float64) / args.fps
    s = args.wave_amp * np.sin(2 * np.pi * args.wave_freq * t_sine)
    blk = np.repeat(np.asarray(center, dtype=np.float32)[None, :],
                    n_hold + n_sine + n_hold, axis=0)
    blk[n_hold:n_hold + n_sine][:, DRIVEN] += s[:, None].astype(np.float32)
    return blk, n_hold, n_sine


ENV_LO14 = ENV_HI14 = None
DRIVEN = sorted(int(x) for x in args.joints.split(",") if x != "")
if not DRIVEN or any(not 0 <= j < 14 for j in DRIVEN):
    raise SystemExit(f"--joints 须为 0-13 的下标：{args.joints}")

if args.analyze is None:
    from g1_traj_interp import create_interpolator  # noqa: E402

    if os.path.exists(args.envelope):
        _env = json.loads(pathlib.Path(args.envelope).read_text())
        _inf = float("inf")
        lo = [(-_inf if v is None else float(v)) for v in _env["action_lo"]]
        hi = [(_inf if v is None else float(v)) for v in _env["action_hi"]]
        ENV_LO14 = np.array(lo[:7] + lo[8:15], dtype=np.float32)
        ENV_HI14 = np.array(hi[:7] + hi[8:15], dtype=np.float32)
        print(f"[护栏] 包络开：{pathlib.Path(args.envelope).name}（臂 14 维钳位）")
    else:
        print(f"[护栏] ⚠ 包络文件缺失，跳过钳位：{args.envelope}")

    if args.wave_amp * 2 * np.pi * args.wave_freq > 0.6:
        print(f"⚠ synthetic 峰值速度 {args.wave_amp * 2 * np.pi * args.wave_freq:.2f}"
              " rad/s 偏高，建议 --wave-amp×--wave-freq 乘积 ≤ 0.1")

    CHUNK_ABS = []       # 模型层绝对目标（chunks 源）
    BLOCKS = []          # 行目标块（链式限幅输入）
    if args.source == "chunks":
        if not args.chunks_dir:
            raise SystemExit("--source chunks 需要 --chunks-dir")
        files = sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.npz")))
        if not files:
            raise SystemExit(f"目录无 chunk_*.npz：{args.chunks_dir}")
        for f in files:
            dd = np.load(f)
            ch = np.asarray(dd["chunk"], dtype=np.float32)
            base = np.asarray(dd["base_arm"], dtype=np.float32)
            abs_blk = (np.concatenate([ch[:, :7], ch[:, 8:15]], axis=1)
                       + base[None, :])
            CHUNK_ABS.append(abs_blk)
            BLOCKS.append(abs_blk)
        n_blocks = len(BLOCKS) if args.rounds <= 0 else min(args.rounds,
                                                            len(BLOCKS))
        BLOCKS, CHUNK_ABS = BLOCKS[:n_blocks], CHUNK_ABS[:n_blocks]
    else:
        # 干跑占位：中心=0（真机路径按当前臂位重建）；行数/节奏与真机一致
        blk, _, _ = make_sine_block(np.zeros(14, dtype=np.float32))
        BLOCKS, CHUNK_ABS = [blk], [blk.copy()]
        n_blocks = 1
    INTERP = create_interpolator(args.interp, dim=14, input_hz=args.fps,
                                 output_hz=args.fps * PER)


def build_rows(cur, tgt_blk):
    """链式限幅 + 可选 EMA（主脚本 build_rows 同构）。"""
    rows = np.empty_like(tgt_blk)
    prev = np.asarray(cur, dtype=np.float32)
    for kk in range(len(tgt_blk)):
        prev = prev + np.clip(tgt_blk[kk] - prev,
                              -args.delta_max, args.delta_max)
        rows[kk] = prev
    if args.ema_alpha > 0:
        a = float(args.ema_alpha)
        acc = rows[0]
        for kk in range(1, len(rows)):
            acc = a * rows[kk] + (1.0 - a) * acc
            rows[kk] = acc
    return rows


# ── 日志（swing_<ts>.log，主脚本同款 Tee；--analyze 离线模式不建文件）──
_log_path = str(_ROOT / "logs" / time.strftime("swing_%Y%m%d_%H%M%S.log"))
_log_fh = (open(_log_path, "a", buffering=1, encoding="utf-8")
           if args.analyze is None else None)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except ValueError:
                pass    # teardown 已关日志文件后的残余 flush

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except ValueError:
                pass


# ── q 退出（replay 台同款）──
class QuitWatcher:
    """后台线程监听键盘 q：任何阶段即时退出（SDK 阻塞 C++ 调用吞 Ctrl-C）。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._old = None
        self._wake_r = None
        self.active = threading.Event()
        if os.isatty(self.fd):
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.restore)
            r, w = os.pipe()
            os.set_blocking(w, False)
            signal.set_wakeup_fd(w)
            signal.signal(signal.SIGINT, lambda *_: None)
            signal.signal(signal.SIGTERM, lambda *_: None)
            self._wake_r = r
            self.active.set()
            threading.Thread(target=self._loop, daemon=True).start()
            print("（循环期间随时按 q 或 Ctrl-C 退出；确认提示符处用回车；"
                  "急停第一优先级）")

    def restore(self):
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def pause(self):
        self.active.clear()

    def resume(self):
        self.active.set()

    def _loop(self):
        while True:
            try:
                fds = [self._wake_r] if self._wake_r is not None else []
                if self.active.is_set():
                    fds.append(sys.stdin)
                if not fds or not select.select(fds, [], [], 0.2)[0]:
                    continue
                if (self._wake_r is not None
                        and select.select([self._wake_r], [], [], 0)[0]):
                    os.read(self._wake_r, 1)
                    print("\n⛔ 收到中断信号 —— 立即退出"
                          "（已下发目标可能仍在限速执行）")
                    self.restore()
                    _emergency_teardown()
                    os._exit(130)
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出"
                          "（已下发目标可能仍在限速执行）")
                    self.restore()
                    _emergency_teardown()
                    os._exit(2)
            except Exception:
                self.restore()
                _emergency_teardown()
                os._exit(3)


robot = None
WATCH = None


# ── 导航栈 SIGSTOP 搁置（--nav-suspend，run_g1_loop_traj 同款移植）──
# A/B 对比要求与闭环跑同背景条件：导航栈停/不停会改变 tick 服务负载。
NAV_SUSPEND_NAMES = (
    "localization_server", "galbot_fusion_main", "service_navigation_plan",
    "galbot_vtn", "surround_cameras_capture", "swallows", "service_lidar_capture",
)
_NAV_GUARD_SRC = """
import os, sys, time, signal
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
ppid = os.getppid()
pids = [int(x) for x in sys.argv[1].split(',') if x]
while True:                      # 父进程（本脚本）消失才往下走
    time.sleep(0.3)
    try:
        os.kill(ppid, 0)
    except OSError:
        break
for p in pids:
    try:
        os.kill(p, signal.SIGCONT)   # 最后一道网：主进程任何死法都恢复导航栈
    except OSError:
        pass
"""


class NavSuspend:
    """只认领自己 STOP 的 pid（resume 只 CONT 这些），对别人冻结的不碰。"""

    def __init__(self):
        self.stopped = {}                     # pid -> name
        self._guard = None

    def _find_pids(self):
        found = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    arg0 = f.read().split(b"\x00")[0].decode("utf-8", "replace")
            except OSError:
                continue
            base = os.path.basename(arg0)
            if base in NAV_SUSPEND_NAMES:
                found[int(pid)] = base
        return found

    def suspend(self):
        cands = self._find_pids()
        missing = [n for n in NAV_SUSPEND_NAMES if n not in cands.values()]
        if missing:
            print(f"[nav-suspend] ⚠ 未找到（跳过）: {missing}")
        self._guard = subprocess.Popen(
            [sys.executable, "-c", _NAV_GUARD_SRC,
             ",".join(str(p) for p in sorted(cands))],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for pid, name in sorted(cands.items()):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    if f.read().split()[2] == "T":   # 已冻结（非我们所停）：不碰
                        continue
            except (OSError, IndexError):
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
                self.stopped[pid] = name
            except OSError as e:
                print(f"[nav-suspend] ⚠ STOP {name}({pid}) 失败: {e}")
        print(f"[nav-suspend] 已冻结 {len(self.stopped)} 进程: "
              + ", ".join(f"{n}({p})" for p, n in sorted(self.stopped.items()))
              + f" | 守护网 pid={self._guard.pid}")
        print("[nav-suspend] ⚠ 冻结期底盘勿动（localization 位姿停更）；"
              "相机 capture 不在冻结集，取图不受影响")

    def resume(self):
        if not self.stopped:
            return
        for pid in sorted(self.stopped):
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        print(f"[nav-suspend] 已恢复 {len(self.stopped)} 进程"
              "（CONT 幂等；守护网发现主进程退出后自退）")
        self.stopped.clear()


def _teardown():
    # 顺序：导航栈恢复 + 终端还原先于 SDK 关闭——SDK 关闭可能挂死被看门狗
    # 强杀，先做不可等的事（强杀后导航栈冻结/终端无回显才是真事故）
    _nav = globals().get("_NAV")
    if _nav is not None:
        try:
            _nav.resume()
        except Exception:
            pass
    if WATCH is not None:
        WATCH.restore()
    rob = globals().get("robot")
    if rob is not None:
        try:
            rob.request_shutdown()
            rob.wait_for_shutdown()
            rob.destroy()
        except Exception:
            pass
    if _log_fh is not None:
        try:
            _log_fh.close()
        except Exception:
            pass


def _fatal_hook(t, v, tb):
    print(f"\n⛔ 异常退出: {t.__name__}: {v}")
    _teardown()
    os._exit(1)


# ── 波形冲账：正常结束与 q/信号急退共用（急退不丢已录数据）──
_WAVE = None


def _flush_wave():
    wv = globals().get("_WAVE")
    if not wv or not wv["cmd_t_l"]:
        return
    try:
        wave_dir = pathlib.Path(_log_path).with_suffix("")
        wave_dir.mkdir(parents=True, exist_ok=True)
        npz_path = str(wave_dir / "waveform.npz")
        meta = {"source": args.source, "driven_joints": DRIVEN,
                "fps": args.fps, "frames_per_row": PER, "interp": args.interp,
                "delta_max": args.delta_max, "ema_alpha": args.ema_alpha,
                "tfs_ms": args.tfs_ms,
                "burst_reserve_ms": args.burst_reserve_ms,
                "rows_per_block": args.rows_per_block,
                "swap_gap_ms": args.swap_gap_ms,
                "wave": {"freq": args.wave_freq, "amp": args.wave_amp,
                         "cycles": args.wave_cycles, "hold": args.wave_hold},
                "arm_names": ARM_NAMES,
                "start_pos": wv["start_pos"].tolist()}
        rows_all = (np.vstack(wv["rows_all_l"]) if wv["rows_all_l"]
                    else np.zeros((0, 14), np.float32))
        frames_all = (np.vstack(wv["frames_all_l"]) if wv["frames_all_l"]
                      else np.zeros((0, 14), np.float32))
        chunk_abs_all = (np.vstack(CHUNK_ABS) if args.source == "chunks"
                         else rows_all.copy())
        np.savez(npz_path,
                 cmd_t=np.asarray(wv["cmd_t_l"], dtype=np.float64),
                 cmd=np.asarray(wv["cmd_l"], dtype=np.float32),
                 cmd_row=np.asarray(wv["cmd_row_l"], dtype=np.int64),
                 fb_t=np.asarray(wv["fb_t_l"], dtype=np.float64),
                 fb=np.asarray(wv["fb_l"], dtype=np.float32),
                 rows_all=rows_all, frames_all=frames_all,
                 chunk_abs_all=chunk_abs_all,
                 swap_t=np.asarray(wv["swap_t_l"], dtype=np.float64),
                 meta=np.array(json.dumps(meta)))
        print(f"[波形] {npz_path}（指令帧 {len(wv['cmd_t_l'])} / "
              f"回读 {len(wv['fb_l'])}）")
        analyze_wave(npz_path)
    except Exception as e:
        print(f"⚠ 波形落盘/归因失败: {e}")


def _emergency_teardown():
    """q/信号急退清理：波形冲账 + 导航栈恢复。

    不做 SDK 优雅关闭——request_shutdown/wait_for_shutdown 从监听线程调
    会被 SDK 残留线程挂死（20261010_145554 pid 92729 实录，47 线程全
    futex，SIGTERM 都进不去，SIGKILL 才清掉）；traj 脚本同款教训：
    急退路径直接 os._exit。
    """
    _flush_wave()
    _nav = globals().get("_NAV")
    if _nav is not None:
        try:
            _nav.resume()
        except Exception:
            pass


def _arm_hard_exit(delay_s=20.0):
    """独立进程看门狗：delay_s 后 SIGKILL 本进程（正常退出则打空无害）。

    threading.Timer 不行——SDK 关闭挂死时残留线程握死 GIL，任何 Python
    线程（含定时器回调）都无法再执行（swing_20261010_150624 实录：15s
    Timer 4 分钟不响，主线程卡在 wait_for_shutdown 轮询 sleep）。子进程
    SIGKILL 不经过 GIL，必定生效。
    """
    subprocess.Popen(
        [sys.executable, "-c",
         "import os, sys, time\n"
         "time.sleep(float(sys.argv[2]))\n"
         "try: os.kill(int(sys.argv[1]), 9)\n"
         "except OSError: pass\n",
         str(os.getpid()), str(delay_s)],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════
def main():
    global robot, WATCH, BLOCKS, CHUNK_ABS, n_blocks
    from galbot_sdk.g1 import GalbotRobot, JointCommand  # noqa: E402

    print(f"[swing-probe] 源 {args.source} | tick {TICK*1000:.1f} ms"
          f"（{args.fps} Hz）| {PER} 帧/行（≈{args.fps*PER:.0f}Hz "
          f"{args.interp}）| 突发预算 {TICK*1000 - args.burst_reserve_ms:.1f}"
          f" ms/行 | 链式 ±{args.delta_max} | EMA α={args.ema_alpha:g}"
          f" | tfs {args.tfs_ms:g} ms | 驱动关节 {DRIVEN}")

    _wp = np.zeros((3, 14), dtype=np.float32)
    _t0 = time.perf_counter()
    INTERP.interpolate(_wp)
    print(f"[样条] 预热完成（首次构造 {1000*(time.perf_counter() - _t0):.0f} ms）")

    if not args.do_exec:
        cur = np.zeros(14, dtype=np.float32)
        for bi, tgt in enumerate(BLOCKS):
            rows = build_rows(cur, tgt)
            if args.rows_per_block > 0:
                rows = rows[:min(args.rows_per_block, len(rows))]
            d = np.abs(np.diff(np.vstack([cur[None], rows]), axis=0))
            fr = INTERP.interpolate(np.vstack([cur[None], rows]))
            print(f"[干跑] 块{bi+1}: 行{len(rows)} |Δcmd| mean "
                  f"{d[1:].mean()*1000:.1f} max {d[1:].max()*1000:.1f} mrad "
                  f"| 帧间最大步长 "
                  f"{float(np.max(np.abs(np.diff(fr, axis=0))))*1000:.3f} mrad"
                  f" | 帧 {len(fr)}")
            cur = rows[-1]
            if ENV_LO14 is not None:
                cur = np.clip(cur, ENV_LO14, ENV_HI14)
        if args.source == "synthetic":
            print("[干跑] synthetic 中心=0 占位；真机按当前臂位重建（包络"
                  "安全中心），行数/节奏如上。")
        print("[干跑] 零下发，计划如上。")
        return

    # ── 真机 ──
    robot = GalbotRobot()
    if not robot.init():
        raise SystemExit("robot.init 失败")
    time.sleep(2)
    _cst = robot.start_controller("all")
    print(f"start_controller('all') → {_cst}")
    if args.nav_suspend:
        globals()["_NAV"] = NavSuspend()
        globals()["_NAV"].suspend()

    def read_joints():
        v = robot.get_joint_positions([], ARM_NAMES)
        return np.asarray([float(x) for x in v], dtype=np.float32)

    start_pos = read_joints()
    print(f"起始位: {[round(float(v), 3) for v in start_pos]}")

    if args.source == "synthetic":
        center = start_pos.copy()
        for j in DRIVEN:
            c = float(start_pos[j])
            if ENV_LO14 is not None and np.isfinite(ENV_LO14[j]):
                c = max(c, float(ENV_LO14[j]) + args.wave_amp + 0.02)
            if ENV_HI14 is not None and np.isfinite(ENV_HI14[j]):
                c = min(c, float(ENV_HI14[j]) - args.wave_amp - 0.02)
            center[j] = c
        blk, n_hold, n_sine = make_sine_block(center)
        BLOCKS = [blk]
        CHUNK_ABS = [blk.copy()]
        n_blocks = 1
        print(f"[synthetic] 行 {len(blk)}（hold {n_hold} + 正弦 {n_sine} + "
              f"hold {n_hold}）| 驱动 {[ARM_NAMES[j] for j in DRIVEN]} "
              f"幅值 {args.wave_amp*1000:.0f} mrad @ {args.wave_freq} Hz | "
              f"峰值速度 {args.wave_amp*2*np.pi*args.wave_freq:.2f} rad/s | "
              f"其余关节恒令")

    # 起始就位：慢速到首块行0（synthetic 行0=起始位，天然免动）
    first_abs = BLOCKS[0][0]
    _dist = float(np.max(np.abs(first_abs - start_pos)))
    if args.goto_start and _dist > 0.02:
        print(f"[就位] 距首块行0 {_dist*1000:.0f} mrad → 0.2 rad/s 慢速移动…")
        robot.set_joint_positions([float(v) for v in first_abs],
                                  [], ARM_NAMES, True, 0.2, 30.0)
        time.sleep(0.5)
    elif _dist > 0.3:
        print(f"⚠ 未就位且 --no-goto-start：距首块行0 {_dist*1000:.0f} mrad"
              "（风险自担）")

    WATCH = QuitWatcher()
    WATCH.pause()
    input(f"\n⚠ 将真机回放 {n_blocks} 块（源 {args.source}，无推理；"
          "急停就绪后回车开始）...")
    WATCH.resume()

    # 波形记录缓冲
    cmd_t_l, cmd_l, cmd_row_l = [], [], []
    fb_t_l, fb_l = [], []
    swap_t_l = []

    def rec_fb(t, v):
        fb_t_l.append(t)
        fb_l.append(v.copy())

    def send_burst(frames_idx_list, frames_all, row_k):
        """帧突发：绝对截止配速（主脚本同款）+ 逐帧记录带符号指令。"""
        fdt = (TICK - args.burst_reserve_ms / 1000.0) / len(frames_idx_list)
        _t_f = time.perf_counter()
        for _bi, _fi in enumerate(frames_idx_list):
            _fr = frames_all[_fi]
            _cmds = []
            for _ji in range(14):
                _c = JointCommand()
                _c.position = float(_fr[_ji])
                _cmds.append(_c)
            cmd_t_l.append(time.perf_counter())
            cmd_l.append(_fr.copy())
            cmd_row_l.append(row_k)
            _st = robot.set_joint_commands(
                _cmds, joint_names=ARM_NAMES,
                time_from_start_s=args.tfs_ms / 1000.0)
            if not str(_st).startswith("ControlStatus.SUCCESS"):
                return _st
            _t_f += fdt
            if _bi < len(frames_idx_list) - 1:
                _sl = _t_f - time.perf_counter()
                if _sl > 0:
                    time.sleep(_sl)
        return None

    t0 = time.perf_counter()
    n_skip = 0
    n_env_clip = 0
    rows_all_l, frames_all_l = [], []
    globals()["_WAVE"] = dict(cmd_t_l=cmd_t_l, cmd_l=cmd_l,
                              cmd_row_l=cmd_row_l, fb_t_l=fb_t_l, fb_l=fb_l,
                              rows_all_l=rows_all_l,
                              frames_all_l=frames_all_l, swap_t_l=swap_t_l,
                              start_pos=start_pos)
    for bi in range(n_blocks):
        cur = read_joints()
        rows = build_rows(cur, BLOCKS[bi])
        n_rows = (len(rows) if args.rows_per_block <= 0
                  else min(args.rows_per_block, len(rows)))
        rows = rows[:n_rows]
        frames_all = INTERP.interpolate(
            np.vstack([cur[None].astype(np.float32), rows]))
        if ENV_LO14 is not None:
            _pre = frames_all.copy()
            np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :],
                    out=frames_all)
            n_env_clip += int(np.sum(np.any(_pre != frames_all, axis=1)))
        bursts = [list(range(kk * PER, (kk + 1) * PER)) for kk in range(n_rows)]
        rows_all_l.append(rows)
        frames_all_l.append(frames_all)
        swap_t_l.append(time.perf_counter())
        d_row = np.abs(np.diff(np.vstack([cur[None], rows]), axis=0))
        print(f"\n── 回放块 {bi+1}/{n_blocks}（行 {n_rows}）| "
              f"|Δcmd| mean {d_row[1:].mean()*1000:.1f} max "
              f"{d_row[1:].max()*1000:.1f} mrad ──")

        last_tgt = None
        _next_t = time.perf_counter()
        for kk in range(n_rows):
            _now = time.perf_counter()
            if _now < _next_t:
                time.sleep(_next_t - _now)
            _next_t += TICK
            _t_tick = time.perf_counter()
            cur = read_joints()
            rec_fb(_t_tick, cur)
            tgt = rows[kk]
            cmd_delta = ((tgt - last_tgt) if last_tgt is not None
                         else np.zeros_like(tgt))
            jmax = int(np.argmax(np.abs(cmd_delta)))
            sgn = "+" if cmd_delta[jmax] >= 0 else "−"
            d_mrad = float(np.abs(cmd_delta[jmax])) * 1000
            gap0 = float(np.max(np.abs(tgt - cur)))
            drift = float(np.max(np.abs(tgt - start_pos)))
            if drift > args.max_excursion:
                print(f"⛔ 漂移护栏：关节最大偏离 {drift*1000:.0f} mrad > "
                      f"{args.max_excursion*1000:.0f}，停止（机械臂留在原地）")
                _teardown()
                os._exit(5)
            if gap0 < GAP_EPS:
                n_skip += 1
                _fr_s = "· 免发 ·"
            else:
                _st = send_burst(bursts[kk], frames_all, kk)
                if _st is not None:
                    print(f"⛔ 下发非 SUCCESS（{_st}），停止")
                    _teardown()
                    os._exit(4)
                _fr_s = f"{len(bursts[kk])}帧@{args.fps*PER:.0f}Hz"
            last_tgt = tgt.copy()
            _t_ach = time.perf_counter()
            ach = read_joints()
            rec_fb(_t_ach, ach)
            err = float(np.max(np.abs(ach - tgt))) * 1000
            print(f"  拍{kk}: Δcmd[j{jmax}] {sgn}{d_mrad:5.1f} mrad"
                  f" | 导程 {gap0*1000:5.1f} → {_fr_s}"
                  f" | 残差 {err:4.1f} mrad")
        # 块间 hold：不下发，但保持 tick 节奏回读（换块后衰减过程入波形）
        if bi < n_blocks - 1 and args.swap_gap_ms > 0:
            _t_hold_end = time.perf_counter() + args.swap_gap_ms / 1000.0
            while time.perf_counter() < _t_hold_end:
                rec_fb(time.perf_counter(), read_joints())
                time.sleep(max(0.0, TICK - 0.002))

    rows_all = np.vstack(rows_all_l)
    frames_all = np.vstack(frames_all_l)
    print(f"\n== 回放完成 == 块 {n_blocks} | 免发 {n_skip} | 包络钳位帧 "
          f"{n_env_clip} | 总耗时 {time.perf_counter() - t0:.1f} s")

    # ── 波形落盘 + 自动归因分析（q/信号急退走同款 _flush_wave）──
    _flush_wave()
    _arm_hard_exit(20.0)   # 看门狗先行：SDK 关闭挂死则 20s 后强杀保终端
    _teardown()
    os._exit(0)   # 不走解释器关闭（SDK 残留线程段错误坑，traj 同款）


if args.analyze is not None:
    analyze_wave(args.analyze)
else:
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    sys.excepthook = _fatal_hook
    main()
