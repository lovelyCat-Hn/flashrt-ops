#!/usr/bin/env python
"""数据集观测回放：模型逐控制步吃数据集真实观测（图+state+原句），预测动作
与数据集真实动作逐帧对比（bf16 定档，纯离线不碰机器人）。

与 tf_matrix.py 的差别：tf_matrix 抽查孤立帧；本脚本把整条 episode 按
--stride（默认 3 帧=10 Hz 控制步）回放——每个控制步取该帧观测、出 chunk、
消费 stride 步（delta+预测时刻 state 臂位=绝对目标，与真机执行同语义），
覆盖整条轨迹。输出：
  - 逐帧对比数组 /tmp/tf_rollout_ep{N}.npz
  - 对比图 /tmp/tf_rollout_ep{N}.png（夹爪开合时机 + 代表关节 + 误差曲线，
    系统 python3 matplotlib 画图，标签用英文避免缺字体）
  - 终端摘要：臂误差 / delta cos / 夹爪开合帧对齐

用法:
  ~/holy/run.sh ~/holy/scripts/eval/tf_rollout.py [--ep 0] [--stride 3]
"""
import argparse
import ast
import functools
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--ep", type=int, default=0, help="数据集 episode 号（默认 0）")
ap.add_argument("--stride", type=int, default=3,
                help="控制步长帧（3=10Hz，与闭环 steps-per-cmd≈3 同节奏）")
ap.add_argument("--start", type=int, default=0, help="起始帧")
ap.add_argument("--end", type=int, default=-1, help="结束帧（-1=到末尾）")
ap.add_argument("--config", default=None, help="g1.toml 路径（默认内置查找）")
args = ap.parse_args()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "inference"))
import g1_config  # noqa: E402

_cfg = argparse.Namespace(ckpt=args.ckpt)
g1_config.apply(_cfg, {"ckpt": ("run", "ckpt")})
CKPT = pathlib.Path(_cfg.ckpt)

# bf16 定档（2026-09-23 tf_matrix 实证：INT8 毁动作质量）——开关必须在 import 前
os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
os.environ.setdefault("PI05_NO_GRAPH", "1")
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")

import cv2  # noqa: E402
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402

# ── 数据集（episode npz 缓存，与 tf_matrix 共用）──
DS = pathlib.Path("/home/galbot/holy/datasets/pick_place_balence")
CAM_KEY = {"image": "head_right", "wrist_image": "left_arm",
           "wrist_image_right": "right_arm"}
ARM_DIMS = list(range(0, 7)) + list(range(8, 15))   # 14 臂维（右 7 在前）
GRIP = [7, 15]

CACHE = pathlib.Path(f"/tmp/tf_matrix_ep{args.ep}.npz")
if not CACHE.exists():
    print(f"抽取 episode {args.ep} → {CACHE}（系统 python3，一次性）", flush=True)
    subprocess.run([
        "/usr/bin/python3", "-c",
        """
import pathlib, numpy as np, pandas as pd, sys
ds = pathlib.Path(sys.argv[1]); out = sys.argv[2]; ep = int(sys.argv[3])
data = pd.read_parquet(ds / "data/chunk-000/file-000.parquet")
meta = pd.read_parquet(ds / "meta/episodes/chunk-000/file-000.parquet")
row = meta[meta.episode_index == ep].iloc[0]
df = data[data.episode_index == ep].reset_index(drop=True)
np.savez(out,
         states=np.stack([np.asarray(s, np.float32) for s in df["observation.state"]]),
         actions=np.stack([np.asarray(a, np.float32) for a in df["action"]]),
         tasks=np.array([str(row["tasks"])]),
         t0_head=float(row["videos/observation.images.head_right/from_timestamp"]),
         t0_left=float(row["videos/observation.images.left_arm/from_timestamp"]),
         t0_right=float(row["videos/observation.images.right_arm/from_timestamp"]))
print(f"ep{ep}: {len(df)} 帧 → {out}")
""",
        str(DS), str(CACHE), str(args.ep)], check=True)

_z = np.load(CACHE)
states, actions = _z["states"], _z["actions"]
try:
    TASK = ast.literal_eval(str(_z["tasks"][0]))[0]
except Exception:
    TASK = str(_z["tasks"][0])
T0_TS = {"head_right": float(_z["t0_head"]), "left_arm": float(_z["t0_left"]),
         "right_arm": float(_z["t0_right"])}
end = len(states) - 1 - args.stride if args.end < 0 else min(args.end, len(states) - 1 - args.stride)
ctrl_ts = list(range(args.start, end, args.stride))
print(f"ep{args.ep} {len(states)} 帧 | 回放帧 {args.start}~{end} stride={args.stride} "
      f"→ {len(ctrl_ts)} 控制步 | 任务: {TASK}", flush=True)


def frame_at(cam, t):
    p = DS / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_MSEC, (T0_TS[cam] + t / 30.0) * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise SystemExit(f"取帧失败: ep{args.ep} t={t} {cam}")
    return np.ascontiguousarray(
        cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))


# ── 模型 ──
print(f"加载模型 {CKPT}（bf16）...", flush=True)
t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
ns = model._pipe.norm_stats
print(f"模型就绪 load {time.time() - t0:.1f}s", flush=True)

# ── 回放 ──
n_f = len(ctrl_ts) * args.stride
frames_log = np.empty(n_f, np.int32)          # 消费帧号
m_abs = np.empty((n_f, 14), np.float32)       # 模型绝对臂目标
d_abs = np.empty((n_f, 14), np.float32)       # 数据集绝对臂目标
m_g = np.empty((n_f, 2), np.float32)          # 模型夹爪 %
d_g = np.empty((n_f, 2), np.float32)          # 数据集夹爪 %
cos_l, ms_l = [], []
i = 0
t_all = time.perf_counter()
for t in ctrl_ts:
    imgs = {v: frame_at(c, t) for v, c in CAM_KEY.items()}
    st = states[t]
    t1 = time.perf_counter()
    chunk = np.asarray(model.predict(imgs, prompt=TASK,
                                     state=normalize_state(st, ns)))
    ms_l.append((time.perf_counter() - t1) * 1000)
    base = st[ARM_DIMS]
    d0 = actions[t][ARM_DIMS] - base
    cos_l.append(float(np.dot(chunk[0][ARM_DIMS], d0)
                       / max(float(np.linalg.norm(chunk[0][ARM_DIMS])
                                        * np.linalg.norm(d0)), 1e-9)))
    for k in range(min(args.stride, len(chunk))):
        tt = t + k
        if tt > end:
            break
        frames_log[i] = tt
        m_abs[i] = base + chunk[k][ARM_DIMS]        # delta→绝对（基准=本控制步 state）
        d_abs[i] = actions[tt][ARM_DIMS]
        m_g[i] = np.clip(chunk[k][GRIP], 0, 100)
        d_g[i] = np.clip(actions[tt][GRIP], 0, 100)
        i += 1
    if (t - args.start) // args.stride % 20 == 0:
        print(f"  帧 {t}: 推理 {ms_l[-1]:.0f} ms | 累计 "
              f"{time.perf_counter() - t_all:.0f} s", flush=True)
frames_log, m_abs, d_abs = frames_log[:i], m_abs[:i], d_abs[:i]
m_g, d_g = m_g[:i], d_g[:i]

# ── 指标 ──
err = np.abs(m_abs - d_abs)
gerr = np.abs(m_g - d_g)


def first_open(g, thr=20.0):
    idx = np.where(g[:, 0] > thr)[0]
    return int(frames_log[idx[0]]) if len(idx) else None


ds_open, m_open = first_open(d_g), first_open(m_g)
a = np.asarray(ms_l)
print(f"""
== 回放摘要 ep{args.ep}（{i} 帧 / {len(ctrl_ts)} 控制步）==
臂误差   : mean {err.mean():.4f} rad | p95 {np.percentile(err, 95):.4f} | max {err.max():.4f}
delta cos: p50 {np.percentile(cos_l, 50):.3f} | mean {np.mean(cos_l):.3f}（逐控制步首步）
夹爪误差 : mean {gerr.mean():.1f}% | max {gerr.max():.1f}%
开爪时机 : 数据集帧 {ds_open} vs 模型帧 {m_open}
推理延迟 : p50 {np.percentile(a, 50):.0f} / p95 {np.percentile(a, 95):.0f} ms（bf16）""", flush=True)

OUT = pathlib.Path(f"/tmp/tf_rollout_ep{args.ep}.npz")
np.savez(OUT, frames=frames_log, m_abs=m_abs, d_abs=d_abs, m_g=m_g, d_g=d_g,
         t_r=np.asarray(ctrl_ts[:len(cos_l)]), cos=np.asarray(cos_l))
print(f"数据已存 {OUT}", flush=True)

# ── 画图（系统 python3 matplotlib；英文标签避缺字体）──
PLOT = """
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
z = np.load(sys.argv[1])
f, m_abs, d_abs, m_g, d_g = z['frames'], z['m_abs'], z['d_abs'], z['m_g'], z['d_g']
fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
axes[0].plot(f, d_g[:,0], '--', label='dataset'); axes[0].plot(f, m_g[:,0], label='model')
axes[0].set_ylabel('R gripper %'); axes[0].legend(loc='upper left')
axes[1].plot(f, d_g[:,1], '--', label='dataset'); axes[1].plot(f, m_g[:,1], label='model')
axes[1].set_ylabel('L gripper %')
for row, (ji, name) in zip((2, 3), ((1, 'R arm j2/j4 (rad)'), (9, 'L arm j2/j4 (rad)'))):
    axes[row].plot(f, d_abs[:,ji], '--', label='ds j2'); axes[row].plot(f, m_abs[:,ji], label='m j2')
    axes[row].plot(f, d_abs[:,ji+2], '--', label='ds j4'); axes[row].plot(f, m_abs[:,ji+2], label='m j4')
    axes[row].set_ylabel(name); axes[row].legend(loc='upper left', ncol=2, fontsize=8)
axes[4].plot(f, np.abs(m_abs-d_abs).mean(axis=1))
axes[4].set_ylabel('|arm err| rad (mean)'); axes[4].set_xlabel('frame')
for g, ax in ((z['d_g'][:,0], axes[0]), (z['m_g'][:,0], axes[0])):
    idx = np.where(g > 20)[0]
    if len(idx):
        ax.axvline(f[idx[0]], color='g' if g is z['d_g'][:,0] else 'r', ls=':', alpha=.6)
fig.suptitle('tf_rollout: model (solid) vs dataset (dashed), green=ds open, red=model open')
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
print('图已存', sys.argv[2])
"""
subprocess.run(["/usr/bin/python3", "-c", PLOT, str(OUT),
                f"/tmp/tf_rollout_ep{args.ep}.png"], check=True)
