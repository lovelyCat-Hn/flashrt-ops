#!/usr/bin/env python
"""Teacher-forced 诊断矩阵：训练集输入直接推理，输出 delta vs 数据集 delta（离线）。

背景（2026-09-23）：真机排障定论"喂训练集数据输出仍偏 0"。本脚本把该观察
做成系统量化，回答三个问题（全程不碰机器人）：
  ① 偏 0 有多严重：|输出 delta| 量级 vs 数据集 delta、余弦相关性、逐维分布
  ② 模型在读输入吗：换图 / 换 state / 灰图 消融——输出对这些变化有没有反应
  ③ 装配环节影响：prompt 带括号（此前离线探针的错误写法）/ 换任务句 /
     state 不归一化——定位输出异常是否由输入装配引起

数据集事实：3 路相机共用 2 个 mp4，episode 按时间戳分段
（帧 t → from_timestamp + t/30 s）；parquet 全部 episode 在
data/chunk-000/file-000.parquet，episode_index 列过滤。
对比基准：数据集 delta = action[t] - state[t]（臂维 0-6 右 / 8-14 左；
夹爪维 7/15 是绝对 %，单列对比）。

用法:
  ~/holy/run.sh ~/holy/scripts/eval/tf_matrix.py [--ep 0] [--frames 0,100,300]
"""
import argparse
import functools
import os
import pathlib
import sys
import time

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--ep", type=int, default=0, help="数据集 episode 号（默认 0）")
ap.add_argument("--frames", default="0,50,150,300,450",
                help="主扫描帧号列表（逗号分隔，默认 0,50,150,300,450）")
ap.add_argument("--tier", choices=("int8", "enc_only", "bf16"), default="int8",
                help="量化档：int8=全 INT8 / enc_only=编码器 INT8+解码器 bf16 / "
                     "bf16=全 bf16（2026-09-23 实证：全 INT8 解码器毁动作质量）")
ap.add_argument("--config", default=None, help="g1.toml 路径（默认内置查找）")
args = ap.parse_args()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "inference"))
import g1_config  # noqa: E402

_cfg = argparse.Namespace(ckpt=args.ckpt)
g1_config.apply(_cfg, {"ckpt": ("run", "ckpt")})
CKPT = pathlib.Path(_cfg.ckpt)

# ── 开关必须在 load_model 之前设 ──
if args.tier == "int8":
    os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
elif args.tier == "enc_only":
    os.environ["FVK_PI05_RTX_INT8_ENCODER_ONLY"] = "1"
os.environ.setdefault("PI05_NO_GRAPH", "1")
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import subprocess  # noqa: E402
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402

# ── 数据集 ──
# 本环境无 pandas/parquet 库：用系统 python3（/usr/bin/python3，有 pandas）
# 把所需 episode 抽成 npz 缓存（state/action 逐帧 + 元信息），本进程只读 npz。
DS = pathlib.Path("/home/galbot/holy/datasets/pick_place_balence")
CAM_KEY = {"image": "head_right", "wrist_image": "left_arm",
           "wrist_image_right": "right_arm"}     # 模型视图名 → 数据集相机名
ARM_DIMS = list(range(0, 7)) + list(range(8, 15))   # 14 臂维（右 7 在前）
GRIP_DIMS = (7, 15)

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

_z = np.load(CACHE, allow_pickle=False)
_states, _actions = _z["states"], _z["actions"]
TASKS_TXT = str(_z["tasks"][0])
T0_TS = {"head_right": float(_z["t0_head"]), "left_arm": float(_z["t0_left"]),
         "right_arm": float(_z["t0_right"])}
print(f"episode {args.ep}: {len(_states)} 帧 | tasks {TASKS_TXT} | "
      f"视频起点 {T0_TS['head_right']:.2f}s", flush=True)


def frame_at(cam, t):
    """数据集第 t 帧（按时间戳在共享 mp4 内定位）。"""
    p = DS / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_MSEC, (T0_TS[cam] + t / 30.0) * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise SystemExit(f"取帧失败: ep{args.ep} t={t} {cam}")
    return np.ascontiguousarray(
        cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))


def ep_state(t):
    return _states[t]


def ep_delta(t):
    """数据集 delta：action[t] - state[t]（臂维；夹爪维绝对值不动）。"""
    return _actions[t][ARM_DIMS] - _states[t][ARM_DIMS], _actions[t][list(GRIP_DIMS)]


FRAMES = [int(x) for x in args.frames.split(",")]
FRAMES = [min(t, len(_states) - 11) for t in FRAMES]
frames_cache = {}   # t → {视图名: 图}

# ── 模型 ──
print(f"\n加载模型 {CKPT} ...", flush=True)
t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
ns = model._pipe.norm_stats
print(f"模型就绪 load {time.time() - t0:.1f}s | norm_mode="
      f"{ns.get('norm_mode', 'q01_q99') if isinstance(ns, dict) else '?'}", flush=True)

TASK0 = "Left arm pick up A. Right arm pick up A."


def run(images, state_23, prompt):
    """一次推理 → (首步臂维 delta, 夹爪两维, 耗时 ms)。state_23 传原始值，内部归一。"""
    t1 = time.perf_counter()
    chunk = np.asarray(model.predict(images, prompt=prompt,
                                     state=normalize_state(np.asarray(state_23, np.float32), ns)))
    ms = (time.perf_counter() - t1) * 1000
    return chunk[0][ARM_DIMS], chunk[0][list(GRIP_DIMS)], chunk, ms


def cos(a, b):
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


# ── ① 主扫描：同帧输入，模型 delta vs 数据集 delta（单步 + chunk 均值两口径）──
print(f"\n== ① 主扫描 ep{args.ep}（干净 prompt，逐帧 teacher-forced，{args.tier} 档）==",
      flush=True)
print(f"{'帧':>5} {'cos单步':>8} {'cos块均':>8} {'|m块均|':>8} {'|ds块均|':>8} "
      f"{'max|model|':>10} {'爪m/ds':>13}")
cos_l, cos10_l = [], []
for t in FRAMES:
    if t not in frames_cache:
        frames_cache[t] = {v: frame_at(c, t) for v, c in CAM_KEY.items()}
    m_d, m_g, chunk, ms = run(frames_cache[t], ep_state(t), TASK0)
    d_d, d_g = ep_delta(t)
    m10 = chunk[:10][:, ARM_DIMS].mean(axis=0)
    ds10 = np.stack([ep_delta(tt)[0] for tt in range(t, min(t + 10, len(_states)))
                     ]).mean(axis=0)
    cos_l.append(cos(m_d, d_d))
    cos10_l.append(cos(m10, ds10))
    print(f"{t:>5} {cos_l[-1]:>8.3f} {cos10_l[-1]:>8.3f} {np.abs(m10).mean():>8.3f} "
          f"{np.abs(ds10).mean():>8.3f} {np.abs(m_d).max():>10.3f} "
          f"{m_g[0]:>5.1f}/{d_g[0]:<5.1f} | {ms:.0f} ms", flush=True)

# ── ② 消融矩阵（全部在帧 0）──
print("\n== ② 消融矩阵（帧 0；对比基准=主扫描帧 0 输出）==", flush=True)
base_img = frames_cache[FRAMES[0]]
st0, st_end = ep_state(FRAMES[0]), ep_state(min(FRAMES[0] + 300, len(_states) - 11))
base_d = run(base_img, st0, TASK0)[0]
gray = {v: np.full((224, 224, 3), 128, np.uint8) for v in CAM_KEY}
ablations = [
    ("带括号prompt（旧探针错误写法）", base_img, st0,
     f"['{TASK0}']"),
    ("task1 指令句（换任务）", base_img, st0,
     "Left arm places A in the top-left corner. Right arm places A in the top-left corner."),
    ("全灰图（模型还看得到场景吗）", gray, st0, TASK0),
    ("图=帧0 + state=帧+300（错配）", base_img, st_end, TASK0),
    ("左右腕相机互换（验槽位映射）",
     {"image": base_img["image"], "wrist_image": base_img["wrist_image_right"],
      "wrist_image_right": base_img["wrist_image"]}, st0, TASK0),
]
for name, imgs, st, pr in ablations:
    m_d, m_g, _, ms = run(imgs, st, pr)
    dev = float(np.linalg.norm(m_d - base_d))
    print(f"{name:<28} |d|={np.abs(m_d).mean():.3f} max={np.abs(m_d).max():.3f} "
          f"cos(基线)={cos(m_d, base_d):.3f} L2(基线)={dev:.3f} "
          f"爪={m_g[0]:.1f}% | {ms:.0f} ms", flush=True)

# ── ③ 判读 ──
mc, mc10 = float(np.mean(cos_l)), float(np.mean(cos10_l))
print(f"""
== ③ 判读（{args.tier} 档）==
主扫描 {len(cos_l)} 帧：平均 cos 单步={mc:.3f} / 块均={mc10:.3f}
  2026-09-23 定档实测（ep0 五帧）：
    int8     → cos 单步 0.07 / 块均 0.15，夹爪输出 8~21%（数据集 0%）❌
    enc_only → cos 单步 0.28 / 块均 0.27，夹爪 5~19% ❌（编码器 INT8 同样有毒）
    bf16     → cos 单步 0.90 / 块均 0.98，夹爪 0.2~0.4% ✅ 复现训练行为
  结论：微调权重本身是好的；INT8 量化（无论档位）是"伸臂不抓"根因，
  部署定档 bf16。换 ckpt / 换 FlashRT 版本后用本脚本三档复测。""")
