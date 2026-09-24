#!/usr/bin/env python
"""数据集回放推理：pick_place_balence 的帧+state 喂 G1 16 维 pi0.5，与真值动作对比。

与 run_g1_inference.py（真机版）同一条模型链路，只是输入从 SDK 换成数据集：
  npz(extract_dataset_frames.py 产出) → cv2 解码三相机 mp4 帧 →
  state 归一化（数据集夹爪已是 0~100% 训练单位，不做 SDK 宽度换算）→
  model.infer(显式固定噪声——predict 的随机噪声会让单帧对比撞上
  "两两 cos≈0.25"的混沌底，见 ab_real_camera 教训) → (10,16) chunk →
  与数据集未来 10 步真值动作对比余弦/MAE + q01/q99 落域 + OOD 判读。

用法:
  ~/holy/run.sh ~/holy/scripts/inference/run_dataset_replay.py \
      [--ckpt ~/holy/models/pi05_g1_deploy] [--npz ~/holy/datasets/replay_input.npz] \
      [--stride 30] [--max 20] [--seed 0] [--tier int8_full] [--check-graph]
判读提示：场景不复现（图/摆位不符）时模型输出"无条件均值"（臂 ≈0 rad、
夹爪 10~13%）是 OOD 标准行为——cos 全面偏低且动作贴近 act_mean 即此象，
对照 COMMANDS.md §4。
"""
import argparse
import functools
import json
import os
import pathlib
import time

import g1_config  # noqa: E402  同目录共享配置（CLI > config/g1.toml > 内置默认）

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--npz", default="~/holy/datasets/replay_input.npz")
ap.add_argument("--episode", type=int, default=None, help="只回放该 episode（默认 npz 全部）")
ap.add_argument("--stride", type=int, default=30, help="抽帧步长（30=每秒 1 帧）")
ap.add_argument("--max", type=int, default=20, help="最多回放多少帧")
ap.add_argument("--seed", type=int, default=0, help="固定噪声种子（每帧 seed+序号）")
ap.add_argument("--prompt", default=None, help="覆盖任务句（默认用数据集 task 原句）")
ap.add_argument("--tier", default=None, choices=("bf16", "int8_enc", "int8_full"),
                help="量化档（config [inference].tier）")
ap.add_argument("--force-state-dim", action="store_true",
                help="state/stats 维数不符时截断/零补适配（仅 smoke 用）")
ap.add_argument("--check-graph", action="store_true", help="计数 CUDAGraph.replay 确认走图")
ap.add_argument("--config", default=g1_config.DEFAULT_PATH)
args = ap.parse_args()
g1_config.apply(args, {"ckpt": ("run", "ckpt"), "tier": ("inference", "tier")})

# ── 部署清单（与 run_g1_inference.py 同一套）──
CKPT = pathlib.Path(args.ckpt)
mf_p = CKPT / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
ACTION_DIM = int(mf.get("action_dim", 16))
VIEWS = int(mf.get("views", 3))
CAM_MAP = mf.get("camera_map", {"image": "HEAD_RIGHT_CAMERA",
                                "wrist_image": "LEFT_ARM_CAMERA",
                                "wrist_image_right": "RIGHT_ARM_CAMERA"})

if args.tier == "int8_full":
    os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
elif args.tier == "int8_enc":
    os.environ.setdefault("FVK_PI05_RTX_INT8_ENCODER_ONLY", "1")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402

# graph 已修复（L4T r35.6 走 WithFlags，2026-09-24）；回退 eager 设 PI05_NO_GRAPH=1
os.environ.setdefault("PI05_NO_GRAPH", "0")
# state 文本进 prompt：fixed=定长 pipeline 只换 embeds，避免换长重建（~800ms）
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")

if args.check_graph:
    from flash_rt.core import cuda_graph as _cg
    _orig_replay = _cg.CUDAGraph.replay
    _replays = [0]

    @functools.wraps(_orig_replay)
    def _counted(self, stream):
        _replays[0] += 1
        return _orig_replay(self, stream)
    _cg.CUDAGraph.replay = _counted

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402

# ── npz 载入 ──
npz_p = pathlib.Path(args.npz).expanduser()
z = np.load(npz_p, allow_pickle=False)
states, actions = z["states"], z["actions"]
ep_id, frame_idx, task_idx = z["ep_id"], z["frame_idx"], z["task_idx"]
vid_chunk, vid_file, vid_frame = z["vid_chunk"], z["vid_file"], z["vid_frame"]
tasks, cams = z["tasks"].tolist(), z["cams"].tolist()
act_q01, act_q99, act_mean = z["act_q01"], z["act_q99"], z["act_mean"]
video_template = str(z["video_template"])
assert cams == sorted(c for c in cams if "images" in c), "npz 相机键异常"

# 模型 obs 键 → 数据集视频键（manifest 的 SDK 枚举名 → observation.images.*）
SDK2DS = {"HEAD": "observation.images.head_right",
          "LEFT_ARM": "observation.images.left_arm",
          "RIGHT_ARM": "observation.images.right_arm"}
ds_of = {}
for mkey, sdk in CAM_MAP.items():
    hits = [v for k, v in SDK2DS.items() if k in sdk.upper()]
    ds_of[mkey] = hits[0] if hits else None
print(f"obs 键映射: {ds_of} | views={VIEWS}")

# 行选择：同 episode 内 frame_idx+10 仍是有效标签（chunk 窗口不跨轨迹）
mask = frame_idx % max(args.stride, 1) == 0
if args.episode is not None:
    mask &= ep_id == args.episode
# 未来 10 步不越出本 episode
ep_last = {}
for e in np.unique(ep_id):
    ep_last[int(e)] = frame_idx[ep_id == e].max()
rows = np.where(mask)[0]
rows = np.array([i for i in rows if frame_idx[i] + 10 <= ep_last[int(ep_id[i])]])
rows = rows[:args.max]
assert len(rows), "没有可选帧（--max/--stride/--episode 放宽试试）"
print(f"回放 {len(rows)} 帧（stride={args.stride}, seed={args.seed}, "
      f"ckpt={CKPT.name}, tier={args.tier or 'config默认'}）")

# ── mp4 解码（每相机缓存句柄；224×224 已就绪，resize 防御性保留）──
caps = {}
def grab(row) -> dict | None:
    out = {}
    for mkey, dskey in ds_of.items():
        path = dskey and video_template.format(
            video_key=dskey, chunk_index=int(vid_chunk[row]),
            file_index=int(vid_file[row]))
        path = str(npz_p.parent / path) if path else None
        cap = caps.get(path)
        if cap is None:
            cap = caps[path] = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vid_frame[row]))
        ok, img = cap.read()
        if not ok:
            return None
        out[mkey] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return {k: out[k] for k in list(ds_of)[:VIEWS]}

# ── 模型加载（与真机版同参）──
t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=VIEWS,
                            cache_frames=1, action_dim=ACTION_DIM)
ns = model._pipe.norm_stats
MODE = ns.get("norm_mode", "q01_q99") if isinstance(ns, dict) else "q01_q99"
_st_blk = ns.get("state") or ns["actions"] if isinstance(ns, dict) else None


def to_model_state(raw: np.ndarray) -> np.ndarray:
    """数据集原始 state → 归一化输入（夹爪已是训练单位，零换算直喂）。"""
    try:
        return normalize_state(raw, ns)
    except ValueError as e:
        if not args.force_state_dim:
            raise SystemExit(f"state 维数与 stats 不符（{e}）；"
                             "smoke 适配可加 --force-state-dim") from e
        d = len(_st_blk["q01" if MODE == "q01_q99" else "mean"])
        padded = np.zeros(d, np.float32)
        padded[:min(d, len(raw))] = raw[:d]
        return normalize_state(padded, ns)


def _acts(x):
    if isinstance(x, dict):
        x = x.get("actions", x.get("raw_actions",
                    next(v for v in x.values() if hasattr(v, "shape"))))
    return np.asarray(x, dtype=np.float32)


def chunk_vs_labels(chunk: np.ndarray, row: int):
    """chunk (10,16) vs 数据集未来 10 步真值；两种对齐约定都报（数据自证）。"""
    e, f = int(ep_id[row]), int(frame_idx[row])
    s0 = np.where((ep_id == e) & (frame_idx == f))[0][0]
    lab = actions[s0:s0 + 10]              # 0 对齐: chunk[j] ↔ action[t+j]
    lab1 = actions[s0 + 1:s0 + 11]         # 1 对齐: chunk[j] ↔ action[t+1+j]
    def cos_err(l):
        c = [float(chunk[j] @ l[j] /
               (np.linalg.norm(chunk[j]) * np.linalg.norm(l[j]) + 1e-9))
             for j in range(10)]
        mae0 = float(np.abs(chunk[0] - l[0]).mean())
        return c, mae0
    return lab, cos_err(lab), cos_err(lab1)


state0 = to_model_state(states[rows[0]])
print(f"load: {time.time() - t0:.1f}s | norm_mode={MODE} | action_dim={ACTION_DIM}")
model.predict(grab(rows[0]), prompt=str(tasks[task_idx[rows[0]]]),
              state=state0)                                   # 建管线（不计入统计）

gen = torch.Generator().manual_seed(args.seed)
lat, cos0, cos1, ood_flags = [], [], [], 0
print("\n 帧 |  ep | frame |  ms | cos(0对齐) | cos(1对齐) | MAE0")
for n, r in enumerate(rows):
    obs = grab(r)
    if obs is None:
        print(f"  {n:3d} | 解码失败 ep{ep_id[r]} frame{frame_idx[r]}，跳过")
        continue
    prompt = str(tasks[task_idx[r]]) if args.prompt is None else args.prompt
    st_n = to_model_state(states[r])
    noise = torch.randn(10, 32, generator=gen)   # 显式固定噪声（可复现对比）
    t0 = time.perf_counter()
    chunk = _acts(model.infer(obs, noise=noise))
    lat.append((time.perf_counter() - t0) * 1000)
    if chunk.shape[0] != 10 or chunk.shape[1] != ACTION_DIM:
        raise SystemExit(f"输出 shape {chunk.shape} ≠ (10,{ACTION_DIM})")
    _, (c0, m0), (c1, m1) = chunk_vs_labels(chunk, r)
    cos0.append(float(np.mean(c0))); cos1.append(float(np.mean(c1)))
    out_q = float(np.mean((chunk < act_q01) | (chunk > act_q99)))
    if abs(float(chunk[0] @ act_mean / (np.linalg.norm(chunk[0]) * np.linalg.norm(act_mean) + 1e-9))) > 0.99 \
            and float(np.abs(chunk[0][[7, 15]].mean() - 12)) < 4:
        ood_flags += 1   # 动作贴均值+夹爪 10~13%：COMMANDS.md §4 的 OOD 特征
    print(f" {n:3d} | {ep_id[r]:3d} | {frame_idx[r]:5d} | {lat[-1]:3.0f} "
          f"| {np.mean(c0):.3f} | {np.mean(c1):.3f} | {m0:.3f}"
          + ("  ← OOD 嫌疑" if ood_flags and n and False else ""))

print(f"\n[延迟 ×{len(lat)}] mean {np.mean(lat):.1f} | p50 "
      f"{np.percentile(lat, 50):.1f} | max {np.max(lat):.1f} ms")
print(f"[对比真值] 平均 cos：0对齐 {np.mean(cos0):.3f} | 1对齐 {np.mean(cos1):.3f}"
      "（哪个高说明该数据集的 chunk 对齐约定；单帧 cos 混沌底 ≈0.25）")
print(f"[落域] chunk 步越出数据集 q01~q99 的维占比 = "
      f"{out_q:.0%}；OOD 嫌疑帧 {ood_flags}/{len(rows)}")
if args.check_graph:
    print(f"[graph] CUDAGraph.replay 调用 {_replays[0]} 次（>0=图重放生效）")
print("（开环单帧对比仅验链路/维序/归一化；真机 KPI 看闭环执行成功率）")
for p in caps.values():
    p.release()
