#!/usr/bin/env python
"""G1 双臂 16 维 pi0.5 整体推理入口（真机版，只读不执行任何运动）。

链路: SDK 三相机取图 + SDK 显式名读 23 维关节 → state 归一化（ckpt 自带
norm_stats 的语义）→ pi0.5 推理 → (10, 16) 动作打印 + 三层实时性打点。

配置自动来自部署目录的 flashrt_deploy.json（g1_ckpt_prep.py 产出）：
  action_dim / state_dim / views / camera_map / gripper 标定 / norm_mode
没有 manifest 时用内置 G1 缺省。换微调权重 = 换 --ckpt 目录，零代码改动。

state 23 维装配（与 pick_place_balence 数据集逐维对齐——⚠【右臂在前】，
meta/info.json 权威定义；显式名字读取，group 模式返回顺序不可信）:
  [0:7]   right_arm_joint1..7 (rad)
  [7]     right_gripper_joint1  SDK 开口宽度(米) → 百分比(需标定)
  [8:15]  left_arm_joint1..7 (rad)
  [15]    left_gripper_joint1 同上
  [16:21] leg_joint1..5
  [21:23] head_joint1..2
曾按"左臂在前"装配，把右臂统计喂给左臂读数（首跑 [-1,1] 外 15 维的主因
之一，预热也把右臂目标下给左臂、机械臂甩到背后）——2026-09-22 修正。

用法（需要 SDK 环境）:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/run_g1_inference.py \
      [--ckpt ~/holy/models/pi05_g1_smoke] [--prompt "..."] \
      [--rounds 10] [--hold 10] [--tier int8_full] [--ctrl-hz 50] \
      [--grip-wmin 0.0 --grip-wmax 0.08]
"""
import argparse
import functools
import json
import os
import pathlib
import time

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_smoke")
ap.add_argument("--prompt", default="Left arm pick up the block. Right arm pick up the block.")
ap.add_argument("--rounds", type=int, default=10, help="连续取图+推理轮数")
ap.add_argument("--hold", type=int, default=10, help="冻结帧纯推理次数")
ap.add_argument("--tier", default="int8_full", choices=("bf16", "int8_enc", "int8_full"))
ap.add_argument("--ctrl-hz", type=float, default=30.0,
                help="控制频率（数据集 fps=30，10 步 chunk 窗口=333ms）")
ap.add_argument("--grip-wmin", type=float, help="夹爪零开度 SDK 宽度(米)，覆盖 manifest")
ap.add_argument("--grip-wmax", type=float, help="夹爪满开度 SDK 宽度(米)，覆盖 manifest")
ap.add_argument("--force-state-dim", action="store_true",
                help="state/stats 维数不符时截断/零补适配（仅 smoke 用；真微调 ckpt 勿开）")
args = ap.parse_args()

# ── 部署清单 ──
CKPT = pathlib.Path(args.ckpt)
mf_p = CKPT / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
ACTION_DIM = int(mf.get("action_dim", 16))
VIEWS = int(mf.get("views", 3))
CAM_MAP = mf.get("camera_map", {"image": "HEAD_RIGHT_CAMERA",
                                "wrist_image": "LEFT_ARM_CAMERA",
                                "wrist_image_right": "RIGHT_ARM_CAMERA"})
grip = mf.get("gripper", {})
GRIP_WMIN = args.grip_wmin if args.grip_wmin is not None else grip.get("width_min")
GRIP_WMAX = args.grip_wmax if args.grip_wmax is not None else grip.get("width_max")

# ── 开关必须在 load_model 前设 ──
if args.tier == "int8_full":
    os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
elif args.tier == "int8_enc":
    os.environ.setdefault("FVK_PI05_RTX_INT8_ENCODER_ONLY", "1")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402

# graph 抓图在 r35.5 驱动上段错误，默认绕过（DEPLOY.md 处置表 #6）
os.environ.setdefault("PI05_NO_GRAPH", "1")
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402

# echo 机基线（BENCHMARKS.md）：(tier, views) → (空载纯推理, 负载纯推理, 负载端到端) ms
# 注意分层：234/304 是负载纯推理；端到端另列，勿跨层对照（2026-09-22 修正）
BASELINE = {("int8_full", 3): (189.5, 235.0, 257.5), ("bf16", 3): (None, 304.0, None)}

# ── state 装配表（显式名，顺序即数据集维序：右臂在前！）──
STATE_NAMES = ([f"right_arm_joint{i}" for i in range(1, 8)]
               + ["right_gripper_joint1"]
               + [f"left_arm_joint{i}" for i in range(1, 8)]
               + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])
GRIP_IDX = (7, 15)   # dim7=右夹爪, dim15=左夹爪（数据集 0~100% 语义）


def read_state(robot) -> tuple[np.ndarray, list[str]]:
    """显式名字读 23 维 state；返回 (state, 未读到/降级的维说明)。"""
    vals = robot.get_joint_positions([], STATE_NAMES)
    notes = []
    if not vals or len(vals) != len(STATE_NAMES):
        notes.append(f"关节读取失败/长度异常({len(vals) if vals else 0})，全部置 0")
        return np.zeros(len(STATE_NAMES), np.float32), notes
    st = np.array(vals, dtype=np.float32)
    if GRIP_WMIN is None or GRIP_WMAX is None:
        notes.append("夹爪未标定(width_min/max)，SDK 宽度原样透传（须与训练单位一致）")
    else:
        for i in GRIP_IDX:
            st[i] = (st[i] - GRIP_WMIN) / (GRIP_WMAX - GRIP_WMIN + 1e-9) * 100.0
    return st, notes


def grab_views(robot) -> dict | None:
    """按 manifest 相机映射抓 VIEWS 路 + 软解 224 + BGR→RGB。失败 None。"""
    out = {}
    for key, cam in CAM_MAP.items():
        if len(out) >= VIEWS:
            break
        d = robot.get_rgb_data(getattr(SensorType, cam))
        if not d or not d.get("data"):
            return None
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        out[key] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out if len(out) == VIEWS else None


def stats(ms):
    ms = np.array(ms)
    return (f"mean {ms.mean():.1f} | p50 {np.percentile(ms, 50):.1f} | "
            f"min {ms.min():.1f} | max {ms.max():.1f} ms")


# ── ① SDK 初始化（只读）──
robot = GalbotRobot()
sensors = {getattr(SensorType, c) for c in list(CAM_MAP.values())[:VIEWS]}
if not robot.init(sensors):
    print("robot.init 失败（机器人上电了吗？相机服务在跑吗？）")
    raise SystemExit(1)
time.sleep(3)

# ── ② state 装配 + 模型加载 ──
state_raw, notes = read_state(robot)
for n in notes:
    print(f"[state] {n}")
print(f"tier={args.tier} views={VIEWS} action_dim={ACTION_DIM} "
      f"ckpt={CKPT.name}\nstate 原始: {np.round(state_raw, 3).tolist()}")

t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=VIEWS,
                            cache_frames=1, action_dim=ACTION_DIM)
ns = model._pipe.norm_stats
MODE = ns.get("norm_mode", "q01_q99") if isinstance(ns, dict) else "q01_q99"
_st_blk = ns.get("state") or ns["actions"] if isinstance(ns, dict) else None


def to_model_state(raw: np.ndarray) -> np.ndarray:
    """SDK 原始 state → 归一化模型输入；维数不符且开了 --force-state-dim 时适配。"""
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


state_n = to_model_state(state_raw)
if args.force_state_dim and len(state_n) != len(state_raw):
    print(f"[state] ⚠ 截断/零补 23→{len(state_n)} 维适配 smoke stats"
          "（真微调 ckpt 勿这样跑）")
sat = [f"dim{i}" for i in range(len(state_n)) if abs(state_n[i]) > 1.0]
print(f"load: {time.time() - t0:.1f}s | norm_mode={MODE} | "
      f"state 归一化 [-1,1] 外 {len(sat)} 维: {sat or '无'}")

first = grab_views(robot)
if first is None:
    print("抓图失败（数据未就绪？稍等重试）")
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    raise SystemExit(1)
model.predict(first, prompt=args.prompt, state=state_n)   # 建管线（不计入统计）

try:
    # ── ③ ①层：冻结帧纯推理 ──
    lat = []
    for _ in range(args.hold):
        t0 = time.perf_counter()
        model.predict(first, prompt=args.prompt, state=state_n)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"[① 纯推理 ×{args.hold}] {stats(lat)}")

    # ── ④ ②③层：连续取图+推理（每轮重读 state）──
    grab_ms, infer_ms, round_ms, acts = [], [], [], None
    for i in range(args.rounds):
        t0 = time.perf_counter()
        obs = grab_views(robot)
        state_i, _ = read_state(robot)
        state_n_i = to_model_state(state_i)
        t1 = time.perf_counter()
        if obs is None:
            print(f"  轮{i}: 取图失败，跳过")
            continue
        acts = model.predict(obs, prompt=args.prompt, state=state_n_i)
        t2 = time.perf_counter()
        grab_ms.append((t1 - t0) * 1000)
        infer_ms.append((t2 - t1) * 1000)
        round_ms.append((t2 - t0) * 1000)
    n = len(round_ms)
    print(f"[② 取图+读关节+解码 ×{n}] {stats(grab_ms)}")
    if n:
        print(f"[③ 端到端节拍 ×{n}] {stats(round_ms)} | 动作步率 "
              f"{10 / (np.mean(round_ms) / 1000):.1f} 步/秒（每轮出 10 步）")

    # ── ⑤ 实时性判定 ──
    window = 10 * 1000.0 / args.ctrl_hz
    p50 = float(np.percentile(infer_ms, 50)) if infer_ms else float("nan")
    print(f"[实时性] 控制窗口 = 10 步 ÷ {args.ctrl_hz}Hz = {window:.0f} ms → "
          + ("✅ 推理追得上执行" if p50 <= window
             else f"❌ 推理落后 {(p50 - window):.0f} ms（降控制频率/减视角/修 graph）"))
    base = BASELINE.get((args.tier, VIEWS))
    if base and n:
        i_p50 = float(np.percentile(infer_ms, 50))
        r_p50 = float(np.percentile(round_ms, 50))
        if base[1]:
            print(f"[基线对照] echo 机 {args.tier}/{VIEWS}视角 负载纯推理 {base[1]:.1f} ms → "
                  f"本次 {'快' if i_p50 < base[1] else '慢'} {abs(i_p50 - base[1]):.1f} ms")
        if base[2]:
            print(f"[基线对照] echo 机 负载端到端 {base[2]:.1f} ms → "
                  f"本次 {'快' if r_p50 < base[2] else '慢'} {abs(r_p50 - base[2]):.1f} ms")
    if acts is not None:
        a = np.asarray(acts)
        print(f"[动作示例] 首步({a.shape[1]} 维): {np.round(a[0], 3).tolist()}")
        print("（仅打印，未执行。夹爪维应在 0~100；关节维应与 state 同量级）")
finally:
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("SDK 已关闭")
