#!/usr/bin/env python
"""pi0.5 真机相机推理入口：SDK 取图 → prompt 推理 → 动作打印（只读，不执行任何运动）。

实时性打点分三层:
  ① 纯推理延迟   —— 冻结一帧连跑 --hold 次（CUDA/前端稳定耗时）
  ② 取图+解码    -- 每轮 SDK 抓 3 路 + cv2 软解 224 的 CPU 耗时
  ③ 端到端节拍   -- 取图+推理整轮耗时，折算动作步率（步/秒），并对照
                    控制频率窗口（--ctrl-hz，默认 50Hz：10 步块需 200ms）判定能否跟上
  另与 echo 机基线（BENCHMARKS.md，2026-09-21）对照。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  flashpy ~/holy/scripts/inference/run_pi05_camera.py \
      [--prompt "..."] [--rounds 10] [--hold 10] [--views 3] \
      [--tier int8_full|bf16|int8_enc] [--ctrl-hz 50]
"""
import argparse
import os
import time

# ── 参数 ──
ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="pick up the black bowl on the stove and place it on the plate")
ap.add_argument("--rounds", type=int, default=10, help="连续取图+推理轮数")
ap.add_argument("--hold", type=int, default=10, help="冻结帧纯推理次数")
ap.add_argument("--views", type=int, default=3, choices=(2, 3))
ap.add_argument("--tier", default="int8_full", choices=("bf16", "int8_enc", "int8_full"))
ap.add_argument("--ctrl-hz", type=float, default=50.0, help="控制频率（实时性判定用）")
args = ap.parse_args()

# ── 开关必须在 load_model 前设 ──
if args.tier == "int8_full":
    os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
elif args.tier == "int8_enc":
    os.environ.setdefault("FVK_PI05_RTX_INT8_ENCODER_ONLY", "1")

import functools
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402

# graph 抓图在 r35.5 驱动上段错误，默认绕过（详见 DEPLOY.md 处置表 #6）
os.environ.setdefault("PI05_NO_GRAPH", "1")
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402

CKPT = "/home/galbot/holy/models/pi05_lerobot_base"
VIEW_MAP = {2: [("image", SensorType.HEAD_LEFT_CAMERA),
                ("wrist_image", SensorType.LEFT_ARM_CAMERA)],
            3: [("image", SensorType.HEAD_LEFT_CAMERA),
                ("wrist_image", SensorType.LEFT_ARM_CAMERA),
                ("wrist_image_right", SensorType.RIGHT_ARM_CAMERA)]}
# echo 机基线（BENCHMARKS.md）：(tier, views) → (空载 ms, 真机负载 ms)；无数据的组合为 None
BASELINE = {("int8_full", 2): (161.2, None), ("int8_full", 3): (189.5, 234.0),
            ("bf16", 3): (None, 304.0), ("int8_enc", 2): (183.5, None)}


def grab_one(robot) -> dict | None:
    """抓一路组：SDK 取图 + 软解 + resize224 + BGR→RGB。失败返回 None。"""
    out = {}
    for key, st in VIEW_MAP[args.views]:
        d = robot.get_rgb_data(st)
        if not d or not d.get("data"):
            return None
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        out[key] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out


def stats(ms):
    ms = np.array(ms)
    return (f"mean {ms.mean():.1f} | p50 {np.percentile(ms, 50):.1f} | "
            f"min {ms.min():.1f} | max {ms.max():.1f} ms")


# ── ① SDK 初始化（只读传感器） ──
robot = GalbotRobot()
if not robot.init({st for _, st in VIEW_MAP[args.views]}):
    print("robot.init 失败（机器人上电了吗？相机服务在跑吗？）")
    raise SystemExit(1)
time.sleep(3)

state8 = np.zeros(8, dtype=np.float32)
try:
    arm = robot.get_joint_positions(["right_arm"], [])
    if arm and len(arm) >= 7:
        state8[:7] = np.array(arm[:7], dtype=np.float32)
except Exception:
    pass
print(f"tier={args.tier} views={args.views} ctrl={args.ctrl_hz}Hz "
      f"state[:7]={np.round(state8[:7], 3)}")

# ── ② 加载模型 + 冒烟 ──
t0 = time.time()
model = flash_rt.load_model(CKPT, config="pi05", num_views=args.views, cache_frames=1)
print(f"load: {time.time() - t0:.1f}s")
first = grab_one(robot)
if first is None:
    print("抓图失败（数据未就绪？稍等重试）")
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    raise SystemExit(1)
model.predict(first, prompt=args.prompt, state=state8)   # 建 pipeline（不计入统计）

try:
    # ── ③ ①层：冻结帧纯推理 ──
    lat = []
    for _ in range(args.hold):
        t0 = time.perf_counter()
        model.predict(first, prompt=args.prompt, state=state8)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"[① 纯推理 ×{args.hold}] {stats(lat)}")

    # ── ④ ②③层：连续取图+推理 ──
    grab_ms, infer_ms, round_ms, acts = [], [], [], None
    for i in range(args.rounds):
        t0 = time.perf_counter()
        obs = grab_one(robot)
        t1 = time.perf_counter()
        if obs is None:
            print(f"  轮{i}: 取图失败，跳过")
            continue
        acts = model.predict(obs, prompt=args.prompt, state=state8)
        t2 = time.perf_counter()
        grab_ms.append((t1 - t0) * 1000)
        infer_ms.append((t2 - t1) * 1000)
        round_ms.append((t2 - t0) * 1000)
    n = len(round_ms)
    print(f"[② 取图+解码 ×{n}] {stats(grab_ms)}")
    if n:
        print(f"[③ 端到端节拍 ×{n}] {stats(round_ms)} | 动作步率 "
              f"{10 / (np.mean(round_ms) / 1000):.1f} 步/秒（每轮出 10 步）")

    # ── ⑤ 实时性判定 ──
    window = 10 * 1000.0 / args.ctrl_hz
    p50 = float(np.percentile(infer_ms, 50))
    print(f"[实时性] 控制窗口 = 10 步 ÷ {args.ctrl_hz}Hz = {window:.0f} ms → "
          + ("✅ 推理追得上执行" if p50 <= window
             else f"❌ 推理落后 {(p50 - window):.0f} ms（降控制频率/减视角/修 graph）"))
    base = BASELINE.get((args.tier, args.views))
    if base:
        ref = base[1] if base[1] else base[0]
        tag = "真机负载" if base[1] else "空载"
        delta = p50 - ref
        print(f"[基线对照] echo 机 {args.tier}/{args.views}视角 {tag}: {ref:.1f} ms → "
              f"本次 {'快' if delta < 0 else '慢'} {abs(delta):.1f} ms")
    print(f"[动作示例] 首步: {np.round(np.asarray(acts)[0], 4)}（仅打印，未执行）")
finally:
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("SDK 已关闭")
