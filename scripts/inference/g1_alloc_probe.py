#!/usr/bin/env python
"""③层偶发慢推理归因探针（只读，真机）：新鲜张量 vs 固定缓冲。

背景（2026-09-23）：冻结帧纯推理 230ms 稳定，但每轮新取图的推理
230~850ms 大方差（服务负载不变）——嫌疑=输入张量地址/对齐变化触发
引擎慢路径。三组对照（同一模型实例内，勿跨实例比较——已知坑）：
  A 固定 obs 连打 4 发       → 预期全快（基线）
  B 每发前新取图（旧③路径）  → 预期偶发/频繁慢
  C 固定缓冲原地解码后打 4 发 → 预期全快 ⇒ 修复=缓冲复用
"""
import argparse
import functools
import os
import sys
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--long", type=float, default=0.0,
                help="追加 N 秒长循环（逐发打印时间戳+耗时），定位间歇慢窗口")
args = ap.parse_args()

os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
os.environ.setdefault("PI05_NO_GRAPH", "1")

import numpy as np  # noqa: E402
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
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402

CKPT = "/home/galbot/holy/models/pi05_g1_deploy"
PROMPT = "Left arm pick up the block. Right arm pick up the block."
CAM = (("image", "HEAD_RIGHT_CAMERA"), ("wrist_image", "LEFT_ARM_CAMERA"),
       ("wrist_image_right", "RIGHT_ARM_CAMERA"))
STATE_NAMES = ([f"right_arm_joint{i}" for i in range(1, 8)]
               + ["right_gripper_joint1"]
               + [f"left_arm_joint{i}" for i in range(1, 8)]
               + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])

sensors = {getattr(SensorType, c) for _, c in CAM}
robot = GalbotRobot()
if not robot.init(sensors):
    raise SystemExit("robot.init 失败")
time.sleep(3)

model = flash_rt.load_model(CKPT, config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
ns = model._pipe.norm_stats

# 固定缓冲（C 组用）：地址全程不变，原地解码
BUFS = {k: np.empty((224, 224, 3), np.uint8) for k, _ in CAM}


def grab(dst=None) -> dict:
    """dst=None 每次新分配（A/B 组）；否则原地写入固定缓冲（C 组）。"""
    out = {}
    for key, cam in CAM:
        d = robot.get_rgb_data(getattr(SensorType, cam))
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        small = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if dst is None:
            out[key] = np.ascontiguousarray(rgb)
        else:
            dst[key][:] = rgb
            out[key] = dst[key]
    return out


state_n = normalize_state(
    np.array(robot.get_joint_positions([], STATE_NAMES), np.float32), ns)


def batch(tag, obs, n=4):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model.predict(obs, prompt=PROMPT, state=state_n)
        ts.append((time.perf_counter() - t) * 1000)
    print(f"{tag}: {' | '.join(f'{x:.0f}' for x in ts)} ms", flush=True)


model.predict(grab(), prompt=PROMPT, state=state_n)   # 建管线
batch("A 固定obs 连打    ", grab())
for _ in range(3):
    batch("B 新取图→打(旧③) ", grab(), n=1)         # 每发前新取图（逐发）
batch("C 固定缓冲原地解码", grab(BUFS))
batch("C 固定缓冲再打    ", BUFS)

if args.long > 0:
    print(f"── 长循环 {args.long:.0f}s ──", flush=True)
    t_end, i = time.time() + args.long, 0
    while time.time() < t_end:
        i += 1
        t = time.perf_counter()
        model.predict(grab(), prompt=PROMPT, state=state_n)
        print(f"L{i:03d} {time.strftime('%H:%M:%S')} "
              f"{(time.perf_counter() - t) * 1000:.0f} ms", flush=True)

robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
os._exit(0)
