#!/usr/bin/env python
"""sdkfree_infer_probe.py — 无 SDK 进程、文件帧冻结观测的纯推理计时（只读）。

用途：把「剩余环境差值」切成两半——
  A. 本进程持 SDK 订阅/线程的成本（本探针=0，与 run_g1_inference ① 层对照）
  B. 采集守护等栈侧成本（A/B 都含，对照空载基准）
观测从磁盘 JPEG 喂入（sdk_camera_smoke.py 存的帧），全程一次 get_rgb_data 都没有。

用法（flash_pyrt311）:
  python sdkfree_infer_probe.py [--ckpt 路径] [--rounds 20] [--prompt "..."]
对照：
  run_g1_inference --hold 20（①层,同帧冻结）= 本值 + SDK 在场成本
"""
import argparse
import functools
import os
import pathlib
import sys
import time

os.environ.setdefault("PI05_NO_GRAPH", "0")
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
os.environ["FLASH_RT_PI05_ACTION_CHUNK_SIZE"] = "50"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_pick_deploy")
ap.add_argument("--prompt",
                default="Left arm pick up A. Right arm pick up A.")
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--frames", default="/tmp/sdk_cam_test")
args = ap.parse_args()

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402

# 三视角映射与 run_g1_inference 同款（manifest camera_map 缺省）
VIEWS = {
    "image": "head_right.jpg",
    "wrist_image": "left_color.jpg",
    "wrist_image_right": "right_color.jpg",
}


def load_obs():
    out = {}
    for key, fn in VIEWS.items():
        img = cv2.imread(os.path.join(args.frames, fn))
        assert img is not None, fn
        out[key] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out


obs = load_obs()
t0 = time.time()
model = flash_rt.load_model(args.ckpt, config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
ns = model._pipe.norm_stats
state_n = normalize_state(np.zeros(23, np.float32), ns)

model.predict(obs, prompt=args.prompt, state=state_n)  # 建管线，不计入

lat = []
for _ in range(args.rounds):
    t = time.perf_counter()
    model.predict(obs, prompt=args.prompt, state=state_n)
    lat.append((time.perf_counter() - t) * 1000)
ms = np.array(lat)
print(f"[无SDK冻结帧 ×{args.rounds}] mean {ms.mean():.1f} | p50 "
      f"{np.percentile(ms, 50):.1f} | min {ms.min():.1f} | max {ms.max():.1f} ms"
      f" | load {time.time()-t0:.1f}s")
