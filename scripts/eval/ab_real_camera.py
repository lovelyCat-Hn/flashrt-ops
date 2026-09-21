#!/usr/bin/env python
"""真机相机图 A/B：BF16 vs 全 INT8，只推理不执行。

流程:
  1. galbot_sdk 只读方式抓 3 路相机（HEAD_LEFT / LEFT_ARM / RIGHT_ARM），
     cv2 软解 + resize 224 + BGR→RGB；抓 N 组后立即关闭 SDK（不碰任何运动接口）
  2. 同一批图像喂两个档位（BF16 / FORCE_INT8），predict 前同种子对齐初始噪声
  3. BF16 下同种子双跑一次做确定性自检（差异应为 0，否则对比方法失效需注明）
  4. 输出未归一化动作余弦 / 绝对误差；抓到的图存 /tmp/cam_samples.png

用法（SDK 运行库路径必须带上）:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/ab_real_camera.py [N组]
"""
import os
import sys

import functools
import numpy as np
import cv2
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEED = 20260921
PROMPT = "pick up the black bowl on the stove and place it on the plate"

# ── 无 graph（本机抓图段错误绕过，见 bench_pi05.py） ──
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402

VIEWS = [
    (SensorType.HEAD_LEFT_CAMERA, "image"),
    (SensorType.LEFT_ARM_CAMERA, "wrist_image"),
    (SensorType.RIGHT_ARM_CAMERA, "wrist_image_right"),
]
CKPT = "/home/galbot/holy/models/pi05_lerobot_base"


def grab_one(robot) -> dict | None:
    out = {}
    for st, key in VIEWS:
        d = None
        for _ in range(6):  # 帧未就绪时重试
            d = robot.get_rgb_data(st)
            if d and d.get("data"):
                break
            time.sleep(0.5)
        if not d or not d.get("data"):
            return None
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (224, 224))
        out[key] = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


import time  # noqa: E402

# ── ① 只读抓图 ──
robot = GalbotRobot()
ok = robot.init({st for st, _ in VIEWS})
if not ok:
    print("robot.init 失败（相机服务未就绪？）")
    sys.exit(1)
time.sleep(3)

state8 = np.zeros(8, dtype=np.float32)
try:
    arm = robot.get_joint_positions(["right_arm"], [])
    if arm and len(arm) >= 7:
        state8[:7] = np.array(arm[:7], dtype=np.float32)
        print(f"真实右臂关节(前7维): {state8[:7]}")
except Exception as e:
    print(f"读关节失败，state 用零: {e}")

samples = []
for i in range(N):
    s = grab_one(robot)
    if s is None:
        print(f"第 {i} 组抓图失败，跳过")
        continue
    samples.append(s)
    time.sleep(1.0)

robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
print(f"SDK 已关闭，抓到 {len(samples)} 组图像")
if not samples:
    sys.exit(1)

# 存拼图供过目
rows = []
for s in samples:
    rows.append(np.concatenate([s[key] for _, key in VIEWS], axis=1))
cv2.imwrite("/tmp/cam_samples.png", cv2.cvtColor(np.concatenate(rows, axis=0),
                                                 cv2.COLOR_RGB2BGR))
print("图像拼图: /tmp/cam_samples.png")

# ── ② 两档推理（固定噪声：predict 建流水线，infer(noise=) 出对比动作） ──
# 噪声在 flow-matching 隐动作空间，形状 (chunk=10, 32)；每样本一份，跨档共用
fixed_noises = [torch.randn(10, 32) for _ in samples]

acts = {}
for mode, flags in (("bf16", {}), ("int8_full", {"FVK_PI05_RTX_FORCE_INT8": "1"})):
    os.environ.update(flags)
    model = flash_rt.load_model(CKPT, config="pi05", num_views=3, cache_frames=1)
    # 建 prompt/state 流水线 + 预热（这两次结果不记录）
    model.predict(samples[0], prompt=PROMPT, state=state8)
    model.predict(samples[0], prompt=PROMPT, state=state8)
    def _acts(x):
        if isinstance(x, dict):
            x = x.get("actions", x.get("raw_actions", next(v for v in x.values()
                                                           if hasattr(v, "shape"))))
        return np.asarray(x, dtype=np.float32)

    out = []
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        a = _acts(model.infer(s, noise=fixed_noises[i]))
        dt = (time.perf_counter() - t0) * 1000
        out.append(a)
        print(f"[{mode}] 样本{i}: {dt:.0f} ms")
    if mode == "bf16":  # 确定性自检：同噪声双跑，应严格为 0
        a2 = _acts(model.infer(samples[0], noise=fixed_noises[0]))
        d = np.abs(a2 - out[0]).max()
        print(f"[确定性自检] BF16 同噪声双跑 max_diff = {d:.2e} "
              f"({'OK' if d == 0 else '非零！kernel 非确定或噪声未对齐'})")
    acts[mode] = np.stack(out)
    del model
    torch.cuda.empty_cache()

# ── ③ 对比 ──
ref, tgt = acts["bf16"], acts["int8_full"]
print("\n===== 真机相机图 A/B（相对 BF16） =====")
cos_all = []
for i in range(len(samples)):
    a, b = ref[i].ravel(), tgt[i].ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cos_all.append(cos)
    print(f"样本{i}: cos={cos:.5f}  bf16动作[0]={a[:3]}  int8动作[0]={b[:3]}")
d = np.abs(tgt - ref)
print(f"\nint8_full: cos mean {np.mean(cos_all):.5f} | min {np.min(cos_all):.5f} | "
      f"max_abs_diff {d.max():.5f} | mean_abs_diff {d.mean():.5f}")
