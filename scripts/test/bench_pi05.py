#!/usr/bin/env python
"""pi0.5 推理延迟基准：BF16 vs 编码器 INT8 vs 全 INT8。

用法:
  ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/bench_pi05.py [bf16|int8_enc|int8_full] [迭代次数]

说明:
- 首次 predict 会构建 pipeline + 抓 CUDA Graph（含 autotune），很慢，不计入统计
- 之后每次 predict 走缓存 prompt 的图，测的是稳定延迟
- 输入是全零假数据：只测延迟不校数值（校准文档确认零输入与真实分布 cos>0.998）
"""
import os
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "bf16"
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 30
views = int(os.environ.get("PI05_VIEWS", "2"))  # 2 或 3（真机三摄用 3）

# env 开关必须在 load_model 之前设置
if mode == "int8_enc":
    os.environ["FVK_PI05_RTX_INT8_ENCODER_ONLY"] = "1"
elif mode == "int8_full":
    os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"

import functools
import numpy as np  # noqa: E402
import torch  # noqa: E402
import flash_rt  # noqa: E402

# Orin 上 CUDA graph 抓图（cudaStreamEndCapture）确定性段错误，默认关图跑 eager。
# PI05_NO_GRAPH=0 可重新启用抓图（待排查）。
if os.environ.get("PI05_NO_GRAPH", "1") == "1":
    import flash_rt.frontends.torch.pi05_rtx as _fe

    _orig_init = _fe.Pi05TorchFrontendRtx.__init__

    @functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
        kw["use_cuda_graph"] = False
        _orig_init(self, *a, **kw)

    _fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init
    print("[bench] CUDA graph 已禁用（PI05_NO_GRAPH=1），eager 模式")

CKPT = "/home/galbot/holy/models/pi05_lerobot_base"

print(f"=== 模式: {mode} | 迭代: {n_iter} ===")
t0 = time.time()
model = flash_rt.load_model(CKPT, config="pi05", num_views=views)
print(f"load: {time.time() - t0:.1f}s | num_views={views}")

names = {2: ["image", "wrist_image"],
         3: ["image", "wrist_image", "wrist_image_right"]}[views]
images = {n: np.zeros((224, 224, 3), dtype=np.uint8) for n in names}
state = np.zeros(8, dtype=np.float32)

t0 = time.time()
actions = model.predict(images, prompt="pick up the black bowl on the stove",
                        state=state)
first = time.time() - t0
print(f"首次 predict（建图+抓图）: {first:.1f}s | 输出 shape: {actions.shape}")

lat = []
for i in range(n_iter):
    t0 = time.perf_counter()
    model.predict(images, state=state)
    lat.append((time.perf_counter() - t0) * 1000)

lat = np.array(lat)
print(f"稳定延迟 ms: mean {lat.mean():.1f} | p50 {np.percentile(lat, 50):.1f}"
      f" | min {lat.min():.1f} | max {lat.max():.1f}")
alloc = torch.cuda.memory_allocated() / 2**30
print(f"显存 allocated: {alloc:.1f} GiB")
print(f"=== {mode} 完成 ===")
