#!/usr/bin/env python
"""pi0.5 权重加载验证：只加载不出动作，确认 checkpoint + norm_stats + 前端链路通。

用法:
  ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/load_pi05.py
（新 shell 先: unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD）
"""
import logging
import pathlib
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

import torch  # noqa: E402
import flash_rt  # noqa: E402

CKPT = pathlib.Path("/home/galbot/holy/models/pi05_lerobot_base")

print("flash_rt:", flash_rt.__version__, "| torch:", torch.__version__,
      "| capability:", torch.cuda.get_device_capability())

t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05")
print(f"load_model 完成，耗时 {time.time() - t0:.1f}s")
print("返回类型:", type(model).__name__)

alloc = torch.cuda.memory_allocated() / 2**30
reserved = torch.cuda.memory_reserved() / 2**30
print(f"显存占用: allocated {alloc:.1f} GiB / reserved {reserved:.1f} GiB")
print("OK — 权重加载链路（safetensors + norm_stats + 前端）全部就绪")
