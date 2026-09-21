#!/usr/bin/env python
"""pi0.5 INT8 量化加载（Orin sm87 路线）。

要点（来自 docs/calibration.md 与 pi05_rtx.py 代码注释，2026-09-21 核对）：
- Orin 上 INT8 不走 load_model(precision=...)（那是 Ascend 专用，其他架构会 raise），
  而是走环境变量开关，在前端 __init__ 时读取：
    FVK_PI05_RTX_INT8_ENCODER_ONLY=1  编码器 INT8 + 解码器 BF16 —— 推荐：
        编码器 M 大收益高；解码器 M=10，INT8 CUTLASS tile 浪费比 cuBLASLt BF16 还慢
    FVK_PI05_RTX_FORCE_INT8=1         编码器+解码器全 INT8 —— 仅 A/B 用
    FVK_PI05_RTX_INT8_VISION=1        vision 动态逐行 INT8 —— 未验证，别开
        （vision 静态 per-tensor 曾把 encoder cosine 打到 0.282，已永久禁用）
- INT8 权重 scale：加载时按行静态算（absmax/127）
- INT8 激活 scale：运行时动态，**不需要校准产物**（FP8/THOR 那套才需要）
- 自检：日志出现 "INT8 quantized N encoder GEMM weights" 即生效

用法:
  ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/load_pi05_int8.py
（新 shell 先: unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD）
"""
import logging
import os
import pathlib
import time

# 必须在 load_model 构造前端之前设置
os.environ.setdefault("FVK_PI05_RTX_INT8_ENCODER_ONLY", "1")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

import torch  # noqa: E402
import flash_rt  # noqa: E402

CKPT = pathlib.Path("/home/galbot/holy/models/pi05_lerobot_base")

print("flash_rt:", flash_rt.__version__, "| torch:", torch.__version__,
      "| capability:", torch.cuda.get_device_capability())
print("开关: FVK_PI05_RTX_INT8_ENCODER_ONLY =",
      os.environ["FVK_PI05_RTX_INT8_ENCODER_ONLY"])

t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05")
print(f"load_model 完成，耗时 {time.time() - t0:.1f}s")
print("返回类型:", type(model).__name__)

alloc = torch.cuda.memory_allocated() / 2**30
reserved = torch.cuda.memory_reserved() / 2**30
print(f"显存占用: allocated {alloc:.1f} GiB / reserved {reserved:.1f} GiB")
print("OK — INT8(编码器) + BF16(解码器) 前端就绪")
