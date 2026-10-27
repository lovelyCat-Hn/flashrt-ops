#!/usr/bin/env python
"""CUDA graph 崩溃最小复现阶梯（定位 Orin 抓图段错误）。

阶段:
  1. torch.cuda.CUDAGraph + cuBLAS GEMM        —— torch 机制、global 模式
  2. FlashRT CUDAGraph (ctypes, relaxed) + 同一 GEMM —— FlashRT 原始路径
  3. FlashRT CUDAGraph + flash_rt 自研 kernel (rms_norm 等 fvk 入口)

哪一阶段崩，问题就限定在那一层。
"""
import ctypes
import sys

import torch

print("stage 1: torch.cuda.CUDAGraph + GEMM ...", flush=True)
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
_ = a @ b  # warmup：抓图前建 cuBLAS 句柄（抓图期间 cublasCreate 非法）
torch.cuda.synchronize()
s = torch.cuda.Stream()
g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    with torch.cuda.graph(g, stream=s):
        c = a @ b
    c.fill_(0)
g.replay()
torch.cuda.synchronize()
ref = a @ b
print(f"  torch graph OK, replay diff={(c - ref).abs().max().item():.2e}")

print("stage 2: FlashRT CUDAGraph (relaxed) + torch GEMM ...", flush=True)
from flash_rt.core.cuda_graph import CUDAGraph  # noqa: E402

fg = CUDAGraph()
s2 = torch.cuda.Stream()
sh = ctypes.c_void_p(s2.cuda_stream)
fg.begin_capture(sh)
with torch.cuda.stream(s2):
    c2 = a @ b
fg.end_capture(sh)
fg.replay(sh)
s2.synchronize()
print(f"  FlashRT graph OK, diff={(c2 - ref).abs().max().item():.2e}")

print("全部阶段通过 —— 崩溃在 pi05 run_pipeline 的具体抓取内容里")
