#!/usr/bin/env python
"""pi0.5 量化档位数值一致性 A/B：bf16 vs int8_enc vs int8_full。

方法:
- 三档各自独立加载（同进程顺序跑，档位间 del + empty_cache）
- 每档跑 N 个样本：种子随机噪声图像 x2 视角 + 随机 state + 轮换 prompt
- 每次 predict 前 torch.manual_seed(固定种子)——两档吃同一份初始噪声，
  解码为确定性 ODE，输出差异即纯量化误差
- cache_frames=1：每帧跑全量流水线，排除 temporal KV 复用干扰对比
- 无 CUDA graph（本机抓图崩溃，见 bench_pi05.py 说明）

判读参考（未归一化动作余弦）:
  mean>=0.995 且 min>=0.98 → 可用；0.95~0.99 → 谨慎；<0.95 → 该档位不可用

用法:
  ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/ab_compare_pi05.py [N]
"""
import os
import sys

import numpy as np
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEED_BASE = 20260921

# 无 graph（本机抓图崩溃绕过）
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402

CKPT = "/home/galbot/holy/models/pi05_lerobot_base"
PROMPTS = [
    "pick up the black bowl on the stove and place it on the plate",
    "open the top drawer of the cabinet",
    "put the red mug on the table",
]

MODES = {"bf16": {}, "int8_enc": {}, "int8_full": {}}

# ── 生成固定测试观测（与模式无关，先于加载生成） ──
samples = []
for i in range(N):
    rs = np.random.RandomState(SEED_BASE + i)
    imgs = {
        "image": rs.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "wrist_image": rs.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    }
    state = rs.uniform(-1.5, 1.5, (8,)).astype(np.float32)
    samples.append((imgs, PROMPTS[i % len(PROMPTS)], state))

acts = {}
for mode, flags in MODES.items():
    for k, v in flags.items():
        os.environ[k] = v
    model = flash_rt.load_model(CKPT, config="pi05", cache_frames=1)
    out = []
    for i, (imgs, prompt, state) in enumerate(samples):
        torch.manual_seed(SEED_BASE + i)          # 同噪声跨档对齐
        a = model.predict(imgs, prompt=prompt, state=state)
        out.append(a.astype(np.float32))
    acts[mode] = np.stack(out)
    np.save(f"/tmp/ab_acts_{mode}.npy", acts[mode])
    del model
    torch.cuda.empty_cache()
    print(f"[{mode}] {N} 样本完成")

# ── 对比 ──
ref = acts["bf16"]
print("\n===== A/B 结果（相对 bf16，未归一化动作） =====")
print(f"{'样本':>4} {'int8_enc cos':>14} {'int8_full cos':>14}")
cos_enc, cos_full = [], []
for i in range(N):
    a, b, c = ref[i].ravel(), acts["int8_enc"][i].ravel(), acts["int8_full"][i].ravel()
    ce = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cf = float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-12))
    cos_enc.append(ce)
    cos_full.append(cf)
    print(f"{i:>4} {ce:>14.5f} {cf:>14.5f}")

for name, cs in (("int8_enc", cos_enc), ("int8_full", cos_full)):
    d = np.abs(acts[name] - ref)
    print(f"\n{name}: cos mean {np.mean(cs):.5f} | min {np.min(cs):.5f} | "
          f"max_abs_diff {d.max():.5f} | mean_abs_diff {d.mean():.5f}")
