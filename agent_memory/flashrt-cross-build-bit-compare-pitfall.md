---
name: flashrt-cross-build-bit-compare-pitfall
description: FlashRT pi0.5 跨构造位等不可比——每次构造重跑 GEMM autotune 选不同算法；位等断言只在同构造内成立
metadata:
  node_type: memory
  type: project
  originSessionId: b40f9b71-0573-41af-b5b3-38d6bf6678ff
  modified: 2026-10-09T01:48:20.340Z
---

FlashRT pi0.5 RTX（Orin sm_87）引擎**每次前端构造都重跑 GEMM autotune**，按计时选算法，两次构造会选到不同 CUTLASS kernel → 浮点累加顺序不同 → 输出位不同。

2026-10-09 实测（pi05_g1_onlypick_deploy，seed-7 噪声，bf16 无图）：
- 同构造重复推理：逐位相等（构造内确定）
- 跨构造 plain vs plain：max 0.347 / mean 0.011（未归一化动作空间，16 维）
- 跨构造 plain vs rtc-off：与上一行**数值完全相同** → rtc=True 零额外贡献

**Why:** RTC off 路径"逐位不变"测试首版写成跨构造对照位等，连挂三次全是这个坑，不是泄漏。

**How to apply:**
- 任何位等（bit-identity）断言/探针必须在**同构造内**做（同一 pipeline 实例跑两态，如 rtc_prefix=None vs =prefix）；
- 跨构造只能 allclose（未归一化空间容差 ≥0.5）或比 cos（[[galbot-pi05-env-setup]] 的量化消融用 cos 正是这个原因）；
- autotune 日志（"tested N algos, best=k"）每次构造都会重打，best 序号不同即算法不同；
- 生产脚本进程内只构造一次，不受影响。
