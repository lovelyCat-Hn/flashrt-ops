---
name: galbot-g1-cuda-graph-instantiate-fix
description: G1 Jetson CUDA graph 段错误根因：L4T r35.6 iGPU 旧式 cudaGraphInstantiate 必崩，须走 WithFlags；已修复并翻转 PI05_NO_GRAPH
metadata: 
  node_type: memory
  type: project
  originSessionId: 944554af-5725-4cba-af39-0c03a553a954
  modified: 2026-09-24T02:26:48.220Z
---

G1 Jetson（L4T r35.6，iGPU）"动态图无法加载"破案实录（2026-09-24）：

- **根因**：旧式 `cudaGraphInstantiate` 符号在该机 tegra 驱动里对**任意图**必段错误（连单节点 memset 都崩，gdb 确认崩在 libcuda.so.1），与 libcudart 版本无关（conda 11.8.89 与系统 11.4.298 都崩）；`cudaGraphInstantiateWithFlags` 入口正常。torch 原生抓图能过是因为 torch 内部调的就是 WithFlags。
- **修复**：FlashRT `flash_rt/core/cuda_graph.py`（提交 4822d755）改走 WithFlags，argtypes 显式声明（aarch64 上 ctypes 把 int 当 32 位传 64 位 flags 参数也是隐患），<11.4 回退旧符号。AMD 孪生 hip_graph.py 早已是 WithFlags 写法。
- **教训**：脚本注释当时误记为"cudaStreamEndCapture 段错误/r35.5"，实为 Instantiate 段/r35.6——崩溃现场要 gdb 拿第一手，别信二手注释。
- **现状**：所有推理脚本 `PI05_NO_GRAPH` 默认已翻转 1→0（ops d73282f，已推）；回退 = 前缀 `PI05_NO_GRAPH=1`。bench 实测收益约 3%（int8_full 189.2 vs 195.2ms；bf16 261.5 vs 268.1ms）——流水线 GPU-bound，派发开销小，**别期待图带来翻倍加速**。replay 已挂计数器确认稳态真实命中。
- **待办**：真机闭环（run_g1_loop）尚未用图模式验证，[[galbot-flashrt-env-migration-pack]] 部署的新机如遇 graph 崩先查驱动版本是否 r35.x iGPU。
