# W8A16-decoder kernel patch（sm_87 手写 weight-only INT8 GEMM）

对应本地 FlashRT 提交 `8f08ca85`（2026-10-07），7 文件：

- `csrc/gemm/w8a16_skinny_sm87.cuh` / `.cu` — 核心kernel（mma m16n8k16 +
  cp.async 双缓冲 + 寄存器内 int8→bf16 反量化 + K-split 确定性 reduce）
- `csrc/bindings.cpp` — `w8a16_skinny_gemm` 绑定（variant/split_k 可选）
- `CMakeLists.txt` — 新 .cu 加入编译
- `benchmarks/w8a16_skinny_sm87/main.cu` — 微基準（正确性 vs fp64 + 5 shape 扫描）
- `flash_rt/models/pi05/pipeline_rtx.py`、`flash_rt/frontends/torch/pi05_rtx.py`
  （这两个 py 同时也在上级 `hotfix_flashrt/flash_rt/` 里，apply_hotfix.sh 会覆盖）

## 应用

```bash
cd ~/holy/FlashRT
git am ~/holy/hotfix_flashrt/flashrt_kernel_patch/0001-*.patch
```

（apply_hotfix.sh 会在检测到 git 仓库时自动尝试；已应用则跳过。）

## 重建 .so（必须！py 补丁单独不足以启用 W8A16）

⚠ 绑定在 .so 里。老 .so + 新 py = flag 默认关、行为不变（安全）；但要真正启用
`FVK_PI05_RTX_W8A16_DECODER=1` 必须重编：

```bash
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
cd ~/holy/FlashRT && rm -rf build
~/miniforge3/envs/flash_pyrt311/bin/cmake -B build -S . \
    -DCMAKE_CUDA_COMPILER=$HOME/miniforge3/envs/flash_pyrt311/bin/nvcc \
    -DPython3_EXECUTABLE=$HOME/miniforge3/envs/flash_pyrt311/bin/python \
    -DCMAKE_BUILD_TYPE=Release
~/miniforge3/envs/flash_pyrt311/bin/cmake --build build -j 8
```

⚠ 必须用 flash_pyrt311 环境的 nvcc（11.8）+ cmake（3.29）——系统 nvcc 11.4 缺
cuda_fp8.h/新 cublasLt 枚举，且其 targets include 路径抢占用户 -isystem，
别在系统工具链上耗（本机 10-07 实录）。

## 启用

```bash
export FVK_PI05_RTX_W8A16_DECODER=1   # 权重-only INT8：激活保持 bf16
```

与 `FVK_PI05_RTX_FORCE_INT8_DECODER`（A8W8）互斥（同设会 raise）；与
`FVK_PI05_RTX_FP8_DECODER` 冲突时自动关 fp8 并告警；chunk_size>64 拒绝。

## 实测（本机 echo 机 2026-10-07，闭环 rot 探针）

| 配置 | p50 | 质量 |
|---|---|---|
| 基线 bf16 chunk=50 | 283ms | — |
| W8A16 chunk=50 | 277/278ms（−5-6ms） | cos_arm 1.0000，grip bias 0.05pp |
| 基线 chunk=10 | 283ms | — |
| W8A16 chunk=10 | 273ms（−10ms） | cos_arm 0.9997，cos_grip 0.9978 |

微基准：M=10 每层 GEMM 和 261→133µs（−49%），M=50 266→214µs；最差 relF =
1 bf16 ULP（多种子验证为输出舍入边界噪声）。NO_GRAPH 模式下 e2e 只兑现约一半
增量（reduce 多发射 + DRAM 争用），graph-on 部署预期更接近 GEMM 增量。
