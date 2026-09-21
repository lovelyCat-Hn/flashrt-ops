# FlashRT pi0.5 Orin 部署基准报告

**日期**：2026-09-21 ｜ **测试机**：galbot-echo（Jetson AGX Orin Developer Kit 64G）
**用途**：跨平台 wheel 迁移验证 + 量化精度/速度定档（本机为测试台，非抓取真机）

## 测试环境

| 项 | 值 |
|---|---|
| 系统 | JetPack 5.1.4 / L4T R35.6.0 / CUDA 11.4 / 驱动 NVRM 35.5.0 |
| 推理引擎 | FlashRT 0.2.0（手写 kernel，arch=rtx_sm87 自动选中） |
| PyTorch | 2.4.0a0+gitee1b680（JP5 专用自编 wheel，bf16 实测 ~32 TFLOPS） |
| 模型 | lerobot/pi05_libero_base（openpi 权重转 HF 格式，812 张量 / 14.5 GB safetensors） |
| 运行方式 | **eager**（CUDA graph 因驱动缺陷禁用，见第五节） |
| 量化 | FlashRT INT8 W8A8：权重按行静态 absmax/127，激活运行时动态逐行，**无需校准数据集** |

## 一、量化档位延迟对比（2 视角，合成零图，30 次稳态均值）

| 档位 | 延迟 | vs BF16 | 显存 allocated |
|---|---:|---:|---:|
| BF16（基线） | 230.8 ms | — | 6.4 GiB |
| 编码器 INT8 + 解码器 BF16（`INT8_ENCODER_ONLY=1`） | 183.5 ms | **−20.5%** | 8.2 GiB |
| 全 INT8（`FORCE_INT8=1`） | **161.2 ms** | **−30.2%** | 8.5 GiB |

- 开关为环境变量（Orin 路线不走 `precision=` 参数，那是 Ascend 专用）
- **显存不降反升**：前端有意保留 bf16 原权重供 pipeline 重建，量化收益在速度
- 代码注释称"解码器 M=10 下全 INT8 更慢"，在 Orin 上**实测不成立**（全 INT8 反而最快，该结论疑似来自 RTX）

## 二、3 视角与真机负载（全 INT8）

| 场景 | 单次延迟 | 50 步动作覆盖（5 次调用累计） |
|---|---:|---:|
| 3 视角，空载 | 189.5 ms（+17.6%） | ≈ 0.95 s |
| 3 视角 + `cache_frames=2`，空载 | 190.4 ms（**eager 下无收益**，该优化与 graph 路径绑定） | — |
| 3 视角，真机服务并发 | 234 ms | ≈ 1.17 s |
| 3 视角 BF16，真机服务并发 | 304 ms | ≈ 1.45 s |

- 第 3 视角代价 +28 ms（SigLIP patch 数 +1/3、prefix token 增多）
- 单次推理输出 **10 步动作块**（action horizon=10，训练配置固定）；ODE 步数 `num_steps=10` 可调、延迟约线性
- 控制回路可行性：50 Hz（执行 10 步=200 ms）下 3 视角 234 ms **追不上**，需降至 40 Hz / 2 视角 / 修复 graph

## 三、数值一致性验证

**① 合成输入 A/B**（12 样本，随机噪声图，同图同噪声，相对 BF16 的未归一化动作余弦）：

| 档位 | cos 均值 | cos 最小 | 最大绝对误差 |
|---|---:|---:|---:|
| 编码器 INT8 | 0.99999 | 0.99991 | 0.027 |
| 全 INT8 | 0.99999 | 0.99994 | 0.016 |

**② 真机相机图 A/B**（5 样本，HEAD_LEFT/LEFT_ARM/RIGHT_ARM 三路实拍，软解 224，同图同噪声）：

- cos 均值 **0.954**，最低 0.827
- **判读基线（关键）**：同一张图、仅更换初始噪声，动作两两余弦 **均值 0.25**（−0.42 ~ 0.95）——flow-matching 动作头对初始噪声高度敏感，属模型固有混沌性
- 结论：INT8 扰动 ≪ 噪声选择自由度，真机图上的 0.954 实为"同一样本的微小变体"，**量化无损结论成立**
- 注：混沌底实测跑在 INT8 档（单模型加载），混沌为模型属性，不影响判读

**③ 真机部署验收 KPI 建议**：任务执行成功率；单帧余弦仅作哨兵指标（阈值可设 0.8，低于则报警复测）。

## 四、CUDA graph 问题（未解，已定位）

- 现象：首次 predict 的 graph 抓取 **100% 段错误**（确定性复现）
- 定位（gdb 原生栈）：`cudaGraphInstantiate` → tegra 驱动 `libcuda.so.1`（NVRM 35.5.0）内部
- 已排除：FlashRT 抓图封装（ctypes relaxed 最小复现通过）、cudart 11.4/11.8 版本（都崩）、多 runtime 实例、代码版本（上游最新 HEAD）
- 旁证：官方 `examples/orin/bench_pi05.py` 同样崩；torch 小图抓取正常 → 大图/特定节点内容触发驱动缺陷
- 绕过：`use_cuda_graph=False`（脚本内置）。**新机驱动不同，值得用 verify_deploy.sh 第 ⑤ 项重探**；官方文档带 graph 的生产档参考值为 12.2 Hz（cache_frames=2）

## 五、外部对照（引至 FlashRT 官方 docs/benchmark_comparison.md，非本机实测）

| 配置 | 硬件 | 延迟 |
|---|---|---:|
| OpenPI 参考实现（上游，3 视角） | Jetson AGX Thor | 714 ms（1.4 Hz） |
| FlashRT NVFP4（3 视角） | Jetson AGX Thor | 51.5 ms → **13.9×** |
| OpenPI 参考实现 | RTX 5090 | 244 ms（4.1 Hz） |

本机同板 OpenPI 参考实现未测（JP5.1.4 上 JAX 依赖受限）；按官方跨板数据与 Orin eager PyTorch 通行水平估计，FlashRT INT8 相对参考实现约有数倍到一个数量级优势。

## 六、方法与复现

- 延迟：首次调用（建 pipeline）不计入，稳态 30 次；合成输入为零图（视觉计算量与内容无关，耗时有代表性）
- 精度 A/B 三要素：同图缓存、同初始噪声、`cache_frames=1`（排除时序 KV 复用）
- 噪声对齐必须走 `model.infer(obs, noise=torch.randn(10,32))` 显式传（`torch.manual_seed` 控不住：首次调用走 calibrate 路径额外消耗 RNG）；`infer` 返回 dict，动作取 `["actions"]`
- 复现脚本（`~/holy/scripts/`，分 `inference/` 推理冒烟、`test/` 性能诊断、`eval/` 精度评估）：`test/bench_pi05.py`（延迟三档）、`eval/ab_compare_pi05.py`（合成 A/B）、`eval/ab_real_camera.py`（真机相机 A/B，只读不执行）、`test/graph_repro.py`（graph 最小复现）、`verify_deploy.sh`（部署验收，根目录）
