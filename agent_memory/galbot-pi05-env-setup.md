---
name: galbot-pi05-env-setup
description: pi0.5 环境准备：Miniforge 已装（TUNA 源）、pi05 env=py3.11；设备是 Orin sm_87 非 Xavier；FlashRT Orin INT8 路线已验证可行
metadata: 
  node_type: memory
  type: project
  originSessionId: 688dacf4-1a8a-4f2b-b926-938022b6de08
  modified: 2026-09-21T09:24:10.558Z
---

用户计划在 G1 机器人（**AGX Orin Developer Kit 64G**，**JetPack 5.1.4 / L4T R35.6.0**（39 个 l4t 包统一 35.6.0-20240828 已核实；用户曾标 5.1.3，NVRM 驱动串报 35.5.0 是 NVIDIA 点版本不 bump 怪癖，勿混淆），CUDA 11.4，r535 驱动（对 11.8/12.x runtime 向后兼容），Ubuntu 20.04 aarch64）上部署 pi0.5。曾误判为 Xavier sm_72（/etc/nv_tegra_release 显示 BOARD: t186ref，不可靠），**已确认是 Orin sm_87（Ampere）**。CUDA 不能升级（会破坏其他 SDK 环境），一切方案必须兼容 CUDA 11.4。

已完成（2026-09-20）：
- Miniforge3 装在 `~/miniforge3`（conda 26.7.2），GitHub 直连慢，安装包走 TUNA 镜像
- `~/.condarc` 已配 TUNA conda-forge 源
- 已建 `pi05` 环境（Python 3.11.16）
- FlashRT 克隆在 `~/holy/FlashRT`，CUTLASS v4.4.2 已放入 `third_party/cutlass`

FlashRT 部署线（主路线）：
- 用户方案：FlashRT（手写 kernel 实时推理引擎）部署量化 pi0.5 + 后续自己做权重量化
- **Orin sm_87 路线已验证可行**：CMakeLists 有专门的 "Jetson Orin SM87 Pi0.5 fast path"——ENABLE_SM80_INT8_CUTLASS 在 sm_87 默认 ON（SM80 家族 CUTLASS INT8 kernel）；FA2 attention 对 sm_80/86/87/89/120/121 启用
- **CUTLASS v4.4.2 SM80 家族最低 CUDA = 11.4**（README 官方表格），与本机 CUDA 11.4 正好匹配；12.4+ 要求只针对 Hopper/Blackwell 路径
- FlashRT cmake 无硬性 CUDA 版本门禁，sm_87 自动检测
- Ampere 上 FP8/NVFP4 均不可用，量化格式走 INT8（校准文档 docs/calibration.md）
- FlashRT 要求 Python ≥3.10，**flashrt env 建成 3.8 装不了，需重建为 3.11**
- **CMake 探针已验证（2026-09-20）**：configure 过（CUDAToolkit 11.4.315 + gcc 9.4 + CUTLASS INT8/FA2/HYVLA_ORIN/SDE 宏全就位），但**编译证明 nvcc 11.4 硬性不够**——`cuda_fp8.h` 是 11.8 头文件（散布 8+ 核心文件）、`__grid_constant__` 是 11.7 特性、源码含真 FP8 intrinsics（sm89 文件），打补丁不可行
- **FlashRT 编译已通过（2026-09-20 14:45）**：conda 11.8 工具链编译成功，产物 `~/holy/FlashRT/flash_rt/flash_rt_fa2.cpython-311-*.so`(53M) + `flash_rt_kernels.cpython-311-*.so`(4.1M)，37 个 cubin 全 sm_87，无缺失依赖；cudart 动态链接（echo 机 ldd 实证：两个 .so 都链 conda env 的 libcudart/cublas 11.8，此前"静态链接"记录有误）。SLIM+SDE+HYVLA 全生效
- **torch 已编译+安装+验证通过（2026-09-21）**：2.4.0a0+gitee1b680（v2.4.1 代码，shallow clone 无 tag 所以版本串带 a0），wheel 123M 在 `~/holy/pytorch/dist/`，已 pip 装进 flash_pyrt311。验证 24/24 matmul 过，bf16/fp16 ≈32 TFLOPS（峰值~42 的 75%），fp32 ≈3.4 TFLOPS。**运行时库解析机制**：`libtorch_cuda.so` 带 `RUNPATH=$ORIGIN:$CONDA_PREFIX/lib`——加载器**先查 conda lib 目录**。用户已把 conda 的 libcublas/libcublasLt 软链到系统 11.4（原 11.8 备份在 `$CONDA_PREFIX/lib/cublas_backup/`），故 torch 实际加载 11.4，24/24 验证过。**注意：将来 conda update 若重装 libcublas 会恢复真 11.8 文件，行为可能翻回**，届时重打软链
- **wheel 可移植性**：只含 sm_87 SASS 无 PTX（Xavier 不可用）；NEEDED 全是 JetPack5.1.x 自带库（cublas/cublasLt/cudart/cufft/curand/cusparse/nvToolsExt 的 .so.11），无 cuDNN 依赖；目标机条件=Orin 家族+JP5.1.x+Python3.11，免编译直接装
- **cuBLAS 现象（三层，重要）**：① python 进程里 `cublasCreate_v2` 作为**第一个 CUDA 调用**会返回 ALLOC_FAILED(3)（纯 C 程序没事、与库版本无关，系统 11.4 也中过）；② torch 进程加载 conda 11.8 cublas 时曾密集失败（2048 matmul 5 连挂）——2026-09-21 用户把 conda cublas 软链到系统 11.4 后 24/24 通过；③ 仍存阵发坏窗口，疑似相机栈(gmsl_service_daemon)锁页内存波动，`ulimit -l`=64MB 偏小是候选因素。**兜底**：`~/holy/cuda_warmup.py`（重试暖场），所有启动入口必加；下次现场失败立刻 strace + pin_memory 探针抓根因
- **onnx 子模块解法**：pytorch 的 check_submodules 是纯文件存在性检查且 onnx 的 CMakeLists 用 `file(GLOB_RECURSE onnx/*.cc)` 收源——GitHub tarball（onnx v1.16.0）解压进 third_party/onnx 即可构建，不需要 git。setup.py:380 的 skip 补丁用户保留未撤销（无害）。onnx/third_party/benchmark、pybind11 空目录不影响（torch 不开 ONNX_BUILD_BENCHMARKS/PYTHON）
- **FlashRT editable 安装完成（2026-09-21）**：`pip install -e ".[torch]"` 成功，flash_rt 0.2.0，INSTALL.md §7 验证全过（flash_rt/torch (8,7)/numpy 1.26.4/kernels 可导入）。**运行环境就绪**。下一步：INT8 权重量化（docs/calibration.md）+ HyVLA Orin 前端（flash_rt/frontends/torch/hyvla_orin.py，编译时已带 FLASHRT_HAVE_HYVLA_ORIN=1）
- 构建命令组合：`-DGPU_ARCH=87 -DFA2_ARCH_NATIVE_ONLY=ON -DFA2_HDIMS='96;128;256' -DFA2_DTYPES='bf16' -DFLASHRT_SLIM_BUILD=ON -DFLASHRT_ENABLE_PI05_SDE=ON -DFLASHRT_ENABLE_HYVLA=ON`；SLIM 裁掉 Motus/Qwen3.6/NVFP4 四族，INT8/FA2/HYVLA 不受影响
- **红线：flash_pyrt311 里装任何带 numpy 依赖的包必须显式钉 `numpy==1.26.4`**（2026-09-21 装 opencv-python-headless 时被 pip 升到 2.4.6，torch 2.4 自编 wheel 是 numpy 1.x ABI，会炸）。兼容组合：`numpy==1.26.4 + opencv-python-headless==4.10.0.84`（OpenCV 5.x 要 numpy≥2，不能用）
- **pi05 权重已就位（2026-09-21，echo 机 `~/holy/models/pi05_lerobot_base`）**：HF `lerobot/pi05_libero_base` 原仓就 5 个文件（model.safetensors 14G/812 张量 + config + 前后处理器 json），权重是 openpi JAX checkpoint 转的（`paligemma_with_expert.*` 命名）。**该 HF 仓不带归一化统计**，processor json 的 `features` 是空的；FlashRT `unnormalize_actions` 是 openpi 分位数反归一化复刻（q01/q99、先 clip 会错），**必须用 openpi 官方 GCS 的 norm_stats.json**（`openpi-assets.storage.googleapis.com/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json`，state 8 维/actions 7 维真 q01/q99，本机可直连），已放 `assets/physical-intelligence/libero/norm_stats.json`（loader 首选路径），`load_norm_stats` 干跑通过。**不要**用 lerobot 数据集 `physical-intelligence/libero` 的 meta/stats.json（只有 min/max，FlashRT 会当 q01/q99 用，官方注释明说 1~5% 漂移）。
- **echo 机端到端推理已跑通（2026-09-21）**：加载 12s（arch=rtx_sm87 自动选中），predict 输出 (10,7)。还差两个随权重之外的文件：`paligemma_tokenizer.model`（4.26MB，GCS `storage.googleapis.com/big_vision/paligemma_tokenizer.model`，HF 的 google/paligemma 是 gated 仓拿不到；慢时用 `curl -C -` 续传）放 `~/.cache/flash_rt/`。脚本在 `~/holy/scripts/`，分子目录：`inference/`（load_pi05.py / load_pi05_int8.py 加载冒烟）、`test/`（bench_pi05.py 延迟基准，用法 `python test/bench_pi05.py bf16|int8_enc|int8_full [n]`；graph_repro.py 抓图复现）、`eval/`（ab_compare_pi05.py 合成 A/B、ab_real_camera.py 真机相机 A/B），根上 verify_deploy.sh 为装机验收入口。
- **INT8 量化（Orin 路线，无需校准产物）**：`precision="int8"` 是 Ascend 专用、其他架构 raise；Orin 走 env 开关（前端 __init__ 时读）：`FVK_PI05_RTX_INT8_ENCODER_ONLY=1` 编码器 INT8+解码器 BF16、`FVK_PI05_RTX_FORCE_INT8=1` 全 INT8、`FVK_PI05_RTX_INT8_VISION=1` 别开（静态版把 cosine 打到 0.282 已永久禁用、动态版未验证）。INT8 权重 scale 加载时按行静态算，激活 scale 运行时动态——`calibrate()` 在 INT8 只用于暖场。**echo 机实测（eager，30 次均值）**：bf16 230.8ms / int8_enc 183.5ms(−20%) / int8_full 161.2ms(−30%)——代码注释预言全 INT8 解码器更慢在 Orin 上不成立（疑似 RTX 结论）。**3 视角（真机三摄，`num_views=3`，键名 image/wrist_image/wrist_image_right）全 INT8 eager 189.5ms（+28ms）**；`cache_frames=2` 在 eager 下无收益（与 graph 路径绑定，文档 12.2Hz 生产档含 graph）。**FlashRT 入口是现成 224×224×3 uint8，解码/缩放在用户相机管线**——GMSL 原始帧近乎免费，JPEG CPU 软解 3 摄可吃 30-90ms 须避开；集成时打点实测解码耗时并做解码∥推理流水；gmsl_service_daemon 锁页波动 + pinned buffer 压力集成时留意 ulimit -l。**数值 A/B 已定档（2026-09-21，`ab_compare_pi05.py`，12 样本种子对齐噪声）**：int8_enc/int8_full 相对 bf16 的未归一化动作余弦 mean 0.99999、min 0.9999+，max_abs_diff ≤0.027，两档数值都安全；**已定 `FVK_PI05_RTX_FORCE_INT8=1`（全 INT8，最快且误差最小）**。A/B 要点：predict 前 `torch.manual_seed` 对齐初始噪声、`cache_frames=1` 排除 KV 复用。**真机相机图 A/B（`ab_real_camera.py`，3 路 SDK 相机只读抓图+cv2 软解 224）**：predict 层控不住噪声（首调用 calibrate 路径额外消耗 RNG，同种子对不齐），必须用 `model.infer(obs, noise=torch.randn(10,32))` 显式固定噪声（32=flow-matching 隐动作维，非输出 7 维；infer 返回 dict 取 ['actions']），同噪声双跑严格 0。真机 5 样本 cos mean 0.954/min 0.827——对照"同图换噪声"的混沌底（两两 cos mean 0.25，范围 −0.42~0.95）远在正常散布内，**INT8 扰动 ≪ 噪声选择自由度，定档结论维持全 INT8**；真机 KPI 应看执行成功率而非单帧 cos。真机负载下延迟 bf16 304ms/int8 234ms（机器人服务并发，比空载高 ~25%）。INT8 显存反而升（8.2/8.5G vs 6.4G）：前端故意把 bf16 原权重留在显存供 pipeline 重建。
- **未解问题：CUDA graph 抓图确定性段错误（已深挖，驱动层缺陷）**：崩点经 gdb 定位为 `cudaGraphInstantiate` → tegra 驱动 `libcuda.so.1`（NVRM 35.5.0）内部段错误，100% 复现。已排除：FlashRT 抓图封装（ctypes relaxed 模式最小复现通过，`~/holy/scripts/test/graph_repro.py`）、cudart 版本（11.4/11.8 都崩）、多 runtime 实例（进程内单实例）、上游代码（本机 HEAD=origin HEAD 无修复）。torch 小图抓取正常 → 与 pi05 全流水线图的**内容/规模**相关；**官方 `examples/orin/bench_pi05.py` 同样崩**（可作上游 issue 复现器）。绕法 `use_cuda_graph=False`（脚本 monkeypatch，`PI05_NO_GRAPH=1`）；eager 全 INT8 已 161ms，graph 收益非数量级。可向 FlashRT 上游报 issue。
- 构建提速：FA2 的 CUTLASS 3.x 模板占大头编译时间，可开 `FA2_ARCH_NATIVE_ONLY=ON` 省 ~66%
- 备用路线（未走）：Jetson-PI provider（llama.cpp/GGML fork，FindJetsonPI.cmake）

待办与坑：
- **环境变量泄漏三件套**（bashrc 全局 export，跑 pi0.5/编译前必须 unset）：
  - `PYTHONPATH=/data/galbot/lib`（重复 export 3 次）
  - `LD_LIBRARY_PATH=/data/galbot/lib:/usr/local/cuda-11.4/lib64`（bashrc:124）——**最毒**：SDK 自带旧 libcurl.so.4.8.0 会劫持系统 curl/conda 的 HTTPS（CA 包坏，报 certificate problem），已致"网络不通"误诊；前辈已在 bashrc:139 写了 `curl()` wrapper 隔离，说明此坑早有前科
  - `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`（SDK+mujoco TLS 溢出 workaround）
  - 网络/镜像本身完全正常：TUNA 和 USTC 实测均 <0.5s，v4 出口正常，DNS 正常
- openpi（训练侧）在 CUDA 11.4 上仍需 JetPack 5 专用 jax/torch wheel
- Docker 有 nvidia-docker2 但 daemon.json 未注册 nvidia runtime，`--gpus all` 不可用
- 相关：[[galbot-g1-digital-twin-setup]]
