# 装机与 FlashRT 使用手册

**适用**：已完成 `flashrt_bundle_v2.tar` 部署的 Orin（AGX Orin 64G / JetPack 5.1.x）
**配套文档**：部署迁移见 `DEPLOY.md`，实测数据见 `BENCHMARKS.md`，本篇是日常操作手册

---

## 1. 机器上有什么（目录布局）

| 路径 | 内容 |
|---|---|
| `~/miniforge3/envs/flash_pyrt311/` | Python 3.11.16 运行环境（torch 2.4.0a0 自编 wheel + FlashRT 0.2.0 + numpy 1.26.4 + opencv 4.10，**全部编译产物已内置，零编译**） |
| `~/holy/FlashRT/` | FlashRT 源码树（editable 安装指向这里；含已编译 fa2/kernels .so） |
| `~/holy/models/pi05_lerobot_base/` | pi0.5 权重 14.5G + `assets/.../norm_stats.json`（openpi 官方 q01/q99 统计，勿删） |
| `~/.cache/flash_rt/paligemma_tokenizer.model` | PaliGemma 分词器（4.26MB，勿删） |
| `~/holy/scripts/` | 脚本（`inference/` 推理冒烟、`test/` 性能与诊断、`eval/` 精度评估，根上 `verify_deploy.sh` 为装机验收入口） |
| `~/holy/cuda_warmup.py` | cuBLAS 暖场兜底（生产启动入口必加） |
| `~/holy/{DEPLOY,USAGE,BENCHMARKS}.md` | 三份文档 |

## 2. 环境使用铁律（先读这节）

```bash
# FlashRT 专用 python（本机交互 shell 没有 conda 命令，也不要去配 conda init）
alias flashpy=/home/galbot/miniforge3/envs/flash_pyrt311/bin/python
```

环境变量三态规则（`/data/galbot/lib` 是机器人 SDK 的库目录）：

| 场景 | 启动前 |
|---|---|
| 纯 FlashRT 推理/基准 | `unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD` |
| FlashRT + GalbotSDK 同进程 | `export LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib` |
| 什么都不做（新 shell 默认） | bashrc 会带上 SDK 路径 → 纯推理也能跑，只是不干净，推荐显式 unset |

## 3. 常用任务

### 3.1 推理冒烟（~30s，`scripts/inference/`）

```bash
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
flashpy ~/holy/scripts/inference/load_pi05.py          # BF16 加载验证
flashpy ~/holy/scripts/inference/load_pi05_int8.py     # INT8 档加载验证
# 期望末行: OK — 权重加载链路（safetensors + norm_stats + 前端）全部就绪
```

**真机相机推理（带实时性打点，只读不执行）**——SDK 三路取图 → 推理 → 动作打印，
耗时分三层（纯推理 / 取图解码 / 端到端节拍+动作步率），自动对照控制频率窗口
（默认 50Hz）判定能否跟上，并与 echo 机基线比较：

```bash
LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
flashpy ~/holy/scripts/inference/run_pi05_camera.py \
    [--prompt "..."] [--rounds 10] [--hold 10] [--views 3] [--tier int8_full] [--ctrl-hz 50]
# 前提: 机器人已上电、相机服务在跑；echo 机实测: 取图解码 ~20ms，3视角全INT8 端到端 ~248ms/轮
```

### 3.2 延迟基准（换机/换驱动后必跑，`scripts/test/`）

```bash
flashpy ~/holy/scripts/test/bench_pi05.py int8_full 30     # 档位: bf16 | int8_enc | int8_full
PI05_VIEWS=3 flashpy ~/holy/scripts/test/bench_pi05.py int8_full 30   # 三视角（真机配置）
```

INT8 档位语义（环境变量，脚本内已设）：
- `FVK_PI05_RTX_FORCE_INT8=1` **全 INT8——已定档**，最快且数值最稳
- `FVK_PI05_RTX_INT8_ENCODER_ONLY=1` 保守档（编码器 INT8 + 解码器 BF16）
- `FVK_PI05_RTX_INT8_VISION=1` **禁用**（静态版把余弦打到 0.28，已永久关闭；动态版未验证）

### 3.3 量化精度 A/B（`scripts/eval/`）

```bash
flashpy ~/holy/scripts/eval/ab_compare_pi05.py 12      # 合成噪声图，12 样本
flashpy ~/holy/scripts/eval/ab_real_camera.py 5        # 真机相机（只读抓图，不执行任何动作）
# 前提: 相机服务在跑 + 机器人已上电（吃过"没上电"的亏）；拼图在 /tmp/cam_samples.png
```

判读：cos ≥0.99 完全一致；真机图看 **0.8 哨兵线**——动作头混沌底本身就是 0.25（换噪声即变），0.95 级别=无损。

### 3.4 graph 探测（换驱动后跑一次）

```bash
bash ~/holy/scripts/verify_deploy.sh    # 第 ⑤ 项自动探；或 flashpy ~/holy/scripts/test/graph_repro.py
# 崩(段错误) = 驱动缺陷仍在，保持 PI05_NO_GRAPH=1；通过 = 可开 graph 拿更快延迟
```

所有脚本默认 `PI05_NO_GRAPH=1`（绕过 r35.5 驱动 `cudaGraphInstantiate` 段错误，详见 BENCHMARKS.md 第四节）。

### 3.5 G1 16 维微调权重接入（微调完成后三步，2026-09-22 全链路已验证）

> 新机先用：`bash ~/holy/hotfix_flashrt/apply_hotfix.sh`（bundle 里的 FlashRT 源码
> 缺两个本地补丁；verify_deploy.sh ③ 段会自动探测并提示）。

```bash
# ① 装配 + 预检（微调输出目录 → FlashRT 部署目录，软链不复制 14G）
flashpy ~/holy/scripts/inference/g1_ckpt_prep.py \
    --src <微调输出目录> --out ~/holy/models/pi05_g1_ft \
    --stats <训练用的 stats.json> --mode mean_std   # 语义务必与训练配置一致！
# ② 对齐校验（数学往返 + 引擎语义分发实证 + 落域）
flashpy ~/holy/scripts/eval/norm_align_check.py --ckpt ~/holy/models/pi05_g1_ft
# ③ 真机推理（只读不执行；manifest 自动配置，零参数）
LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
flashpy ~/holy/scripts/inference/run_g1_inference.py --ckpt ~/holy/models/pi05_g1_ft
```

要点（详见 BENCHMARKS.md G1 节与记忆）：
- **pi0.5 原生动作隐空间就是 32 维**，16/7 维只是消费侧切片——权重不用改，
  `load_model(action_dim=16)` + config 声明即可（FlashRT 本地补丁 6425fe8d+）
- **归一化语义**：norm_stats.json 顶层 `"norm_mode"` 标记分发——`q01_q99`（openpi
  分位，默认/兼容旧行为）或 `mean_std`（lerobot MEAN_STD）。**选错 = 动作系统性
  畸变**，lerobot 管线微调默认 MEAN_STD，openpi 默认分位
- **state 归一化是调用方责任**：FlashRT 不归一化 state，直接把原始关节值喂进去
  会被 256-bin 离散化打满。必须 `normalize_state(raw, norm_stats)` 后再传 `state=`
  （`run_g1_inference.py` 已内置）
- **state 23 维布局**（与 pick_place_balence 逐维对齐，2026-09-22 真机逐维核验）：
  `[l_arm×7, l_grip, r_arm×7, r_grip, leg_joint1-5, head_joint1-2]`；
  夹爪数据集是 0~100%，SDK 读数是开口宽度（米），标定后经 manifest 换算
- 夹爪标定：`--grip-wmin/--grip-wmax`（满/零开度 SDK 宽度），写入 manifest

## 4. 与 GalbotSDK 联用（同进程，已验证）

```python
# 启动时: export LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib
import sys; sys.path.insert(0, "/data/galbot/lib")
from galbot_sdk.g1 import GalbotRobot, SensorType

robot = GalbotRobot()
robot.init({SensorType.HEAD_LEFT_CAMERA, SensorType.LEFT_ARM_CAMERA,
            SensorType.RIGHT_ARM_CAMERA})      # 只开传感器，不碰运动
# ... robot.get_rgb_data(SensorType.X) → dict["data"] 是压缩图，cv2.imdecode 软解
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()   # 收尾三件套
```

- 相机映射建议：HEAD_LEFT→`image`，LEFT_ARM→`wrist_image`，RIGHT_ARM→`wrist_image_right`；
  **G1 数据集（pick_place_balence）主相机是 head_right**，微调部署用
  `run_g1_inference.py` 的 manifest 映射（HEAD_RIGHT→`image`）
- 读关节**一律显式名字模式** `get_joint_positions([], names)`（group 模式返回顺序实测乱序）；
  `get_joint_names()` 全量 23 个 = leg5 + head2 + 双臂14 + 双夹爪2
- 参考 `~/workspace/GalbotSDK-1.7.1/examples/g1/python/`（注意运行时是 1.8.1）；`tutorials/example6_execute_vla.py` 是 VLA 集成骨架
- ⚠️ pi0.5 输出 (10,7) 是 LIBERO 臂动作空间，与 G1 臂**不是恒等映射**——先开环观察，勿直接闭环执行

## 5. 模型相关速查

| 事项 | 结论 |
|---|---|
| INT8 校准 | 不需要（权重静态按行 + 激活动态逐行；FP8 那套校准是 Thor 专用） |
| ODE 步数 | `load_model(num_steps=N)`，**load 时定**，延迟约线性、精度换速度 |
| 动作块长度 | 固定 10（训练配置），不可改；要覆盖 50 步 = 跑 5 次（真机流水执行） |
| 跨调用状态 | 每次从全新噪声采样；连续性靠新观测（`state=` 每帧喂）+ prompt/KV 缓存 |
| 固定噪声复现 | `model.infer(obs, noise=torch.randn(10,32))`，返回 dict 取 `["actions"]` |
| 动作输出维 | `load_model(action_dim=N)`（默认 7；G1 微调 16 维走 manifest 自动传） |
| 归一化语义 | norm_stats.json 顶层 `norm_mode`：`q01_q99`（默认）/`mean_std`；`normalize_state`/`unnormalize_actions` 按此分发 |
| state 输入 | **必须调用方归一化**（`normalize_state`），FlashRT 只做 256-bin 离散化进 prompt |
| 上游更新 | 纯 Python 改动 `git -C ~/holy/FlashRT pull` 即生效；碰 CUDA 源码要重编（11.8 工具链）。本地有未推上游补丁（action_dim/norm_mode，见 git log） |

## 6. 红线与禁忌

1. **装任何包必须钉 `numpy==1.26.4`**（pip 会偷偷升到 2.x 炸掉 torch ABI）；OpenCV 用 `==4.10.0.84`
2. **JetPack 6 / CUDA 12 上不可用此包**（无 .so.11 运行库）
3. 不跑 `~/miniforge3/bin/conda init`（与本机 miniconda 打架）
4. 不开 `FVK_PI05_RTX_INT8_VISION=1`
5. 生产启动入口加 `cuda_warmup()`（cuBLAS 阵发坏窗口兜底，见 `~/holy/cuda_warmup.py` 头注释）
6. norm_stats.json 和 tokenizer.model 是踩坑补出来的，**删了模型就加载不了**
7. **state 必须归一化后再喂**（`normalize_state`）——原始关节值直接喂会被离散化打满 bin，静默劣化
8. **norm_mode 语义必须与训练配置一致**——lerobot 默认 MEAN_STD、openpi 默认分位，接权重前先跑 `norm_align_check.py`
