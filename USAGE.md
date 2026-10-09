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
| `~/holy/scripts/` | 脚本（`inference/` 直接可跑入口 + `g1_config.py` 共享配置、`g1/` 整备与标定、`dataset/` 数据集工具、`eval/` 精度评估、`test/` 性能与诊断、`probes/` 一次性探针与封存件，根上 `verify_deploy.sh` 为装机验收入口） |
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
flashpy ~/holy/scripts/g1/g1_ckpt_prep.py \
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
- **state 23 维布局**（**右臂在前**——2026-09-30 only_place 运动时间线探针 +
  代码 docstring/消费端双重确认；旧版此处误写 l_arm 在前，记反会把臂甩背后。
  两数据集同构）：
  `[r_arm×7, r_grip, l_arm×7, l_grip, leg_joint1-5, head_joint1-2]`；
  夹爪数据集是 0~100（33=持物起步），SDK 读数是开口宽度（米），标定后经 manifest 换算
- 夹爪标定：`--grip-wmin/--grip-wmax`（满/零开度 SDK 宽度），写入 manifest

### 3.6 G1 闭环执行（run_g1_loop，2026-10-07 工作点 v3 定案）

```bash
# 前提: 3.5 三步全绿 + warmup 已跑；急停在手边
~/holy/run.sh ~/holy/scripts/inference/run_g1_loop.py --exec --rounds 60

# 工作点已写进 config/g1.toml [loop]（n_action_steps=25 / 合步 25 / pace 0.35 /
# div 1.5 / near-div 2 / near_gap 0.06 / speed 1.0 / delta_max 0.3），无需 CLI 传参。
# 速度律: 臂速÷数据集原速 = n/(30×pace×div)，当前组合 ≈1.59× 原速；
# R 落盒实测 10.0s（数据集 13s，空爪纪录）。
# v3 前置 --nav-suspend（导航栈冻结→推理 329±7ms，水位≈0.34-0.35）。
# pace 0.35 贴水位由 hold 兜底（推理迟到=hold 最后目标单调收敛，零深尾垃圾；
# --late-action stale=旧 chunk 深尾续航，原 A 脚本行为，勿用于破水位档，
# 仅 A/B 对照实验用）。参数命名对照 lerobot 术语: docs/lerobot-alignment.md
# 图模式默认开（WithFlags 热修）；异常回退: 前缀 PI05_NO_GRAPH=1
```

任务时间线（only_place 101 轨中位，**双臂严格串行**，0/101 并行）：
左臂启动 0.9s → 左爪释放 3.7s → 右臂启动 7.2s（L 释放后 +3.4s）→ 全程 13.5s。
实测（2026-09-30 60 轮判别跑）：L 释放 12-15 轮、L 回闭 19、R 释放 37-39、
R 回闭 51，全程 35.9s → **双臂预算 52-60 轮**（≈数据集时间线 2.76×）。

- ⚠️ **q 停纪律：右爪落盒才停**——左爪落盒时右臂还没开始（严格串行）
- ⚠️ **轮数给足**：30 轮只够左臂半程（9/30 实证：恰在 R 启动窗边缘截断，无判别力）
- 落盒前是"梦游期"，日志零判别力，勿据前几轮判成败
- **全文日志自动落盘** `logs/loop_<时间戳>.log`（--log-file 可改路径），判读直接发文件免复制
- 异常：Ctrl-C / 急停；graph 异常加前缀 `PI05_NO_GRAPH=1` 回退 eager

### 3.7 native 固定节拍执行器（run_g1_loop_native，2026-10-09 新增）

v3 的 A/B 孪生：删配速律家族（pace/div/合步/hold 全套），改为 lerobot 式固定节拍
——tick=1/30s 消费 1 条 action，速度由 Δaction/tick 涌现 ≈**1.0× 数据集原速**；
消费到水位（n_action_steps=25/50）触发新推理，新块落地**整块换入弃尾**。
参数命名与语义对照 lerobot：docs/lerobot-alignment.md §九。

```bash
# 冒烟首跑（15fps + 0.5× 速度天花板，1-2 块看节奏再上 30fps）
~/holy/run.sh ~/holy/scripts/inference/run_g1_loop_native.py --exec --nav-suspend \
    --rounds 2 --fps 15 --speed 0.5

# Part D 真机三态 A/B（各 60 轮；native 侧用 10-09 定案工作点 6拍/2窗）
# ① 基线 = v3 配速律（上方 3.6 原命令）
~/holy/run.sh ~/holy/scripts/inference/run_g1_loop.py --exec --nav-suspend --rounds 60
# ② native 裸跑（≈1.0× 原速；星饿 0 窗；tick 过冲 p95<10ms；接缝对照主指标）
~/holy/run.sh ~/holy/scripts/inference/run_g1_loop_native.py --exec --nav-suspend \
    --rounds 60 --cmd-every 6 --arrive-div 2
# ③ native+RTC 前缀引导（同 ② + 引导；预期接缝 < 块内均值）
~/holy/run.sh ~/holy/scripts/inference/run_g1_loop_native.py --exec --nav-suspend \
    --rounds 60 --cmd-every 6 --arrive-div 2 --rtc-horizon 10
```

与 v3 的关键语义差（判读前必读）：
- **tick 绝对截止**：睡到 `next_t += TICK`，落后超 1 拍重锚绝不连发（追发=抖动源）
- **星饿=不下发**（刚性伺服保持），超 `--starve-limit 2.0s` 安全停机——与 v3
  hold 兜底（单调收敛）不同，是"冻结"不是"收敛"
- **`--speed 2.0` 是天花板不是配速**：不读 config `[loop].speed`（那是 v3 配速律参数）；
  实际速度=每拍 Δaction/33ms，默认即数据集原速
- **`--arrive-div 2.0`（默认）**：速度=导程÷(窗口拍数×tick×此值)——窗内永不到点，
  伺服不刹停（v3 半程覆盖思想的逐拍版）；1.0=窗末到点（10-09 真机实证启停锯齿，勿回）。
  div 与 N（cmd-every）耦合：速度上限 v_max=限幅÷(N×tick×div)，平衡滞后≈限幅×
  div/k（k=伺服实际覆盖率≈1-2，实测不确定）；**div4@N3 实测锯齿消但滞后爬顶限幅
  300**（151516），N 变窗变时 div 要反向调保 v_max
- **`--const-speed 0`（默认）=判案实验关**；>0 时每条指令速度恒定（仍受 `--speed`
  截断），把速度从"导程→速度"反馈环里拿出来——lerobot 固件 Profile_Velocity
  固定档的等价物（SDK `speed_rad_s` 本义就是速度上限，见 g1 `__init__.pyi`）。
  **10-09 判案结果：证伪**——恒速 1.0 使小导程段 20-80ms 到点+急停循环，
  抖感加剧（150621）；v∝gap 配速律正是防到点机制，方向是"推离到点边界"
  而非拆环
- **`--cmd-every 3`（默认）**：合拍下发，每 N 拍一条位置指令（目标=块内第 k+N-1 行
  绝对位），伺服整窗跑连续轮廓——逐拍指令的 30Hz 重规划激励是持续抖振主源，
  速度律改不掉；合拍=v3 合步家族的细粒度版。消费/换块/水位语义
  不变（仍 1 行/拍，夹爪仍逐拍）；1=逐拍指令旧行为。
  **10-09 五跑定案（真机）**：抖感随指令沿频率单调（30Hz≫10Hz>5Hz），
  **exec 推荐工作点 `--cmd-every 6 --arrive-div 2`**（5Hz 沿、v_max 0.75、
  滞后 mean 95/顶 294，快段顶限幅属预期；152355 五块含快慢段体感不抖，
  残差与 v3 同档）；默认 3 只因与 div2.0 对称，单独 3/2.0 会骑到点边界
  锯齿 hunting（142939/143546 实测 24↔253 mrad）
- **`--n-action-steps` 只定换块时机**（预取水位阈值），不是消费配额——整块换入弃尾
- **`--rtc-horizon 0`（默认）=RTC 关**；>0 时换块把旧块未消费尾段重锚后作引导前缀
  传引擎（消接缝尖峰；机制与消融数据见 BENCHMARKS §八）。前缀窗口**右移
  inference_delay 拍**（=推理 p50÷tick，10-09 修正）：快照在抓帧时刻、落地在
  ~10 拍后，不右移=引导新块倒退已走过的 10 拍（141706 实证接缝 251 mrad 与
  off 无异）
- 遥测新增：tick_ms/tick_jit、星饿窗、**接缝尖峰**（换块后首拍 |Δcmd| vs 块内均值）
- 日志落 `logs/loop_native_<时间戳>.log`；其余护栏（包络/漂移/夹爪/--grip-state-cmd）与 v3 全同

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
