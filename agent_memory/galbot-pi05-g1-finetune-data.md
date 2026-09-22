---
name: galbot-pi05-g1-finetune-data
description: G1 真机数据集 pick_place_balence 判读 + 16 维部署全链路已搭好（prep/align/run 三件套，2026-09-22 真机验证）；等微调权重直接 --ckpt 接入
metadata: 
  node_type: memory
  type: project
  originSessionId: 24ded4cc-d86c-4589-8234-2f1634018dc7
  modified: 2026-09-22T08:54:20.297Z
---

**G1 真机数据集判读（2026-09-21，echo 机 `~/datasets/pick_place_balence/`，zip 857M 解压 870M）**

- **格式**：lerobot **v3.0** 标准格式（meta/info+stats+tasks+episodes、data/chunk-000/*.parquet、videos/ 外挂 mp4），robot_type=`galbot_one_golf`，**30fps，199 条轨迹 / 97,149 帧**，三相机 `head_right / left_arm / right_arm` 均已 224×224。任务指令英文双臂句式（task_index 0/1）。meta/stats.json 含全套 min/max/mean/std/q01..q99。
- **动作/状态语义（2026-09-22 二次修正）**：`action` **16 维** = [**右臂7(rad) + 右夹爪(0~100%) + 左臂7 + 左夹爪(0~100%)**]；`observation.state` **23 维** = 上述 16 维 + **leg_joint1-5 + head_joint1-2**。**块序【右臂在前】——meta/info.json 的 features names 是唯一权威**。曾误记"左臂在前"并按它装配 state/下发目标：右臂统计喂左臂读数，预热把机械臂甩到背后（用户"到背面来抓东西了"）；离线复算旧错序恰好复现真机首跑"[-1,1] 外 15 维"。腿/头直读始终正确（episode0 第0帧 leg 0.580/1.433/0.902/0/0 + head −0.061/0.330 与 SDK 实测全等）。episode0 第0帧=标准工作位（各轨迹同一起点位，与 state.mean 几乎重合——"均值幻影"假设已排除）。数据集现已在本机 `~/holy/datasets/pick_place_balence/`（870M，带三相机 mp4）。SDK `get_joint_names()` 恰 23 个关节一一对应。
- **关键结论：与 pi05_libero 的 7 维"对应不上，也不该对应"**——正确路线=微调 16 维 G1 版 pi05（x86 GPU，lerobot 管线，prompt 沿用数据集英文句式）。

**16 维部署全链路已搭好并真机验证（2026-09-22）**——pi0.5 原生动作隐空间 32 维（action_out_proj [32,1024]），16/7 维只是消费侧切片，权重零改动。三件套（`~/holy/scripts/`）：
1. `inference/g1_ckpt_prep.py`：微调输出 → 部署目录（软链 14G + config + norm_stats.json 带 `norm_mode` 标记 + flashrt_deploy.json 部署清单）。预检：safetensors header 张量校验（键名带 `.weight` 后缀！）、config/stats 维数三方一致、tokenizer 在位。
2. `eval/norm_align_check.py`：Part A 双语义数学往返（1e-5）→ Part B state 落域+常量窄维预警 → Part C **单实例 stats 翻转法**实证引擎分发（反解 raw 一致 6e-08、mean_std 仿射符合 8e-09）→ Part D 落域。
3. `inference/run_g1_inference.py`：真机只读推理。manifest 自动配置零参数；显式名字读 23 维关节 → `normalize_state` → predict → (10,16) + 三层打点（ctrl-hz 默认 30=数据集 fps）。实测（源机，服务常驻负载）：①纯推理 232ms（≈基线）、②取图 22ms、③端到端 mean 450/p50 400/min 257/max 803ms——**min=基线但方差大，autotune 块中途重现 ×2 指向显存分配翻转引发引擎重规划**，未解（下一步 tegrastats 定位）。**推理→SDK→关节闭环已打通（2026-09-22）**：`g1_pose_warmup.py` 预热到工作位（臂 16 维归一化全部入域，余 3 维 leg 窄维噪声）→ `run_g1_execute.py --exec` 3 步全 SUCCESS、回读最大偏差 2.3 mrad。工具已入库（483de44）。

**FlashRT 本地补丁（未推上游）**：`6425fe8d` action_dim 参数+三处切片可配置；`bff59419` `unnormalize_actions` 按 norm_stats 顶层 `"norm_mode"` 分发（`q01_q99` 默认旧行为逐位不变 / `mean_std` = x*std+mean）+ 新增 `normalize_state`（flash_rt/core/utils/actions.py）。**关键事实：FlashRT 不归一化 state**——`discretize_pi05_state` 直接把输入按 [-1,1] 分 256 bin 进 prompt，喂原始关节值=静默打满 bin，必须调用方先 `normalize_state`。**教训：monkeypatch 前端 `__init__` 必须 `functools.wraps`，否则 api.py 按签名转发静默丢参**。**坑：跨两次模型实例同噪声 raw 不逐位一致**（14G 常驻改变显存布局/cuBLAS 路径），对齐实验须同实例内做（stats 翻转法）。

**归一化语义（最隐蔽坑）**：norm_mode 选错=动作系统性畸变。lerobot 管线默认 MEAN_STD→`mean_std`；openpi 默认分位→`q01_q99`。数据集窄维 7 个（leg/head，分位宽 <1e-3）：部署摆位须与采集一致否则 bin 打满；run 脚本已内置越界维提示。**剩余待办**：① 夹爪标定（数据集 0~100% vs SDK 开口宽度米，`--grip-wmin/--grip-wmax` 钩子已留）；② x86 微调本身；③ 微调权重到手后 `prep → align → run` 三步接入。
- 系统 python3.8 有 pandas+pyarrow（读 lerobot parquet 用它）；flash_pyrt311 无 pandas（勿装，numpy 钉版红线）。
