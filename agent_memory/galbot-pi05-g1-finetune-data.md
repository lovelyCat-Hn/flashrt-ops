---
name: galbot-pi05-g1-finetune-data
description: G1 真机数据集 pick_place_balence 判读 + 16 维部署全链路已搭好（prep/align/run 三件套，2026-09-22 真机验证）；等微调权重直接 --ckpt 接入
metadata: 
  node_type: memory
  type: project
  originSessionId: 24ded4cc-d86c-4589-8234-2f1634018dc7
  modified: 2026-09-22T01:36:36.757Z
---

**G1 真机数据集判读（2026-09-21，echo 机 `~/datasets/pick_place_balence/`，zip 857M 解压 870M）**

- **格式**：lerobot **v3.0** 标准格式（meta/info+stats+tasks+episodes、data/chunk-000/*.parquet、videos/ 外挂 mp4），robot_type=`galbot_one_golf`，**30fps，199 条轨迹 / 97,149 帧**，三相机 `head_right / left_arm / right_arm` 均已 224×224。任务指令英文双臂句式（task_index 0/1）。meta/stats.json 含全套 min/max/mean/std/q01..q99。
- **动作/状态语义（已完全证实）**：`action` **16 维** = [左臂7(rad) + 左夹爪(0~100%) + 右臂7 + 右夹爪(0~100%)]；`observation.state` **23 维** = 上述 16 维 + **leg_joint1-5 + head_joint1-2**（2026-09-22 真机逐维核验：SDK 读数 0.580/1.432/0.902/-0.001/0.001/-0.061/0.330 与数据集统计逐维吻合）。SDK `get_joint_names()` 恰 23 个关节与数据集一一对应。
- **关键结论：与 pi05_libero 的 7 维"对应不上，也不该对应"**——正确路线=微调 16 维 G1 版 pi05（x86 GPU，lerobot 管线，prompt 沿用数据集英文句式）。

**16 维部署全链路已搭好并真机验证（2026-09-22）**——pi0.5 原生动作隐空间 32 维（action_out_proj [32,1024]），16/7 维只是消费侧切片，权重零改动。三件套（`~/holy/scripts/`）：
1. `inference/g1_ckpt_prep.py`：微调输出 → 部署目录（软链 14G + config + norm_stats.json 带 `norm_mode` 标记 + flashrt_deploy.json 部署清单）。预检：safetensors header 张量校验（键名带 `.weight` 后缀！）、config/stats 维数三方一致、tokenizer 在位。
2. `eval/norm_align_check.py`：Part A 双语义数学往返（1e-5）→ Part B state 落域+常量窄维预警 → Part C **单实例 stats 翻转法**实证引擎分发（反解 raw 一致 6e-08、mean_std 仿射符合 8e-09）→ Part D 落域。
3. `inference/run_g1_inference.py`：真机只读推理。manifest 自动配置零参数；显式名字读 23 维关节 → `normalize_state` → predict → (10,16) + 三层打点。实测：3 视角 INT8 纯推理 232-238ms、取图+读关节 21ms、端到端 256-259ms（38.5-39 步/秒）。

**FlashRT 本地补丁（未推上游）**：`6425fe8d` action_dim 参数+三处切片可配置；`bff59419` `unnormalize_actions` 按 norm_stats 顶层 `"norm_mode"` 分发（`q01_q99` 默认旧行为逐位不变 / `mean_std` = x*std+mean）+ 新增 `normalize_state`（flash_rt/core/utils/actions.py）。**关键事实：FlashRT 不归一化 state**——`discretize_pi05_state` 直接把输入按 [-1,1] 分 256 bin 进 prompt，喂原始关节值=静默打满 bin，必须调用方先 `normalize_state`。**教训：monkeypatch 前端 `__init__` 必须 `functools.wraps`，否则 api.py 按签名转发静默丢参**。**坑：跨两次模型实例同噪声 raw 不逐位一致**（14G 常驻改变显存布局/cuBLAS 路径），对齐实验须同实例内做（stats 翻转法）。

**归一化语义（最隐蔽坑）**：norm_mode 选错=动作系统性畸变。lerobot 管线默认 MEAN_STD→`mean_std`；openpi 默认分位→`q01_q99`。数据集窄维 7 个（leg/head，分位宽 <1e-3）：部署摆位须与采集一致否则 bin 打满；run 脚本已内置越界维提示。**剩余待办**：① 夹爪标定（数据集 0~100% vs SDK 开口宽度米，`--grip-wmin/--grip-wmax` 钩子已留）；② x86 微调本身；③ 微调权重到手后 `prep → align → run` 三步接入。
- 系统 python3.8 有 pandas+pyarrow（读 lerobot parquet 用它）；flash_pyrt311 无 pandas（勿装，numpy 钉版红线）。
