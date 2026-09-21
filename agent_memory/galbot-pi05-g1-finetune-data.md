---
name: galbot-pi05-g1-finetune-data
description: G1 真机数据集 pick_place_balence 判读：lerobot v3.0 双臂 16 维格式，与 pi05_libero 7 维不对应也不应硬映射；正确路线=微调 16 维 G1 版 pi05
metadata: 
  node_type: memory
  type: project
  originSessionId: 24ded4cc-d86c-4589-8234-2f1634018dc7
  modified: 2026-09-21T10:20:22.322Z
---

**G1 真机数据集判读（2026-09-21，echo 机 `~/datasets/pick_place_balence/`，zip 857M 解压 870M）**

- **格式**：lerobot **v3.0** 标准格式（meta/info+stats+tasks+episodes、data/chunk-000/*.parquet、videos/ 外挂 mp4），robot_type=`galbot_one_golf`，**30fps，199 条轨迹 / 97,149 帧**，三相机 `head_right / left_arm / right_arm` 均已 224×224（pi0.5 原生输入尺寸）。任务指令是英文双臂句式（"Left arm pick up A. Right arm pick up A." / "places A in the top-left corner"，task_index 0/1）。
- **动作语义（解码确定）**：`action` **16 维** = [左臂 7 关节(rad) + 左夹爪(0~100 百分比) + 右臂 7 关节 + 右夹爪(0~100)]；action[t]≈state[t]（遥操格式，动作=指令关节位形）。`observation.state` **23 维** = 上述 16 维 + 3 个常量维（0.58/1.43/0.90，躯干/升降未动）+ 4 个小范围维（头/腰）。
- **关键结论：与 pi05_libero 的 7 维"对应不上，也不该对应"**——libero 7 维是另一台臂的归一化动作（第 7 维 [−1,1] 夹爪，前 6 维疑似归一化 EEF 量），与 G1 双臂 16 维关节空间在维度/单位/运动学上全不同，硬映射必 off-distribution。
- **16 维部署预检已通过（2026-09-21）**：pi0.5 原生动作隐空间就是 **32 维**（action_out_proj [32,1024]，config max_action_dim=32），7 只是消费侧切片 → 16 维放得下，**权重一个字节不用改**。已打 FlashRT 补丁（本地 commit 6425fe8d，未推上游）：前端 `__init__` 新增 `action_dim` 参数（默认 7 行为不变）+ 三处 unnorm 切片改 `self._out_action_dim` + `api.load_model` 向前端转发 action_dim。smoke checkpoint `~/holy/models/pi05_g1_smoke/`（软链原权重 + config action.shape=[16] + 16 维占位 norm_stats），`load_model(action_dim=16)` → 输出 **(10,16)** ✓。**教训：monkeypatch 前端 `__init__` 必须加 `functools.wraps`，否则 `inspect.signature` 看到的是 `(*a,**kw)`，api.py 的按签名转发会静默丢弃 action_dim**（四个脚本已修）。
- **正确路线**：这批数据就是微调就绪格式 → 用 lerobot/openpi 训练管线在 **x86 GPU 机器**上微调出 **16 维动作空间的 G1 版 pi05**（prompt 直接沿用数据集英文句式）→ FlashRT 部署该 checkpoint。**部署侧剩余校验**：归一化语义对齐（lerobot 训练用 MEAN_STD，FlashRT `unnormalize_actions` 是 q01/q99 分位数——语义不同须对齐，最隐蔽）；state 23 维的 prompt 编码路径（base 是 8）。base 模型开环观察价值有限（语义不匹配），链路验证已由 ab_real_camera 完成。
- 系统 python3.8 有 pandas+pyarrow（读 lerobot parquet 用它）；flash_pyrt311 无 pandas（勿装，numpy 钉版红线）。
