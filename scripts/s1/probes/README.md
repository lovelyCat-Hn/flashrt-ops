# S1 probes（回放台 / 执行链隔离工具）

## replay_stream_bench.py — 流式回放台

把轨迹数据喂回 **run_s1_loop 同一条下发路径**，隔离模型变量（G1
`scripts/g1/probes/replay_stream_bench.py` 的 S1 镜像，2026-10-28）。
用途：①真机首验 tfs×singorix WBC 语义的最安全入口（轨迹冻结、无模型方差、
delta-max 链钳自带慢速就位效应）；②量化执行链（节拍迟到/块缝/回读残差）；
③遥测落 `~/holy/evidence/s1_replay/` 为闭环对比预留同格式。

**同构保证**：直接 import run_s1_loop 的代码对象（stream_row/plan_row/
ema_row/send_grippers/sleep_to 原封复用），回放只换"动作来源"。
刻意比主执行器多一层：**每 tick 重读臂关节刷新链钳基准**（G1 traj 同款
per-tick re-read；主执行器现为每块读一次，真机 A/B 后再回移）。

### 数据格式（win 上传轨迹）

- `.npz`：`chunks [C,L,D]` 绝对目标 + `joint_names [D]`（S1 运行时实名）+
  可选 `grip [C,L,2]` 夹爪宽度 m〔left,right〕+ 可选 `hz`
- `.json`：同构 `{"chunks": …, "joint_names": […], "grip": null, "hz": 30}`
- `--synthetic N`：本地合成平滑正弦块（离线冒烟/节拍测试，零依赖）
- 支持关节族 = left/right_arm_joint1-7 / torso_lift_joint1（**米**）/
  head_joint1-2；未知关节名直接拒绝。lerobot parquet 转换器等样例到了再加
  （flash_pyrt311 无 pyarrow/pandas）。

生成 npz 示例：

```python
import numpy as np
np.savez("traj.npz", chunks=chunks, joint_names=names, grip=grip)  # 后两个可选
```

### lerobot 数据集直转：lerobot_to_traj.py

S1 遥测数据集（如 `~/datasets/clean_ready_224_1028`，galbot_one_s1，action
16 维右臂在前、爪=0–100%、fps=30）→ traj npz。**在 lerobot_cpu 环境跑**
（pyarrow 所在；flash_pyrt311 无）：

```bash
~/miniforge3/envs/lerobot_cpu/bin/python scripts/s1/probes/lerobot_to_traj.py \
  --dataset ~/datasets/clean_ready_224_1028 --episodes 0,1,2
```

列映射按 info.json names 按名对号（不依赖列序）；episode 过滤按 data parquet
的 `episode_index` 列（meta 的 dataset_from/to_index 是全库全局索引，勿当文件
内偏移）；爪 %→m 走 base.py 同常量同公式；每集按 50 行切伪块、尾块 hold-last。
⚠ 该数据集 meta/stats.json 臂值范围（±0.5）与 data 实测（±2.0 rad）矛盾
（全库量纲一致，stats 疑清洗前旧口径）——norm 判读勿引这份。

真数据首验读数（ep0-2，2026-10-28）：遥测增量几乎全 <0.05 rad/tick（钳位
≈0，30Hz 回放忠实复现）；chunk0 钳位 257=从零位到工作位姿的就位段（链钳
0.05/tick≈1.5 rad/s 渐进）；集界缝 15-31 mrad、集内伪块缝 1-7 mrad。

### A/B 维度（v1 精简；G1 台 stream/direct/stepwait/tfs 谱系按需再移植）

| 键 | 默认 | 说明 |
|---|---|---|
| `--frames-per-row` | 1 | 1=纯 30Hz 直发（执行器默认）；>1=行内插值微流 |
| `--rows-per-cmd` | 1 | 跨行打包行数 R：命令窗口=R×行周期（30Hz 实时窗口律配方=2×fr12） |
| `--filter-alpha` | **0** | 回放忠实默认 0（显式传才滤波；与执行器 toml 档无关） |
| `--max-joint-delta-per-step` | 0.05 | 臂链钳 rad/tick；0=关仅限位 |
| `--swap-gap-ms` | 0 | 块间 hold 模拟推理落地窗（G1 定案忠实节奏=0） |
| `--rounds` | 1 | 序列重复遍数 |

### 真机窗口律（10-28 定性，用户耳听判据 + PVT toml 机制面）

S1 臂 PVT 控制器（`/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/
{left,right}_arm_pvt_ctrl.toml`）每关节 limit_vel=3.0 rad/s / limit_acc=25 /
**target_pos 一阶低通 cutoff 3Hz @1kHz**（τ≈53ms，阶跃走完 ~150-250ms）。
卡顿判据塌缩为**命令窗口**（=R×行周期）：

| 配方 | 窗口 | 听感 |
|---|---|---|
| 15/30Hz × fr1（单点 tfs=0） | 无（指令="现在"） | 卡卡（步频力矩反复） |
| 30Hz × fr2/fr6（单行 batch） | 33.3ms | 「还好」（前瞻余量薄，≈20ms lookahead 易被抖动击穿） |
| 15Hz × fr2/fr12/fr24 | 66.7ms | 安静；点距越细越佳（fr24=2.78ms 最佳） |

→ 执行器 30Hz 实时等价配方 = **`--rows-per-cmd 2 --frames-per-row 12`**
（窗口 66.7ms + 点距≈2.8ms，两行一次 batch，零额外延迟——整块已在内存）。
爪逐行独立 API 不受影响（A/B 矩阵实证爪无罪）。G1 同族 bypass 文档旁证：
底层 MCU 点流按 4ms 重采样、队列保 ~20ms 前瞻。

### 安全

dry-run 默认**零 SDK**（任何机器可跑；dry-run 状态走虚拟跟随，钳位计数=
真实链钳行为而非冻结态伪影）。`--exec` 需交互确认 + `start_controller`
（急停手不离，模式切换失力矩 10-27 教训）。包络 = S1_LIMITS 钳位 +
delta-max 臂链钳（每 tick 重读基准）。Ctrl-C 即退并落盘部分遥测。
SDK 会话退出走优雅三连 request_shutdown/wait_for_shutdown/destroy（缺它=
退出段 C++ 析构 Segmentation fault，10-28 实证）。

### 用法

```bash
# 离线冒烟（回归闸，stub 零运动）：
PYTHONPATH=/data/galbot/lib:~/holy/scripts/s1/inference \
  ~/miniforge3/envs/flash_pyrt311/bin/python scripts/s1/probes/smoke_offline.py
# 干跑（真机旁先看计划）：
~/miniforge3/envs/flash_pyrt311/bin/python scripts/s1/probes/replay_stream_bench.py --data traj.npz
# 真机回放：
PYTHONPATH=/data/galbot/lib ~/miniforge3/envs/flash_pyrt311/bin/python \
  scripts/s1/probes/replay_stream_bench.py --data traj.npz --exec
```

遥测（每块一行）：`rows | 帧流 | 钳位 | 缝 mrad（块界计划跳变）|
迟到 p50/max ms（节拍落后量；>1 拍=G1 铁律里的重锚场景）|
残差 max（块尾回读 vs 末行目标；臂 mrad / torso mm / head mrad）`。
