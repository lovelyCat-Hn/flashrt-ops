---
name: galbot-g1-pvt-trajectory-hazard
description: G1 PVT 轨迹接口真机危险实录——零运动探针仍致剧烈抖动，轨迹路线已封存，平滑需求走 track 提速路线
metadata: 
  node_type: memory
  type: project
  originSessionId: 688dacf4-1a8a-4f2b-b926-938022b6de08
  modified: 2026-09-22T10:07:33.207Z
---

**G1 PVT 轨迹接口（SDK 1.8.1 `execute_joint_trajectory`）真机危险实录（2026-09-22）**

- 用"目标=当前读数"的零运动轨迹探针（`~/holy/scripts/inference/g1_traj_probe.py`，已封存打标）逐变体试输入形态，**机械臂仍剧烈抖动一次**。零运动目标 ≠ 零风险：PVT 的 velocity 前馈和控制器故障转换都真实作用于关节。
- 实测：变体 C（`velocity=0.5` 前馈）与 E（单组 joint_groups）/ F（单点轨迹）→ 控制器 FAULT + 抖动；A-D（2 点轨迹，joint_names14 或双组 + position±velocity）返回 SUCCESS 但主脚本同形态 4 点轨迹却 INVALID_INPUT，差异未明。
- **后续**：抖动后 `move_whole_body_joint_zero` → MotionStatus.FAULT，SDK 无故障清除接口，最终整机重启解决。一次探针 = 抖动 + 故障 + 停机重启的真机代价。
- **Why**：闭环源 SDK 无接口文档，校验规则与故障行为不可预测；抖动在真机上不可接受。
- **How to apply**：拿到厂商文档或台架条件前，勿在真机跑 `g1_traj_probe.py` 或 `run_g1_loop.py --chunk-mode traj`。平滑运动走 track 路线，但**提速是错的杠杆（2026-09-22 实测证伪）**：`set_joint_positions` 点位控制每个目标全停，冲击∝速度——0.6 明显更抖、0.02 反而平稳（但太慢）。正解=**合步 `--steps-per-cmd`**：一轮 3 步合成 1 条指令，SDK 平滑轮廓一次滑到位，起停次数降为 1/3（每条位移限幅 = spc×delta-max，行程风险不变）。相关：[[galbot-pi05-g1-finetune-data]]
