---
name: galbot-s1-sdk-and-body-basics
description: S1 机 SDK/本体要点：SDK 1.10.0（galbot_sdk.s1，MachineType.S1）、状态 19 维/关节组含 torso、夹爪行程 0.007–0.11m、厂商 singorix 文档在 /data/galbot/share、同事可跑源码在 ~/yuqiz
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-27T05:26:05.062Z
---

**S1 SDK/本体基线（2026-10-27 盘点，机器 SN=R002AAFF21AA0063，ROBOT_SERIES=S1，HW 2.1，BODY_TYPE=A，双臂 A21）**。与 G1 线的关键差异（来源 [[galbot-s1-machine-state]] 机型差异的落地细节）：

- **SDK 版本**：本机 `/data/galbot/lib` = galbot_sdk **1.10.0**（G1 三台是 1.8.1），专用实现 `GalbotRobotS1`。pybind 全版本绑定（cp38~314，**含 3.11** → flash_pyrt311 可直接 import）；模块有 `s1`/`g1`/`g3` 三个子包，**用法 `galbot_sdk.s1` + `MachineType.S1`**。`s1/__init__.pyi` 类型存根在 `/data/galbot/lib/galbot_sdk/s1/`——完整 API 面可直接读，无需外部文档。
- **关节组差异**（同事 yuqiz 的 G1→S1 移植实测）：G1 = left/right_arm + leg + head → **S1 = left/right_arm + torso + head**；**状态 19 维**（双臂14 + 双夹爪2 + 躯干1 + 头2），G1 是 23 维。**夹爪行程 S1 = 0.007–0.11 m（反馈 0.055）**，G1 是 0–0.12 m（反馈 0.06）——pi0.5 夹爪标定/归一化不能照搬。
- **控制栈**：S1 走 **singorix WBC（全身控制器）**，target 话题 `singorix/wbcs/target`；主从臂遥操链 = galbot_upper(APP) 切 TELEOPERATION_MODE → hpu_launcher → singorix_wbcs_main → 主臂硬件写 wbcs/target。数采集要切 DATA_COLLECTION_MODE。G1 的 joint/PVT 那套接口不直接适用。
- **本地资料（无需下载 SDK 包）**：① 厂商文档 ~30 份 markdown 在 `/data/galbot/share/galbot/singorix_example/docs/`（`singorix_s1_config_reference_guide.md`、`singorix_nx1_python_sdk_guide.md`、quick_start、dexhand、endtool、chassis pose、PVT bypass 等）；② 同事源码 `~/yuqiz/s1_teleop/`（13KB s1_teleop.py + 配置）+ `~/yuqiz/s1_collect/`（采集/复位/读位姿/镜像/goto 全套，G1 without_hil 移植版，离线转 LeRobot v3，profile=s1_pick_3cam_14d/s1_place_4cam_21d）；③ 当日 evidence `~/yuqiz/evidence_20261027/`。
- **⚠ 安全教训（2026-10-27 手臂掉落事件，已写进 yuqiz README）**：切换模式/按接管键前后手臂可能瞬间失去力矩保持（WBCS 控制权切换异常→重力坠落）；接管时另一只手放急停旁；勿让手臂长时间大范围悬空。SDK 日志另有 endtool info 反复超时（`~/galbot_sdk_log/s1/`，左右端工具 info 不可用告警，原因待查）。
- **G1 记忆的适用性**：galbot-sdk-group-mode-ordering-pitfall（显式名字模式防乱序）、arm-control-mode-safety 等方法论仍适用；但关节名/组名/维数/行程全部要按 S1 重核，s1_collect 的 config（复位位姿等）是现成起点。

**SDK 源码包已到位（2026-10-27 下午，用户下载）**：`~/GalbotSDK/` = **1.10.0**，与本机 runtime 版本逐字匹配（VERSION 文件核对）。含 `examples/s1/python/`（galbot_robot/motion/navigation/perception/tutorials 六组，**tutorials/example5_execute_vla.py = VLA 集成模板**、example6_pick_and_place）、`docs/s1/zh/`（mkdocs HTML：api_python_reference/set_config_reference/tutorials）、`galbot_sdk/linux-aarch64-gcc1330/include/.../galbot_sdk_type.hpp`（C++ 头注释=API 语义权威）。`~/lerobot` 也已克隆（machine3 同款）。

**S1 关节速查表已入仓**：`~/holy/docs/s1-joint-api-quickref.md`（未 commit，待用户过目）。核心事实：19 维布局/限位表出自 yuqiz config.py，源头=官方 `/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml`；**左右臂 j2/j4 限位不对称**（[-1.9808,1.7191] vs [-1.7191,1.9808]，映射/包络必须左右各建）；**头部限位两处口径不一致**（SDK 示例注释 ±0.7854/±0.6109 vs TOML ±0.7592/±0.5847）待实测定案；torso 限位 [0,1.0]；夹爪状态维 0–100% 与 pi05 同口径、物理行程 0.007–0.11 m 换算待标定；`set_joint_positions` 签名与 G1 相同，`set_joint_commands` 高频接口（100Hz 示例）+ `set_joint_commands_batch` 是 traj 执行器候选。另有 repo 级 skill `bench-report`（维护 BENCHMARKS.md 用）随 holy/.claude 就位。
