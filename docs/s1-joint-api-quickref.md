# S1 关节与 SDK API 速查（2026-10-27 装机整理）

适用：GalbotSDK **1.10.0**（本机 runtime `libgalbot_sdk.so.1.10.0` 与源码包版本一致）+ S1 机型
（SN=R002AAFF21AA0063，HW 2.1，BODY_TYPE=A，双臂 A21）。
来源：`~/GalbotSDK`（官方源码+文档）、`/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml`（限位权威）、
`~/yuqiz/{s1_teleop,s1_collect}`（真机跑通代码）、`/data/galbot/lib/galbot_sdk/s1/__init__.pyi`。
G1 线结论见 `agent_memory/`（源机/部署机/machine3 全为 G1），本文只记差异与 S1 事实。

## 1. 与 G1 的核心差异（对齐前必读）

| 项 | G1（记忆基线） | S1（本机） |
|---|---|---|
| SDK | 1.8.1，`galbot_sdk.g1` | **1.10.0**，`galbot_sdk.s1`（`MachineType.S1`），实现类 `GalbotRobotS1` |
| 关节组 | 双臂 / leg / head | 双臂 / **torso（升降柱，高度米）** / head（leg 组不存在） |
| 全身状态维 | 23 | **19**（双臂14 + 双夹爪2 + 躯干1 + 头2） |
| 夹爪 | 塞在关节维里，行程 0–0.12 m | **独立 API、物理宽度（米）**：行程 0.007–0.11 m，反馈单侧 0–0.055 m |
| 控制栈 | joint/PVT 那套 | **singorix WBC 全身控制器**（target 话题 `singorix/wbcs/target`） |
| 1.10.0 API 变化 | — | `get_rgb_data` 参数定义**已改版**（G1 1.8.1 记忆里的用法勿照搬）；新增 `emergency_stop`/`resume_from_emergency_stop`、`get_base_velocity`、`motion_plan`/`traj_plan`/`move_line`/`combine_plan`、`subscribe_video_data`、`set_config`/`get_config` |

## 2. 关节组与命名

`S1JointGroup` 枚举（.pyi 4721 行）：`head` / `left_arm` / `right_arm` / `torso` /
`left_gripper` / `right_gripper` / `swerve_chassis`。

关节名：`left_arm_joint1..7`、`right_arm_joint1..7`、`torso_lift_joint1`、`head_joint1..2`。
⚠ **torso 运行时实名 = `torso_lift_joint1`**（get_joint_names 实测，run_s1_loop 首跑抓出）；yuqiz config 文档写 `torso_joint1`，与 SDK 返回不符——按实读名为准。
运行时以 `robot.get_joint_names(only_active_joint, groups)` 为准（传 `[]` 返回全部组）。

## 3. 19 维状态布局与限位（yuqiz 采集口径，已真机验证）

顺序 = `get_joint_positions(["left_arm", "right_arm", "torso", "head"])` 的拼接 + 夹爪插入：

| 索引 | 名称 | 限位 (rad / % / m) |
|---|---|---|
| 0–6 | left_arm_joint1..7 | [-2.9409,2.9409] [-1.9808,**1.7191**] [-2.9234,2.9234] [-2.5918,1.8937] [-2.9234,2.9234] [-0.7592,0.7592] [-1.5620,1.5620] |
| 7 | left_arm_gripper | [0, 100]（%） |
| 8–14 | right_arm_joint1..7 | [-2.9409,2.9409] [**-1.7191**,1.9808] [-2.9234,2.9234] [**-1.8937**,2.5918] [-2.9234,2.9234] [-0.7592,0.7592] [-1.5620,1.5620] |
| 15 | right_arm_gripper | [0, 100]（%） |
| 16 | torso（运行时实名 `torso_lift_joint1`） | [0.0, 1.0]（**米**，升降高度） |
| 17–18 | head_joint1..2 | [-0.7592,0.7592] [-0.5847,0.5847] |

- ⚠ **左右臂 j2/j4 限位不对称**（镜像但绝对值不同），动作映射/包络护栏必须按左右各自建，不能镜像复制。
- ⚠ **torso 是升降柱不是旋转关节**（S1 与 G1 躯干本质不同）：位置语义 = 升降高度（**米**，行程 0–1.0 m）。
  证据：官方示例 `GalbotSDK/examples/s1/cpp/galbot_robot/src/request_target_example.cpp:268`
  `kTorsoJt = 0.60; // torso target height, meter` + yuqiz 位姿 torso 0.65/0.75 实证。勿按 rad 处理。
- 限位出处：`/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml`（head_pvt/arm_pvt/…_ctrl.toml）。
- ⚠ 头部限位两处数字不一致：SDK 示例注释写 [-0.7854,0.7854]/[-0.6109,0.6109]，
  teleop TOML 为 [-0.7592,0.7592]/[-0.5847,0.5847]——**上机以 TOML/运行时读数为准，待实测定案**。
- 夹爪状态维是 **0–100%**（与 pi05 夹爪维同口径），物理宽度换算需实测标定
  （yuqiz 参考：`gripper_action_scale=0.0011`、feedback max 0.055 m、open 0.11 m）。

## 4. 夹爪：独立 API（与 G1 最大的接口差异）

```python
from galbot_sdk.s1 import GalbotRobot, S1JointGroup, ControlStatus
robot.set_gripper_command(S1JointGroup.left_gripper,
                          width_m,        # 物理宽度 0.007–0.11 m
                          velocity_mps,   # 示例 0.05
                          effort,         # 示例 10
                          is_blocking)    # bool
st = robot.get_gripper_state(S1JointGroup.left_gripper)
# st: timestamp_ns / width (m) / velocity / effort / is_moving
```

端工具还有 suction cup、dahuan3 变体（同目录 TOML 有 `left/right_suction_cup_ctrl.toml`、
`left/right_dahuan3_gripper_ctrl.toml`）；1.10.0 新增 `s1_end_tool_raw_passthrough` 透传接口
（示例 `endtool_raw_*.py`）。本机 SDK 日志有 endtool info 反复超时告警，上机前先核端工具在位。

## 5. 关节指令接口（对照 G1 记忆）

| 接口 | 签名要点 | 备注 |
|---|---|---|
| `set_joint_positions` | `(pos, joint_groups, joint_names, is_block, speed_rad_s, timeout_s)` | **签名与 G1 相同**；names 非空则覆盖 groups；返回 `ControlStatus` |
| `set_joint_commands` | `(joint_commands, joint_groups, joint_names, time_from_start_sec)` | 高频控制（官方示例 100Hz 正弦）；`JointCommand.position` |
| `set_joint_commands_batch` | 批量版 | traj 流式执行器候选 |
| `execute_joint_trajectory` | `Trajectory`/`TrajectoryPoint`（带 time_from_start） | 与 G1 同名机制 |
| `emergency_stop` / `resume_from_emergency_stop` | 1.10.0 新增 | 急停红线工具 |

⚠ 安全（2026-10-27 手臂掉落事件教训，详见 `~/yuqiz/s1_collect/README.md`）：
模式切换/接管键瞬间 WBCS 控制权异常时手臂失力矩坠落——切换前后手不离急停，勿大范围悬空久停。

## 6. 底盘/导航

- 组名 `swerve_chassis`（舵轮）；`get/set_base_velocity`、`set_base_pose`、`get_odom`、`stop_base`。
- 导航示例全套在 `examples/s1/python/galbot_navigation/`（navigate_to_goal(_v2)、move_straight_to、
  waypoints、navigate_along_trajectory、relocalized、bounding_box_filter、check_path_reachability…）。
- G1 导航记忆（chassis-frame / nav-ui / slam-mapping / navigation-stack-troubleshooting）方法论可迁移，
  但 singorix 栈的坐标系/话题名不同，逐项核对后再复用。

## 7. 材料索引

| 材料 | 位置 |
|---|---|
| SDK 源码+文档（1.10.0，版本匹配已核） | `~/GalbotSDK/`（`examples/s1/python/`、`docs/s1/zh/`、`galbot_sdk/linux-aarch64-gcc1330/include/galbot_sdk/galbot_sdk_type.hpp`——C++ 头注释是 API 语义权威） |
| VLA 集成模板 | `~/GalbotSDK/examples/s1/python/tutorials/example5_execute_vla.py`（+example6_pick_and_place） |
| API 存根 | `/data/galbot/lib/galbot_sdk/s1/__init__.pyi` |
| 真机跑通的采集/遥操 | `~/yuqiz/s1_teleop/`、`~/yuqiz/s1_collect/`（19 维布局/限位/夹爪标定的出处） |
| 官方限位/PVT 配置 | `/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml` |
| 厂商 singorix 文档（~30 份） | `/data/galbot/share/galbot/singorix_example/docs/`（s1_config_reference、nx1_python_sdk、quick_start…） |
