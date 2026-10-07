---
name: galbot-arm-control-mode-safety
description: 臂=刚性位置伺服(非阻抗)；SDK 臂端 fault 不回传恒 SUCCESS；包络护栏是唯一软防线；Motion.init 脱机挂死
metadata:
  node_type: memory
  type: project
  originSessionId: 44f07522-85b0-4514-990d-7c91716c6107
  modified: 2026-10-07T10:37:38.755Z
---

# G1 臂控制模式与安全护栏（2026-10-06 定案）

## 控制模式：刚性位置伺服（用户已拍板保持）
- echo 机 `robot_config.toml` 注册两臂控制器：`*_arm_pvt_ctrl`(id1, **JCtrlPos2Acc2PosOpen**,
  kp=100/kd=40, 环内无力反馈) + `*_arm_pvt_ctrl_compliance`(id2, JCtrlPosVelAct2Cur, 软 PID
  kp=50/kd=0+电流环+动力学补偿)。`set_joint_positions` 在 libgalbot_sdk.so 里**硬编码路由到
  id1 非柔顺**（符号表无 _compliance 变体），Python 层无切换入口。
- 真正的笛卡尔阻抗（K=500/B=300/M=10, select_matrix=[1,1,1,0,0,0]）只在 v2.2 默认配置
  `right_arm_impedance_pose_ctrl.toml`，echo 未注册；echo 的 compliance 只是"软一点的 PID"非阻抗。
- 运行时切换机制存在：wbcs_test 有 `generate_controller_switch(group, ctrl_id)` +
  `CONTROLLER_OPTIONS_CONTROLER_SWITCH`——vendor 层可切，方法要问 Galbot 方。

## SDK 状态字不透明（关键安全事实）
- **臂端堵转/fault 不回传**：指令恒 `ControlStatus.SUCCESS(tracked)`。10-06 160022 轮6-8
  压桌（残差 108-118 mrad，Δcmd 215 mrad）状态字全程 SUCCESS——与夹爪 fault 报 SUCCESS 同模式
  （见 [[galbot-machine3-deployment-state]]）。
- **残差急停不可行**：残差特征与正常快动作重叠——成功 place 跑 155046 连击 5×≥100 mrad，
  压桌跑只有 2×；任何能抓压桌的阈值都会误杀好跑。已降级为告警打印（≥100×3 段打一次⚠）。
- 夹爪 speed 0.5 m/s 疑超 SDK/固件接受域→爪死（fault）；恢复 0.15 m/s，掉电重启才活。

## 包络护栏（d26f9ce，B 脚本压桌防护）
- `<ckpt>/joint_envelope.json`（models/ gitignore，机器本地，随 ckpt 走）= 任务数据集
  action 臂维(0-6 右,8-14 左)逐维 [min,max]；加载侧 +`--env-margin`(默认 0.15 rad)。
- 指令行越界→截断 chunk 前缀（track 模式消费循 `len(chunk)`，切片即生效）；行 0 越界或
  连续 `--env-abort-n`(5) 轮→停循环。`--no-env-guard` 关闭。夹爪维恒过。
- **⚠ 2026-10-07 修范畴错误（6891103）**：模型输出=delta（[[galbot-pi05-env-setup]] 翻案
  #2、plan_step 2026-09-23 语义修正），护栏比对对象必须是 `BASE_ARM+delta` 绝对构型
  （`chunk_abs_block()`）；原版直接拿 delta 比绝对盒→首块 delta≈0 必拦停（place 切回
  首跑轮 0 即停实录）。当时冒烟"1943 块 0 误报"用的是数据集绝对 action 自比——
  两边都是绝对所以恒过，**没测到模型 delta 输出路径**；np 前置引用 bug 又让护栏从未
  真机服役（带文件即启动崩），两道保障都是假的。干跑新增「包络预检」行。
- 再生成：对任务 parquet action 臂维取 min/max 写 json（夹爪维 null）。
- 无包络文件的 ckpt 自动关=行为不变；place 包络 10-07 已从 only_place 生成。

## Motion.init 脱机挂死（FK 护栏死路）
- `GalbotMotion().init()` 在 motion engine 未起时阻塞 >120s——**运行时 FK 桌高护栏不可行**，
  别再试给 loop 进程加 Motion 依赖。FK API 本身在（`forward_kinematics(target_frame,
  reference_frame, joint_state: Mapping[str, Sequence[float]])`），纯数学签名，但 init 关过不去。

## 行为证据（刚性佐证）
静态跟踪 1.9 mrad（伺服硬保持）、指令切换抖动、PVT 冲击∝速度、碰障 fault 掉电恢复。

相关：[[galbot-machine3-deployment-state]] [[galbot-g1-pvt-trajectory-hazard]] [[galbot-jetson-ik-move-tool]]
