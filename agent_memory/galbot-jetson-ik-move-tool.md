---
name: galbot-jetson-ik-move-tool
description: jetson_ik_move.py 交互式 IK 运动工具：命令集、已验证事实（leg 工作空间、假解坑）、四元数约定
metadata: 
  node_type: memory
  type: project
  originSessionId: d99474fc-a75c-4630-8910-1bbf8eb7ebc2
  modified: 2026-09-17T08:50:31.723Z
---

`~/holy/g1_joint_test_src/jetson_ik_move.py`：Jetson 上的交互式末端 IK 运动工具（2026-09 搭建并实测）。

- 命令：`pose` / `move|goto <链> dx dy dz` / `rot <链> r p y`（度，相对，单次限 30°）/ `solve`（只解不执行）/ `home` / `q`。全部阻塞式，单线程交互。安全：leg 单步限 0.05m、臂 0.15m；碰撞检查常开。
- 执行路径分叉：**手臂**走 `set_end_effector_pose`（帧=链名+`params=Parameter()`，见 [[galbot-motion-frame-namespace-pitfall]]）；**leg** 规划器拒收，走"SDK IK 解 5 角 → `set_joint_positions` 按名直控（0.2 rad/s）"混合路径。
- **leg 工作空间**（FK 扰动实测）：矢状面连杆——x 前后 ✓、z 升降 ✓（升降时 j1/j2/j3 按约 1:2:1 配比协同），y 横移 ✗ 零自由度；j4/j5 只转姿态不平移。**SDK leg IK 对够不着的目标返回"SUCCESS"但解=当前姿态（假解）**——脚本已加解后 FK 残差校验（位置>1cm 或姿态>5° 拒绝执行）。
- 链 EndEffector 物理点 = `*_end_effector_mount_link`（link7 法兰前伸约 11cm，夹爪安装座）；head 链 IK 不可解；支持链：head/left_arm/right_arm/leg/mobile_base/torso。
- 四元数约定 **[qx,qy,qz,qw]**（SDK 全系）；`forward_kinematics` 的 joint_state 形状是 `{"链名": [角度列表]}`。
- 实测精度：手臂 5cm 平移指令与实测位移毫米级吻合。

**How to apply:** 改朝向用 rot（RPY 度）别手改四元数；验证 IK 目标可达性先 `solve`，但 leg 的 solve 要看"解≠当前姿态"才算真可达（或看脚本的残差警告）。恢复数字孪生推流：`python3 jetson_sender.py`（此前已按用户要求停掉，端口 9999 已释放）。
