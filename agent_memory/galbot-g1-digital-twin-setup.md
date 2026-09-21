---
name: galbot-g1-digital-twin-setup
description: G1 数字孪生架构：Jetson 读真机关节角经 TCP 推流，Windows 端 MuJoCo 渲染；脚本位置与关键环境约束
metadata: 
  node_type: memory
  type: project
  originSessionId: d99474fc-a75c-4630-8910-1bbf8eb7ebc2
  modified: 2026-09-17T05:03:38.682Z
---

用户在 Jetson Orin + Windows 间搭的 G1 数字孪生（仿真跟随真机，单向只读）：

- Jetson 端 `~/holy/g1_joint_test_src/jetson_sender.py`：SDK 读关节 → JSON over TCP :9999 @30Hz；Windows 端 `windows_viewer.py`：收 JSON 写 `data.qpos` → `mj_forward` + `launch_passive` 渲染。Jetson IP 用有线口 eth1=192.168.1.88（`hostname -I` 第一个值是 eth0 的 192.168.100.88，别用错）。
- 渲染模型：`/home/galbot/galbot_one_golf_description/mjcf/galbot_one_golf_fixed_base.xml`（无 free joint、fixed base、23 执行器）；Windows 用整仓 clone 到 `D:\work\project_file\galbot_one_golf_description`，**新 clone 的 XML 要重新 `sed -i 's| inertia="shell"||g'`**（GitHub 原文件有 schema 错误，Jetson 本地已修但没提交）。
- 关键约束：viewer 循环只准 `mj_forward`，绝不 `mj_step`（23 个执行器失力 + 重力 → 机器人瘫软）；Jetson 上 SDK+mujoco 同进程需 `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMP_NUM_THREADS=1`（已写入 ~/.bashrc），Windows 不需要。
- 读数配对见 [[galbot-sdk-group-mode-ordering-pitfall]]；辅助脚本：`win_static_test.py`（Windows 静态自检）、`jetson_constant_sender.py`（恒定假数据隔离测试）。
