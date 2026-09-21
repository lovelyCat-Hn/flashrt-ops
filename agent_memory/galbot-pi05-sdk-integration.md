---
name: galbot-pi05-sdk-integration
description: pi0.5↔GalbotSDK 集成要点：3.11 同进程已验证可行（需 LD/PYTHONPATH=/data/galbot/lib）；相机 API 是压缩图 CPU 软解；arm 7-DoF 与 libero 动作空间需映射
metadata: 
  node_type: memory
  type: project
  originSessionId: 24ded4cc-d86c-4589-8234-2f1634018dc7
  modified: 2026-09-21T06:41:33.323Z
---

**GalbotSDK 与 FlashRT/pi0.5 集成要点（2026-09-21，echo 机实查）**

- **运行时版本注意**：`/data/galbot/lib` 装的是 galbot_sdk **1.8.1**（libgalbot_sdk.so.1.8.1），workspace 的 `~/workspace/GalbotSDK-1.7.1/` 是文档+示例源码（examples/g1/python 最有用，tutorials/example6_execute_vla.py 就是 VLA 集成骨架）。
- **同进程集成可行（已验证）**：galbot_sdk 带 cpython-38~314 全系列 pybind 绑定，flash_pyrt311(3.11) 里 `import galbot_sdk.g1` + torch 2.4 共存 OK。启动条件：`LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib`（SDK 运行库与模块都在此；该目录无 CUDA 库，不会劫持 torch）。这正是 bashrc 125/132 行全局 export 的原因，旧 libcurl 毒化 HTTPS 是副作用——跑 FlashRT+SDK 的进程不需要 HTTPS，可接受。
- **相机 API**：`GalbotRobot().init(enable_sensor_set)` 开启；`robot.get_rgb_data(SensorType.X)` → dict `[header, format, data]`，RGB 是**压缩图（format=rgb8），官方示例用 cv2.imdecode CPU 软解**；深度 `16UC1` 原始 uint16（带 height/width/depth_scale）无需解码。内参走 `robot.get_camera_intrinsic()`。SensorType 枚举：HEAD_LEFT/RIGHT_CAMERA、LEFT/RIGHT_ARM_CAMERA(+_DEPTH_CAMERA)、4×SURROUND、LIDAR/IMU。建议映射到 pi0.5：HEAD_LEFT→image、LEFT_ARM_CAMERA→wrist_image、RIGHT_ARM_CAMERA→wrist_image_right（需实测视场确认）。
- **机械臂 API（两套，example2/example8）**：①关节空间 `robot.set_joint_positions(pos, groups, names, is_blocking, max_speed, timeout)`、`execute_joint_trajectory(Trajectory)`（TrajectoryPoint 带 time_from_start，示例 dt=0.008s）、`get_joint_names/positions/states`；②任务空间 `GalbotMotion`：`get_end_effector_pose_on_chain(chain_name, frame_id, reference_frame)`（帧 EndEffector / base_link，链 left_arm/right_arm/head/leg）、`inverse_kinematics`、`forward_kinematics`（要显式 `gm.Parameter()`）、`set_end_effector_pose`、`check_collision`。臂各 7 关节。收尾三件套 request_shutdown/wait_for_shutdown/destroy。[[galbot-motion-frame-namespace-pitfall]] [[galbot-sdk-group-mode-ordering-pitfall]] 的坑仍适用。
- **embodiment 差距（关键预期管理）**：pi05_libero_base 输出 (10,7) 对应 LIBERO 臂的动作空间，与 G1 臂 7 关节**不是恒等映射**（量纲/零位/夹爪语义都未知）；base 模型按 README 是"供微调用"。上真机先开环观察动作合理性，别直接闭环执行。
- **example6 集成模板**：图像字典 → VLA 函数 → 各组关节轨迹 → motion.check_collision → execute_joint_trajectory；把 fake_vla 换成 FlashRT `predict()`、输出 (10,7) 转右臂轨迹即可起步。
