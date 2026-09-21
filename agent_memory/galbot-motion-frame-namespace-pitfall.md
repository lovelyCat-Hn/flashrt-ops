---
name: galbot-motion-frame-namespace-pitfall
description: GalbotMotion 三个位姿 API 各用一套帧命名空间，set_end_effector_pose 还必须显式传 params=Parameter()，桩文件不可信
metadata: 
  node_type: memory
  type: project
  originSessionId: d99474fc-a75c-4630-8910-1bbf8eb7ebc2
  modified: 2026-09-17T07:35:06.409Z
---

Galbot SDK 1.8.1 `GalbotMotion` 的三个位姿接口帧命名空间**互不相同**（.pyi 桩文件完全没写清，实测 2026-09）：

- `inverse_kinematics(...)`：`target_frame="EndEffector"`（链昵称）+ `chain_names=["left_arm"]`
- `get_end_effector_pose(f)` / `forward_kinematics(f)`：f 只认 **link 名**，如 `left_arm_end_effector_mount_link`（链 EndEffector == mount link，同一物理点；模型里没有 gripper_tcp_link）
- `set_end_effector_pose(...)`：`end_effector_frame` 只认**链名** `"left_arm"`/`"right_arm"`，且**必须显式传 `params=Parameter()`**，缺省/其他帧名都报 `INVALID_INPUT: Input parameters do not meet requirements`

**Why:** 调试 jetson_ik_move.py 时 set 报 INVALID_INPUT，试遍 timeout/blocking/link 名无效；最终在用户仓库 `galbot-g1-vla-master/GalbotSDK-1.8.1/docs/g1/en/python/tutorials/example5_pick_and_place.py` 找到官方调用（帧=链名+params=Parameter()）才命中。排查用零动作目标（target=当前位姿）实测，安全。

**How to apply:** 运行时方法名与桩也有出入（`get_supported_chains/links/ee_frames`，无 `*_names` 后缀）；查真实签名用 `motion._impl.set_end_effector_pose.__doc__`（pybind 签名含默认值）。相关读数坑见 [[galbot-sdk-group-mode-ordering-pitfall]]，架构见 [[galbot-g1-digital-twin-setup]]。
