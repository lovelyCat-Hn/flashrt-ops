---
name: galbot-sdk-group-mode-ordering-pitfall
description: Galbot SDK 1.8.1 的 get_joint_positions/get_joint_states 用 joint_groups 模式读取时返回顺序与 get_joint_names 不一致，必须按名字显式读取
metadata: 
  node_type: memory
  type: project
  originSessionId: d99474fc-a75c-4630-8910-1bbf8eb7ebc2
  modified: 2026-09-17T05:03:22.719Z
---

Galbot SDK 1.8.1（G1）中 `get_joint_positions(joint_groups=[...])` / `get_joint_states(joint_groups=[...])` 的返回值顺序与 `get_joint_names(joint_groups=[...])` 的顺序**不一致**（实测 21/21 关节全部错位，值是正确值的乱序排列）。虽然 `__init__.pyi` 文档声称 group 模式"按 group 定义顺序返回"，实测不可信。

**Why:** 2026-09 排查数字孪生"仿真姿态错乱"问题时，先用 read_joint_pos.py（group 模式）读数显示"超限值"（如 leg_joint5 读到 -0.54，限位 ±0.1645），一度误判为机器人 WBC 映射损坏；实际按名字逐个读取后全部正常，机器人从未损坏。

**How to apply:** 读关节一律用显式名字模式：`robot.get_joint_positions(joint_names=names)`（文档保证按传入顺序返回），先把 `get_joint_names()` 的结果存下来再传给每次读取。写相关脚本（[[galbot-g1-digital-twin-setup]]）时禁止 group 模式与 names 做 zip 配对。
