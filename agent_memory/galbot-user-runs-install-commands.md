---
name: galbot-user-runs-install-commands
description: 用户偏好：安装/系统级命令由用户本人执行，我只提供命令清单和判读结果
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 688dacf4-1a8a-4f2b-b926-938022b6de08
  modified: 2026-09-20T05:18:13.540Z
---

用户要求（2026-09-20，FlashRT/pi0.5 环境配置期间）：安装类指令（conda create、pip install、apt 等）不要由我直接执行，发命令清单给用户本人运行。

**Why:** 用户希望对系统变更保持控制权，亲眼过目每条安装命令。

**How to apply:** 涉及安装、升级、删除环境的操作，输出带注释的命令块供用户复制执行；我只直接跑只读检查、代码搜索、构建探测类命令。用户跑完把输出发回来，我负责判读。
