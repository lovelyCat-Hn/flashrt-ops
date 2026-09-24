---
name: galbot-github-remote-setup
description: 本机 GitHub 连接：SSH 账号 lovelyCat-Hn 可用，ghproxy 镜像会卡死 fetch 已移除
metadata: 
  node_type: memory
  type: project
  originSessionId: 944554af-5725-4cba-af39-0c03a553a954
  modified: 2026-09-24T02:26:27.337Z
---

本机（G1 Jetson，2026-09-24 确认）GitHub 连接方式：

- **SSH 直连可用**：`~/.ssh/id_ed25519`（注释 galbot-echo-flashrt），账号 **lovelyCat-Hn**。
- **ghproxy 镜像已移除**（2026-09-24）：曾全局配置 `url.https://ghproxy.com/...insteadOf` 且 FlashRT remote URL 内嵌 ghproxy，在本网络下 fetch 会**无限卡死**；现已 unset 重写、remote 改 SSH。不要再加回镜像。
- 仓库布局：`~/holy` = lovelyCat-Hn/**flashrt-ops**（SSH）；`~/holy/FlashRT` = 上游 flashrt-project/FlashRT（SSH，[[galbot-flashrt-env-migration-pack]] 的运行时源码）。
- git 全局身份是 `32de18 <32de18@users.noreply.github.com>`，与 SSH 账号不一致（用户已知，暂未改）。
- 无 gh CLI、无 HTTPS 凭据存储；push 只能走 SSH。
- **写权限事实**（2026-09-24 push 实测）：lovelyCat-Hn 对上游 flashrt-project/FlashRT **无写权限**（也无同名 fork），用户记忆有误；FlashRT 本地曾积压 2 个提交（407a733f 热修 + 4822d755 graph 修复）待推。flashrt-ops push 正常。
