---
name: galbot-s1-machine-state
description: 本机 = S1 机（第四台 Orin）：2026-10-27 装机基线——R35.6.0 四前提全绿免编译，v4 环境包待从 win 侧传入，~/holy 即仓库根（gitee remote）
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-27T04:59:46.201Z
---

**机型：本机是 Galbot S1 系列人形机器人**（2026-10-27 用户明示）；**前序三台——源机/部署机/第三台 machine3——全部是 Galbot G1**（见 [[galbot-flashrt-env-migration-pack]]、[[galbot-machine3-deployment-state]]）。记忆里的基准数据、工作点、资产（关节布局/动作映射/warmup 位姿/包络/数据集句式）都是 **G1 线的产物**，本仓库 README 标题也是"G1 人形机器人部署"——S1 上复用任何结论/资产前必须逐项核对，不能照搬。四台 hostname 全叫 galbot-echo，勿以 hostname 区分机器。

注意区分：S1/G1 是**机型**；而"源机/部署机/machine3"是**机器角色**。底层 Orin 计算环境三台 G1 机已验证同构（sm_87 + JetPack 5.1.x + cuda-11.4），本机 S1 的 Orin 也全绿（见下），机型差异影响的是**机器人本体层**（关节、相机布置、动作语义），不是计算环境。

**2026-10-27 装机检查（bundle 迁移四前提全绿，免编译）**：AGX Orin Developer Kit（/proc/device-tree/model）、aarch64、L4T **R35.6.0**（与源机同版；machine3 是 R35.6.4，版差无碍）、`/usr/local/cuda-11.4` + cuBLAS .11.6.6.84 在位、galbot@/home/galbot 同路径、磁盘 nvme0n1p1 1.8T 空 1.7T、/data/galbot/lib（SDK 库）在。

**结论：环境直接从 win 侧传，不重编。推荐 v4 免热修包（7.1G，sha256 f05251a30a9fb9037d2e64a10ec5ea43dd56f3586e2e017b3906be90f36c2178）**——⚠ machine3 曾收到 win 传的截断包（合法 gzip 包着被截断的 tar，缺 4.3G），**传完必须核 sha256** 再解压。v4 不含权重：跑推理另拷在役 ckpt（machine3 在役 = pi05_g1_onlypick_deploy）到 `~/holy/models/`。

**代码轨已就位**：SSH 密钥（ed25519 → gitee，账号 holy@holydd）2026-10-27 新配；仓库 `git@gitee.com:tq-llm/flashrt-ops.git` clone 后已上移——**~/holy 即仓库根**（与 machine3 同坑同解；DEPLOY.md 双轨管理约定：环境轨 tar 解 `~/holy/{FlashRT,models,...}`，代码轨仓库根 ~/holy）。repo 级 git 身份未配置，首次 commit 前要设。**agent_memory 快照已整目录拷入 ~/.claude/projects/-home-galbot/memory/（2026-10-27），自动 Recall 已生效**。

**部署进度（2026-10-27 下午）**：v4 包已从 win 机 scp 到位并核过 sha256（逐字一致，7,562,700,800 字节）→ 解压 ~ 完成 → apply_hotfix 报「已在」（no-op，v4 免修符合预期）→ verify_deploy.sh **PASS=5 FAIL=1**：①② 前提/位置全绿；③ 环境链核心全绿（torch 2.4.0a0+gitee1b680 sm_87 (8,7)、flash_rt 0.2.0、kernels ok、numpy 1.26.4、cv2 4.10.0、双本地补丁 ok），唯一 FAIL=③ norm_stats 加载——v4 不带权重所致的预期失败，非环境问题；④⑤ 因无权重自动跳过。tar 包（~/flashrt_bundle_v4_env.tar，7.1G）尚未删。

**待办**：拷在役 ckpt 到 `~/holy/models/pi05_g1_onlypick_deploy/`（model.safetensors ≥5GiB + norm_stats.json + config.json，来源 machine3；machine3 在役线见 [[galbot-machine3-deployment-state]]）→ 重跑 verify_deploy.sh（④⑤ 自动真冒烟，含 graph 探测——本机 r35.6.0 的 graph 行为是新信息，值得记录）→ §6 新机清单（相机视角映射/动作空间/SDK 运行时版本）。⚠ 本仓库资产/配置按 **G1 机型**调好（关节布局/动作映射/warmup 位姿/包络），S1 机复用前需逐项核对。
