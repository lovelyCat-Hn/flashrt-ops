---
name: galbot-machine3-deployment-state
description: 第三台 Orin（本机）部署进度：bundle v2 已传到（7.9G tar.gz）、仓库已上移为 ~/holy 本体、G1 权重+数据集随仓库到位；待解压/热修/验收
metadata:
  node_type: memory
  type: project
  originSessionId: 72f683b3-e256-4dd8-9f16-3a497879a902
  modified: 2026-09-30T06:23:42.362Z
---

**本机 = 第三台机器**（前两台：源机/部署机，hostname 都叫 galbot-echo，本机也叫 galbot-echo——[[galbot-flashrt-env-migration-pack]] 的教训：沟通中勿用 hostname 区分，用"本机（第三台）"）。2026-09-23 状态：

- 系统：aarch64，L4T **R35.6.4**（注意：前两台是 R35.6.0，本机 revision 略新），用户 galbot@/home/galbot 同路径，磁盘 nvme0n1p1 1.9T（1.7T 空闲）。
- SSH 已配（ed25519 → GitHub lovelyCat-Hn，连通验证过）；仓库 clone 后**已上移**：`~/holy` 本身即仓库根（用户最初 clone 成了 ~/holy/flashrt-ops，已 mv 上来）。
- **仓库自带**（不用等 bundle）：`~/holy/datasets/pick_place_balence`（870M，199 轨）+ `~/holy/model/pi05_g1_pretrain/pi05_g1_040000`（8.8G，G1 微调权重）。注意是 `model/` 单数，bundle 解出的是 `models/`（pi05_lerobot_base 基座）——两个目录并存，config/g1.toml 的 ckpt 路径要核对指向。
- bundle 取证定案（2026-09-23）：Windows 包（8,458,354,279 字节）与本机逐字节同源，**是"完整合法 gzip 包着被截断的 tar"**——zcat 干净走完 17,058,560,000 字节，与尾部 footer（ISIZE mod 2³²=4173658112, k=3）严丝合缝；断点在 base 权重 model.safetensors 9.5/14.5GB 处，`.cache/flash_rt`（tokenizer）没进包。**缺的 ~4.3GB 在源头就不存在，重传无用**；base 权重将来要另从源机取。
- **环境已验证可用（2026-09-23）**：cuBLAS 前提 ✓、py3.11.16 ✓、fa2/kernels .so ✓、cuda_warmup ✓、`flash_rt 0.2.0 + torch 2.4.0a0+gitee1b680 (8,7) + kernels ok` 与文档期望逐字一致。热修已打（apply_hotfix.sh 探针过 + FlashRT 本地 git 快照）。
- **缺件**：~~tokenizer~~ **已补齐并已入仓**（2026-09-29 GCS 下载成功 + commit 41c99ac 入 `assets/paligemma_tokenizer.model`，DEPLOY.md 资产清单与处置表 #5 已同步指向仓库副本——此后新机免 GCS）；② base 权重 14.5G（用户明说暂不需要，将来从源机取）；③ ~~9.5GB 断文件~~ **已删**（用户确认，2026-09-29，config/postprocessor/norm_stats 小件保留）。**模型加载前置件已全部就绪**，G1 链路只剩本地组装三步。ops 仓库 git 身份（repo 级）= lovelyCat-Hn <lovelyCat-Hn@users.noreply.github.com>（2026-09-29 用户三选一定下）。
- G1 链路当前工作点（2026-09-29 定）：**ckpt = pi05_g1_place**（bf16，`models/pi05_g1_pretrain/pi05_g1_place/040000/pretrained_model`，QUANTILES→q01_q99）+ **数据集 = only_place**（101 轨/41,579 帧，单任务句 "Left arm places A in the top-left corner. Right arm places A in the top-left corner."）。组装三步全绿：stats 提取（夹爪维 q01=-0.00 同款毛刺正常）→ `pi05_g1_place_deploy`（软链+manifest 标定 0.0005/0.1200 沿用）→ norm_align_check 全绿（往返 8e-6/落域 0%/语义分发实证/同噪声确定性 0）。config/g1.toml [run] 已切。旧 `~/holy/model`（单数）已清空，老 040000 在 models/pi05_g1_pretrain/ 下。
- **only_place 起始位姿关键事实**：101 轨全部从**夹爪 ≈33% 开度**起步（places 持物起点，非 pick_place 的 0%！**用户已确认是任务语义设计**：place 状态=手臂已持物，不是数据毛刺）；臂维起点一致性 ≤0.23 rad；leg/head 与旧数据集完全同值。warmup 夹爪目标已改对齐位姿文件（不再硬编码闭合 0%）。
- **warmup 位姿已外置（a0eb0cc）**：三级来源 `--pose-file > config[warmup].pose_file > 部署目录 episode_start_task0.json`（新工具 `g1_pose_extract.py` 从数据集生成，系统 py3.8 跑）。手调位姿：改 json 的 start_pose 或生成变体换 config 指向。
- 待办（DEPLOY.md §6）：真机只读推理（run_g1_inference）→ 相机视角映射/动作空间开环/SDK 运行时版本核对。**2026-09-29 冒烟进度**：模型加载 6.6s/norm q01_q99/23 维状态（33% 持物夹爪）/SDK 1.8.1 核对全绿；相机阻塞已解除（16:19 采集栈重启后 SDK 6/6 路恢复，`sdk_camera_smoke.py` 自检；史与工具见 [[galbot-machine3-camera-transport-unmatch]]），可继续真机只读推理。**闭环 horizon 工作点更新（2026-09-29 用户拍板）**：五脚本 `--horizon` 默认 10→50（训练原生长度，chunk env 自动设；10 切片弃用），run_dataset_replay 补自设 env（原要手动 export），本机 FlashRT df682a76 含 chunk 补丁可直接跑；10 步冒烟基线 e2e 388.5ms，50 步延迟 echo 实锤几乎不变。
**闭环速度瓶颈定案（2026-09-30，三次修正）**：工位=训练工位（首判"场景域差距"是误用别处 replay 旧样张，已纠正）。视角比对曾判"双腕相机仰视 30-40°=手眼伺服全盲→模型 hedge"——**此判已被用户否决（"腕部相机没有任何问题"）且被实证推翻**：执行器调参后模型持续输出任务级计划（105-496 mrad）并多次完整走通释放序列。rollout 60s+ 的真实根因是执行器配速，已定案：臂速律 = n/(30×pace×div)、pace 焊死 0.455、工作点 20/1.55 或 25/1.8，全套推导与签名见 [[galbot-g1-loop-pace-tuning]]。槽位映射 camera_map 全对；evidence/20260930_viewmap/ 比对流程可复用。重规划 2.2Hz=推理极限镜像。launcher 中途 restart 会起半套栈（只 tcp_server+lidar），必须整机重启。

**2026-09-29 同步**：`git pull` 到 **9f60c16**（master 落后 ~60 提交后追平；远程无 main 分支，主干就叫 master；echo-unified 已完全并入且落后 14 提交，是死支）。热修升 5 文件版并已应用（新增 cuda_graph.py WithFlags 修复 + pi05_rtx/_fp16 CHUNK_SIZE env——**graph 段错误已修复**，旧"R35.x 必崩/NO_GRAPH=1"结论作废，但 bench 收益仅 ~3%，源机定档仍 eager）；agent_memory 快照重导入（15 页，新增 cuda-graph-fix / github-remote-setup）。**本周重大翻案（详见新快照）**：① INT8 定档作废→**部署定档 bf16**（INT8 毁动作质量）；② 动作语义=**增量 delta**（非绝对角）；③ dataset replay/`--horizon 50` 整块推理新管线（依赖 chunk env 补丁）；④ DEPLOY.md v3 纯环境包配方（7.5G，不含 models）。tokenizer 仍缺；`model/` 目录现在是未跟踪状态（不影响 pull）。
