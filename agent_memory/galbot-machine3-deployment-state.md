---
name: galbot-machine3-deployment-state
description: 第三台 Orin（本机）部署进度：bundle v2 已传到（7.9G tar.gz）、仓库已上移为 ~/holy 本体、G1 权重+数据集随仓库到位；待解压/热修/验收
metadata:
  node_type: memory
  type: project
  originSessionId: 72f683b3-e256-4dd8-9f16-3a497879a902
  modified: 2026-10-07T10:16:39.789Z
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

**2026-09-30 下午定案包落地**：① [loop] 工作点 20/20/0.45/1.47/near-div2/speed1.0/delta_max0.3 写入 config+BUILTIN（L 半程真机验证：释放序列干净无振荡）；② 双臂串行时间线定案——**60 轮判别跑已结案**：L 释放 12-15 轮、R 释放 37-39 轮、R 回闭 51、全程 35.9s，**双臂完整序列首次走通**，预算实测 52-60 轮、**q 停=R 落盒才停**（30 轮截断定案，见 [[galbot-g1-loop-pace-tuning]]）；③ cache_frames=2 判死（[[galbot-pi05-cache-frames-dead]]）；④ graph 解封+闭环首验通过（[[galbot-g1-cuda-graph-instantiate-fix]]，p50 379.9ms）；⑤ USAGE.md 新增 §3.6 闭环操作节 + state 布局纠错（右臂在前，探针实证）。**⚠ 判别跑双爪起点 ≈1mm 疑似空爪**——运动/时序全验证成立，物理 place（物体真落盒）待持物复验。

**2026-10-04 RTC A/B 实验【已完结 + 工作点 v2 定案】**：用户目标=压全程时间（当天持物跑 22s vs 数据集 13s；L→R 串行窗口 ~7s 其中 ~5s 是数据固有串行 0/101——**串行结构数据里来的，不动**）。B 脚本 `scripts/inference/run_g1_loop_rtc.py`（2b2c22e，坡升保险丝 273b7a2）与 A 逐行同源，唯一差异=推理迟到改 **RTC hold**（重发最后绝对目标单调收敛+10ms 轮询早退；`--no-rtc-hold` 退回 A 行为）。**当日 14 跑全绿**：hold 200+ 窗全 |Δcmd|=0 零垃圾、9/30 A@0.40 破位灾难绝迹、R 落盒 19.7→**14.0-14.5s 纪录**（数据集 13s）。**工作点定案 v2（0b93786 已入 config+BUILTIN）：25/0.38/1.65（1.33×），入口 B 脚本**——pace 破水位后是死 knob（0.35-0.43 墙钟相同）、div 饱和（三档零差取 1.65=臂最慢）、spc=25 是唯一结构杠杆（−2.3s 且速度律命中）。denoise_step 前缀引导维持未搬（eager 可行但 VJP≈2-3× 推理杀节奏，且本链路每轮新鲜重抽帧无陈旧观测病；候补=W8A16 推理 <200ms 后）。lerobot 源码在 /home/galbot/lerobot。⚠ 全天空爪排练，**持物物理 place 复验仍挂着**。

**2026-10-05 切回 pick 任务**：only_place/place 测试线暂停（持物复验仍未做），回 pick 线。`pi05_g1_040000`（pick_place_balence 微调，训练归一化 QUANTILES）装配为 `models/pi05_g1_pick_deploy`：stats 从 ckpt normalizer 直提、grip 标定 0.0005/0.12 沿用、norm_align_check 全绿；起始位姿从数据集重生成（0% 闭爪），与 `pi05_g1_ft/episode_start_task0.json` 旧件逐维一致。config 三键已切（ckpt/prompt="Left arm pick up A. Right arm pick up A."/pose_file），place 三键在 [run] 注释存档可随时切回；闭环工作点 25/0.38/1.65 不变（速度律与任务无关，pick 时线不同轮数预算需重看）。

**2026-10-07 相机恢复 + SDK 在场遥测补全**：相机 transport 失败窗口经完整重启解除（16:45 那代失败窗口、16:56 重启后 6/6 真载荷，见 [[galbot-machine3-camera-transport-unmatch]]）。`run_g1_inference.py`（只读）三层遥测 n=20：①冻结帧纯推理 bf16 375.3 / W8A16 **356.5（−18.8ms 现场兑现）**，②取图+关节 8.3ms，③端到端 388.9/368.5；冻结帧动作 sanity 全对。loop 388 归因闭合：+138=SDK/相机栈常驻、+13=并行取图（详见 BENCHMARKS 附录）。graph-on W8A16 闭环质量验证仍挂。

**2026-10-07 晚：切回 place + 导航搁置杠杆入 B 脚本（49ea7bf）**：config 三键切回 pi05_g1_place_deploy（pick 三键注释存档），工作点 25/0.38/1.65 不变。B 脚本新增 `--nav-suspend`（默认关）：闭环期 SIGSTOP 冻结导航栈 7 进程，退出自动 CONT（5 退出点显式 resume + 独立守护进程兜底 kill -9，CONT 幂等）；干跑验证冻结/恢复、守护网自退、服务全回 S 态。**修包络护栏 np 前置引用 bug**——解析点在 import numpy 之前，envelope 文件存在即 NameError（10-06 合入后无真机全跑未暴露）；`pi05_g1_place_deploy/joint_envelope.json` 已从 only_place 直提生成（101 eps/41579 帧，models/ 不入库）。⚠ 首跑注意：① place ckpt 无 pick 时代的压桌历史，包络护栏首战；② 持物物理 place 复验仍挂着（此前判别跑疑空爪）。用户自行跑真机 place 测试（首跑只开 --nav-suspend，W8A16 留作隔离变量）。

**2026-09-29 同步**：`git pull` 到 **9f60c16**（master 落后 ~60 提交后追平；远程无 main 分支，主干就叫 master；echo-unified 已完全并入且落后 14 提交，是死支）。热修升 5 文件版并已应用（新增 cuda_graph.py WithFlags 修复 + pi05_rtx/_fp16 CHUNK_SIZE env——**graph 段错误已修复**，旧"R35.x 必崩/NO_GRAPH=1"结论作废，但 bench 收益仅 ~3%，源机定档仍 eager）；agent_memory 快照重导入（15 页，新增 cuda-graph-fix / github-remote-setup）。**本周重大翻案（详见新快照）**：① INT8 定档作废→**部署定档 bf16**（INT8 毁动作质量）；② 动作语义=**增量 delta**（非绝对角）；③ dataset replay/`--horizon 50` 整块推理新管线（依赖 chunk env 补丁）；④ DEPLOY.md v3 纯环境包配方（7.5G，不含 models）。tokenizer 仍缺；`model/` 目录现在是未跟踪状态（不影响 pull）。
