# S1 维护手册（2026-10-27 建档）

> 本机 = Galbot **S1 系列**（第四台 Orin 机；前三台——源机/部署机/machine3——均为 **G1**，
> 它们的记忆与基准在 `agent_memory/`，**G1 本体层结论勿直接照搬到 S1**）。
> 机型差异只影响本体层（关节/相机/动作语义）；Orin 计算环境四台已验证同构（sm_87 + JP5.1.x + cuda-11.4）。
> 关节/API 速查见 [docs/s1-joint-api-quickref.md](docs/s1-joint-api-quickref.md)。

## 0. 本机档案

| 项 | 值 |
|---|---|
| 机型/序列号 | S1 系列，SN=R002AAFF21AA0063（robot_info 实读） |
| 硬件 | HW 2.1，BODY_TYPE=A，左/右臂 **A21**，swerve 舵轮底盘 |
| 计算 | Jetson AGX Orin Developer Kit，L4T **R35.6.0**，cuda-11.4，galbot@/home/galbot |
| SDK | **1.10.0**（`/data/galbot/lib/libgalbot_sdk.so.1.10.0`）＝ 源码包 `~/GalbotSDK` 版本（已核） |
| 仓库 | `~/holy` 即仓库根，remote = `git@gitee.com:tq-llm/flashrt-ops.git`（账号 holy@holydd，SSH 2026-10-27 新配） |

## 1. 环境部署状态（2026-10-27）

- v4 免热修环境包已传/已验/已解压：sha256 与 DEPLOY.md §1 逐字一致（7,562,700,800 字节）。
- `apply_hotfix.sh` 报「已在」= no-op（v4 含全部本地补丁 + W8A16 编译产物，符合预期）。
- `verify_deploy.sh` **PASS=5 FAIL=1**：①② 前提/位置全绿；③ 环境链核心全绿
  （torch `2.4.0a0+gitee1b680` sm_87 (8,7) / flash_rt 0.2.0 / kernels ok / numpy 1.26.4 / cv2 4.10.0 / 双本地补丁 ok）；
  唯一 FAIL = norm_stats 加载——**权重未就位所致的预期失败**（v4 按设计不带权重），④⑤ 自动跳过。
- 包缓存 `~/flashrt_bundle_v4_env.tar`（7.1G）校验已过，确认无误后可删。

## 2. 日常维护 SOP

```bash
# 代码轨更新（scripts/文档以 git 为准；环境轨 tar 之外勿 pip/conda 装包）
cd ~/holy && git pull
# 一键验收
bash ~/holy/verify_deploy.sh
# FlashRT 本地改动后（热修包以 ops 仓库为准的规矩，见 DEPLOY.md §2.1）
# → 同步刷新 ~/holy/hotfix_flashrt/ 对应文件并 commit 推 git
```

- 基准/探针数据 → `BENCHMARKS.md`（规约见 repo skill `bench-report`），证据件落 `evidence/<日期>_<主题>/`。
- agent 记忆：本机记忆在 `~/.claude/projects/-home-galbot/memory/`；重大变更后整目录拷回
  `agent_memory/` 刷快照并 commit（OVERVIEW 顶部「维护约定」三步：细节文件 → MEMORY.md 索引 → OVERVIEW 摘要）。

## 3. S1 特有事项

- **关节对齐**：一切以 `docs/s1-joint-api-quickref.md` 为速查基准——19 维布局与限位（源 = 官方
  `/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml`）、torso 组、独立夹爪 API。
- **脚本位置约定**：S1 专用脚本统一写 `scripts/s1/`（2026-10-27 起，见该目录 README），G1 系脚本勿混放。
- ⚠ **安全红线（2026-10-27 手臂掉落事件）**：模式切换/接管键瞬间 WBCS 控制权异常 → 手臂失力矩坠落；
  切换前后手不离急停，勿让手臂大范围悬空久停（详见 `~/yuqiz/s1_collect/README.md`）。
- **SDK 1.10.0 与 G1 1.8.1 的 API 差异**：`get_rgb_data` 已改版（勿搬 G1 记忆里的取图代码）；
  新增 `emergency_stop`/`get_base_velocity`/`motion_plan` 家族等（quickref §1）。
- **遥操/数采链路**：galbot_upper(APP) 切 TELEOPERATION_MODE → hpu_launcher → singorix_wbcs_main；
  数采切 DATA_COLLECTION_MODE。同事真机代码在 `~/yuqiz/{s1_teleop,s1_collect}`。

## 4. 待办（按优先级）

1. 拷在役权重到 `~/holy/models/pi05_g1_onlypick_deploy/` → 重跑 verify_deploy.sh
   （④⑤ 真冒烟 + **graph 探测在本机 r35.6.0 的首测**，结果记入 BENCHMARKS）。
2. 关节只读探针：`get_joint_names` / `get_joint_positions` / `get_gripper_state` 实测，
   **裁决头部限位口径**（teleop TOML ±0.7592/±0.5847 vs SDK 示例注释 ±0.7854/±0.6109）。
3. 夹爪物理宽度 ↔ 百分比换算标定（行程 0.007–0.11 m，反馈单侧 0–0.055 m；yuqiz 参考值 `gripper_action_scale=0.0011`）。
4. DEPLOY.md §6 新机清单：相机视角映射 / 动作空间开环观察（S1 19 维与 G1 23 维不同，映射要重推）。
5. 配置 repo 级 git 身份（`git config user.name/user.email`，本机尚未设置）。
6. `agent_memory/` 快照缺件：`galbot-sdk-doc-real-source.md` 被 MEMORY.md 索引引用但本 clone 没有——从 G1 机补推。

## 5. 材料索引

| 材料 | 位置 |
|---|---|
| 关节/API 速查 | `docs/s1-joint-api-quickref.md` |
| SDK 源码+文档（1.10.0） | `~/GalbotSDK/`（examples/s1/python 含 tutorials/example5_execute_vla.py） |
| 官方限位/PVT 配置 | `/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml` |
| 真机跑通的遥操/采集 | `~/yuqiz/`（19 维布局/限位/夹爪标定出处） |
| 厂商 singorix 文档 | `/data/galbot/share/galbot/singorix_example/docs/` |
| agent 记忆 | `agent_memory/`（仓库快照）；本机工作记忆 `~/.claude/projects/-home-galbot/memory/` |
