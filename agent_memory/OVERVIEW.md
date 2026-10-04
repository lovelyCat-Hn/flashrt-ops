# 记忆概要（OVERVIEW）

> **给 agent 的读法规约**：需要记忆时**先读本文件**——每条记忆 2~4 句核心 + 「▸ 何时读」。
> 摘要够答题就到此为止；要动代码、复现操作、核实数字，再打开对应细节文件。**不要全量通读记忆目录。**
>
> **维护约定（三步缺一不可）**：新增/修改细节记忆时——① 写细节文件 → ② MEMORY.md 加索引行 → ③ 本文件同步刷新对应摘要段。

## 0. 永远适用的规约（每条会话都要遵守）

- [安装类命令用户亲自执行](galbot-user-runs-install-commands.md) — conda/pip/apt 等安装类命令整理成清单发给用户跑；Claude 只做只读检查和结果判读，不代跑安装。▸ 任何要装东西的场景。
- [本机 GitHub 连接](galbot-github-remote-setup.md) — SSH 账号 lovelyCat-Hn 已配好可用；push 仅限 SSH；ghproxy 镜像会卡死 fetch（已移除**勿加回**）；上游 flashrt-project 无写权限。▸ commit/push/clone 或网络动作报错时。

## 1. 机器识别 + 本机（第三台）部署状态

**机器识别**：三台机 hostname 全叫 galbot-echo，**勿用 hostname 区分**；沟通中称"本机（第三台）"。

- [本机（第三台）部署进度](galbot-machine3-deployment-state.md) — **本机状态权威文件**。L4T R35.6.4，~/holy 即仓库根；闭环工作点 ckpt=pi05_g1_place + 数据集 only_place（101 轨；**起始夹爪 ≈33% 开度是 places 任务语义，非毛刺**），组装三步 + norm_align_check 全绿；闭环 `--horizon` 默认 50；60s+ rollout 瓶颈已定案=执行器配速（"腕部相机全盲"旧判已被用户否决并被实证推翻）；launcher 中途 restart 只起半套栈，必须整机重启。▸ 问"装到哪了/下一步"、改 ckpt/数据集配置前。
- [本机相机 transport 不匹配](galbot-machine3-camera-transport-unmatch.md) — 相机时好时坏，**重启采集栈即愈**（勿直接 kill 采集守护，launcher 会组杀兄弟进程）；失败窗口期相机话题对所有外部进程隐身，embosa 绑定旁路也不通。工具：read_camera_bypass.py / sdk_camera_smoke.py / embosa_topic_tool。▸ 相机话题消失、SDK 取图失败、"刚才还好好的"。

## 2. 环境与推理引擎（FlashRT / pi0.5）

- [pi0.5 环境准备](galbot-pi05-env-setup.md) — **环境红线大全**。Orin sm_87（非 Xavier）；JP5.1.4，CUDA 11.4 不可升；torch 2.4.1 自编 wheel（~32TF bf16，可移植 JP5.1+py3.11，仅 sm_87 SASS）；cuBLAS 三层怪癖→启动入口必挂 cuda_warmup.py 兜底；**部署定档 bf16 终审**（INT8 per-row 判死 + QuaRot 旋转版仿真也判死 tf 0.26，quant想法先过 quarot_sim_ablation.py 零改动仿真；W8A16-decoder 是仅剩无损候选 ~20-30ms 未开工；过程知识：量化在 load 内部执行/PI05_NO_GRAPH 不被消费/编码器每向前 69 次量化调用末层 early-return）；flash_pyrt311 装包必须钉 `numpy==1.26.4`；bashrc 环境变量泄漏三件套（旧 libcurl 毒 HTTPS 有前科）。▸ 装包/编译/量化/网络或 CUDA 怪象排查。
- [FlashRT 环境打包迁移](galbot-flashrt-env-migration-pack.md) — 同路径 tar 解压即用、零编译；软链/editable 随包走；换用户名走干净配方；JP6 不可用；v2 包含权重/norm_stats/tokenizer/scripts。配套 `~/holy/DEPLOY.md` + `verify_deploy.sh` 一键验收。▸ 新机部署/迁移/验收。
- [G1 CUDA graph 段错误修复](galbot-g1-cuda-graph-instantiate-fix.md) — L4T r35.6 iGPU 上旧式 cudaGraphInstantiate 对**任意图**必段错误（单节点 memset 也崩），已修走 WithFlags（FlashRT 4822d755，两机热修都在）；bench 收益仅 ~3%，勿期待翻倍；两机定档分叉：源机 PI05_NO_GRAPH=1，部署机=0。▸ graph 段错误、开/关图模式、跨机性能对比。

## 3. GalbotSDK 与 G1 真机操作

- [pi0.5↔GalbotSDK 集成](galbot-pi05-sdk-integration.md) — SDK 1.8.1 与 FlashRT 同进程 py3.11 已验证（`LD_LIBRARY_PATH`+`PYTHONPATH=/data/galbot/lib`）；相机 API 拿到的是压缩图 CPU 软解（3 路 ~20ms），深度 16UC1 免解；臂 7 关节两套 API（关节空间 / 任务空间 GalbotMotion）；libero (10,7) 动作与 G1 非恒等映射。▸ 写集成代码、取图、选 SDK API。
- [SDK group 模式读取顺序错位](galbot-sdk-group-mode-ordering-pitfall.md) — group 模式读关节与 names 配对会全错；**读关节必须按名字显式读取**。▸ 读关节值出现莫名错位。
- [GalbotMotion 帧命名空间坑](galbot-motion-frame-namespace-pitfall.md) — IK/GET/SET 三套 API 帧名互不通用；set_end_effector_pose 要传链名+显式 `Parameter()`；桩文件不可信。▸ 任务空间运动、IK 结果莫名偏。
- [jetson_ik_move 工具](galbot-jetson-ik-move-tool.md) — IK 交互执行器用法；leg 只能前后/升降（y 是假解坑，配 FK 残差自检）；四元数顺序 [qx,qy,qz,qw]。▸ 手动挪机器人到某位姿/写腿动作。
- [G1 PVT 轨迹接口危险实录](galbot-g1-pvt-trajectory-hazard.md) — 零运动探针也致剧烈抖动，**traj/PVT 路线已封存**；提速加剧抖动（每点全停，冲击∝速度）；平滑走 track+合步。▸ 想走轨迹点路线时（答案：别）。
- [G1 数字孪生](galbot-g1-digital-twin-setup.md) — Jetson 推流 + Windows MuJoCo 渲染（jetson_sender.py TCP:9999 @30Hz）；只 mj_forward 不 mj_step；新 clone 的 XML 要 sed 修 `inertia="shell"`；LD_PRELOAD libgomp。▸ 数字孪生/MuJoCo 可视化。

## 4. 建图与导航

- [G1 建图/定位栈要点](galbot-g1-slam-mapping-stack.md) — FAST-LIO 风格 IESKF；engine_tools 菜单 1 存图 / 4+5 增量更新 / 2 发初始位姿；定位 Score≥0.8 看日志；updata_maps 工具集；bin 质心可判推车覆盖。▸ 建图、增量更新、查图质量。
- [G1 导航栈排障](galbot-g1-navigation-stack-troubleshooting.md) — pns 硬读 `/var/maps/cur/global_cloud_cleaned.pcd` 软链，缺则初始化死循环全拒（**换图必补**）；关节读不到=RT/急停侧（wbcs_test 诊断）；**kill launcher 管的服务会组杀兄弟进程且不再重启，恢复靠冷断电**；幽灵障碍坑（推车人影/地坪 multipath 伪点→start state collision）；换图五步标准流程。▸ 导航被拒、换图、关节数据断流。
- [G1 底盘朝向与导航 UI](galbot-g1-chassis-frame-nav-ui.md) — **车头=yaw 直接所指**（FRONT_OFF=0；曾误判 −x 已回退，此类问题一律以用户最新现场观察为准）；实时 2D 导航工具点障碍/膨胀区/图外一律拒绝、目标点不吸附；SDK 残留线程卡退出→`os._exit`。▸ 底盘导航、朝向争议、脚本退不死。

## 5. 数据集与模型判读

- [G1 微调数据集判读](galbot-pi05-g1-finetune-data.md) — pick_place_balence 16/23 维**右臂在前**（以 info.json 为权威，记反会把臂甩背后）；800ms 慢推理已破案=state 文本进了 prompt，`FLASHRT_PI05_STATE_PROMPT_MODE=fixed` 根治；闭环 5 轮全绿，残留指令切换抖动待调。▸ 数据集维度判读、推理忽慢、抖动排查。

## 6. 闭环执行器（run_g1_loop 调参）

- [G1 闭环速度调参定案](galbot-g1-loop-pace-tuning.md) — **统一速度律：臂速÷数据集原速 = n÷(30×pace×div)**，n=steps-per-round；pace 焊死 0.455（下限=推理水位，**只花不赚**，0.56 实测更差）；div 双语义（到达时刻 div×pace、覆盖率 1/div、残差≈导程×(1−1/div)，兼滞后滤波）；工作点候选 20/1.55（0.95× 原速）或 25/1.8（1.02×）；梦游期铁律（任务 ≈390 chunk 步、**落盒立刻 q 停**、梦游期日志零判别力）；chunk 边界语义速查（合步不跨 chunk、抽帧在发令瞬间、spc>n 静默钳制、spc=n 单指令=无复位解）。▸ 调闭环速度、判读轮日志、归因复位/停走/摆动。

## 7. 仓库配套文档（不在记忆目录，在 ~/holy）

- `~/holy/DEPLOY.md` — 新机部署指南（纯环境包配方 v3，7.5G 不含 models）
- `~/holy/USAGE.md` — 操作手册（装机 + 日常使用）
- `~/holy/BENCHMARKS.md` — 基准报告（对外汇报用，2026-09-21 echo 机全部实测数据）
- `~/holy/scripts/` — 2026-10-04 分层：`inference/` 直接入口（run_g1_loop / run_g1_inference / g1_config.py 等，共享 config/g1.toml）、`g1/` 整备标定、`dataset/` 数据集工具、`eval/` 精度评估、`test/` 性能诊断、`probes/` 探针封存（g1_traj_probe 等）
