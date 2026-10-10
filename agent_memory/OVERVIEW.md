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

- [本机（第三台）部署进度](galbot-machine3-deployment-state.md) — **本机状态权威文件**。L4T R35.6.4，~/holy 即仓库根；**10-08 换新 pick 权重 only_pick 在役**（pi05_g1_onlypick_deploy，用户重训 0930 job、新数据集不在本机；stats 曾现绝对角量级疑云→判明框架挪了取样位置、targets 仍 delta，只读探针已验）；place 三键注释存档可切回（33% 持物起始位姿，包络护栏从 only_place 派生）；首跑被护栏拦停已修（护栏曾拿 delta 比绝对盒，6891103）；**工作点 v3=25/0.35/1.5 已入 config（1.59×，前置 --nav-suspend，R 落盒 10.0s 纪录，见第 6 节）**；闭环 `--chunk-size` 默认 50（10-08 前名 --horizon）；**10-09 native 固定节拍执行器（run_g1_loop_native）+ FlashRT guided RTC 落地（严格 opt-in 默认关），真机调参七跑定案工作点 `--cmd-every 6 --arrive-div 2 --rtc-horizon 10`（五块体感不抖）；同日轨迹流执行器 run_g1_loop_traj 落地（set_joint_commands 8帧/行突发，同事 infer_policy 在役路线，干跑全绿待真机首跑），60 轮 A/B 待跑（USAGE §3.7/§3.8）**；launcher 中途 restart 只起半套栈，必须整机重启；⚠ 全天空爪，持物物理 place 复验挂着；**10-10 traj 线抖动定性三件套落地待跑**（--dump-chunks→增量谱分析→回放台控制器隔离，traj 在役配方=EMA off+reserve 0+delta-max 0.05+rtc-horizon 10 必带；**10-10 蠕动破案=场景 OOD（搬回原工位即活；模型不动先 diff 相机画面 vs 数据集抽帧再怀疑执行链）+ 健康增量谱定案：delta-max 0.05 对原始增量 0.00% 截断=封版，接缝主源=换块重规划失配 p50 135/max 348 mrad；chunk_increments.py 爪维打 mrad 标签=量纲坑（爪维是 0-100 百分比）；**10-10 晚定盘：240Hz traj 线退役、30Hz 口径为准（回放 A/B direct tfs0 滞后 2.1×；direct+tfs33.3 滞后差但手感顺滑、接缝停顿=回放台 hold 假象；30Hz 平滑候选=native 6/2 或逐拍 tfs33.3 待 native 补丁；**终局：回放 A/B 结案（tfs33.3 stall 判死、纯 lerobot=tfs0、inference_latency=服务端节流非电机参数）→ 水位 10 破局跑定在役配方终版 traj 240Hz 流+rtc10/delta0.05/reserve0/n-action-steps 10（接缝塌缩、残差 12.6 全程最佳、R 爪首次真动作；观测新鲜度=接缝第一杠杆；⚠ 这些数是 8 帧流口径）**）；**10-10 深夜：往复摆动归因完成=模型输出抖动实锤**（回放复现摆动→源在数据路径；A1 模型层翻转 108.9/s R 臂≈50%@2.7 mrad、A2 构建层仅 +8% 透传、臂随动 B2 同关节翻转、E 接缝 1.20× 免责）；**240Hz 微流曾退役（默认改 1），**后经 place 干净对照推翻=8 帧流留任**（1帧 残差 32.3/接缝 22.3 vs 8帧 18.5/14.3；执行器已到地板，剩=模型抖动，下一杠杆 EMA A/B）；在役配方=traj+`--frames-per-row 8 --rtc-horizon 10 --delta-max 0.05 --burst-reserve-ms 0 --n-action-steps 10`（place 不带 --grip-state-cmd，pick 必带）；⚠ --fps 勿带（15Hz=时间维 OOD）**；**10-10 深夜 pick 双破案**：场景 OOD=托盘比训练近一倍（数据集全远位）+ **视觉依赖度探针实锤 only_pick 视觉-臂接地微弱**（全黑图臂维≈噪声级、爪维 10.9 点=爪相位视觉驱动；固定点位伸手机制坐实，部署链全链无罪，修复在训练侧——排查序=dataloader 图像映射/视觉塔 freeze/同事 lerobot 配方对照；探针 scripts/probes/vision_dependence_probe.py，证据 evidence/20261010_pick_aim_diff/）**；**深夜三：pick 过渡工作点=15/15/0.35/1.6（0.89×）首次偏位抓取成功**（视觉弱非零，慢速重规划累积转向；near-div 无锁存、占比是速度档函数非 bug；近阻尼=滤抖功臣，代价=持物爬行 R 被饿）**。▸ 问"装到哪了/下一步"、改 ckpt/数据集/工作点配置前、traj 抖动排查时、pick 夹空/不追踪物件时。
- [本机相机 transport 不匹配](galbot-machine3-camera-transport-unmatch.md) — 相机时好时坏，**完整重启采集栈即愈**（10-07 第三次应验；勿直接 kill 采集守护，launcher 会组杀兄弟进程）；失败窗口期相机话题对所有外部进程隐身，embosa 旁路也通不了（健康代旁路能配对但**载荷 0 字节**——相机图像不走裸 reader 通路，旁路只配判"隐身 vs 仅 SDK"）；`transport no support` 行六代守护恒 4 行=启动噪声，非判别信号。工具：read_camera_bypass.py / sdk_camera_smoke.py / embosa_topic_tool。▸ 相机话题消失、SDK 取图失败、"刚才还好好的"。

## 2. 环境与推理引擎（FlashRT / pi0.5）

- [pi0.5 环境准备](galbot-pi05-env-setup.md) — **环境红线大全**。Orin sm_87（非 Xavier）；JP5.1.4，CUDA 11.4 不可升；torch 2.4.1 自编 wheel（~32TF bf16，可移植 JP5.1+py3.11，仅 sm_87 SASS）；cuBLAS 三层怪癖→启动入口必挂 cuda_warmup.py 兜底；**部署定档 bf16 终审**（INT8 per-row 判死 + QuaRot 旋转版仿真也判死 tf 0.26；10-06 复核闭环：旋转实现无 bug、病位=层0→1、通道静态 fold 端到端更差——崩的入口恰好是不能静态化的位点，quant 想法先过 rot_equivalence_probe.py/quarot_sim_ablation.py 零改动仿真；过程知识：量化在 load 内部执行/PI05_NO_GRAPH 不被消费/管线 weights 存设备指针惰性建/编码器每向前 69 次量化调用末层 early-return）。**W8A16-decoder 已落地（10-07 定案，8f08ca85）**：sm_87 手写 weight-only INT8 kernel（激活全程 bf16），`FVK_PI05_RTX_W8A16_DECODER=1` 默认关；微基准 M=10 每层 −49%、闭环 chunk50 −5-6ms / chunk10 −10ms、质量近无损（cos_arm 1.0000/0.9997，dec8 的夹爪 bias 被消除）；graph-on 冒烟已过（p50 229.9ms，vs bf16 基线 −9.4%）；**SDK 在场现场验证 −18.8ms**（冻结帧 375.3→356.5，争用环境兑现反超空载=带宽越紧越值）；loop 388 归因闭合（+138=SDK/相机栈常驻、+13=并行取图；汇报三口径 229.9 空载/375 SDK 在场/388 loop）；模块拆解见相位探针（编码器 133ms 占 53% 是下一个靶）；⚠ 重编 .so 必须 flash_pyrt311 环境 nvcc 11.8（系统 11.4 死路）+ TU-split 头文件模式；kernel patch+README 在 hotfix_flashrt/flashrt_kernel_patch/；残留：graph-on 质量闭环未验、h10 档兑现疑点。flash_pyrt311 装包必须钉 `numpy==1.26.4`；bashrc 环境变量泄漏三件套（旧 libcurl 毒 HTTPS 有前科）。▸ 装包/编译/量化/网络或 CUDA 怪象排查。
- [FlashRT 环境打包迁移](galbot-flashrt-env-migration-pack.md) — 同路径 tar 解压即用、零编译；软链/editable 随包走；换用户名走干净配方；JP6 不可用；v2 包含权重/norm_stats/tokenizer/scripts。配套 `~/holy/DEPLOY.md` + `verify_deploy.sh` 一键验收。▸ 新机部署/迁移/验收。
- [G1 CUDA graph 段错误修复](galbot-g1-cuda-graph-instantiate-fix.md) — L4T r35.6 iGPU 上旧式 cudaGraphInstantiate 对**任意图**必段错误（单节点 memset 也崩），已修走 WithFlags（FlashRT 4822d755，两机热修都在）；bench 收益仅 ~3%，勿期待翻倍；两机定档分叉：源机 PI05_NO_GRAPH=1，部署机=0；本机（第三台）9/30 解封——run_* 四脚本 env 门控默认开图（PI05_NO_GRAPH 在 FlashRT 零消费者，封图全靠脚本 monkeypatch），**闭环 60 轮已验**（p50 380ms）。▸ graph 段错误、开/关图模式、跨机性能对比。
- [cache_frames=2 判死](galbot-pi05-cache-frames-dead.md) — 时序 K/V 复用对 G1 闭环结构性无效：state 进 prompt → 每轮 set_prompt 重置前缀计数 → 永远全量（重置是正确性所需，绕过=拿陈旧本体状态规划）；bench 3.8× 只属于 prompt 恒定的 droid 基准。knob 默认 1 保留，勿再试。▸ 又想拿 K/V 复用提速时。

- [FlashRT 跨构造位等不可比](flashrt-cross-build-bit-compare-pitfall.md) — **每次前端构造重跑 GEMM autotune 选不同算法**（计时选型，两次构造 best 序号不同）→ 浮点累加序不同 → 输出位不同（未归一化 max≈0.35/mean 0.011）。位等断言/探针只能在**同构造内**做；跨构造用 allclose(≥0.5) 或 cos。10-09 RTC off 回归测试连挂三次全是它，不是泄漏。▸ 写位级回归测试、跨进程对比输出、判"引擎是不是被改坏了"。

## 3. GalbotSDK 与 G1 真机操作

- [pi0.5↔GalbotSDK 集成](galbot-pi05-sdk-integration.md) — SDK 1.8.1 与 FlashRT 同进程 py3.11 已验证（`LD_LIBRARY_PATH`+`PYTHONPATH=/data/galbot/lib`）；相机 API 拿到的是压缩图 CPU 软解（本机 10-07 实测 3 路 224 软解+23 维关节+归一化合计 8.3ms），深度 16UC1 免解；臂 7 关节两套 API（关节空间 / 任务空间 GalbotMotion）；libero (10,7) 动作与 G1 非恒等映射。▸ 写集成代码、取图、选 SDK API。
- [SDK group 模式读取顺序错位](galbot-sdk-group-mode-ordering-pitfall.md) — group 模式读关节与 names 配对会全错；**读关节必须按名字显式读取**。▸ 读关节值出现莫名错位。
- [SDK 正主文档位置与 API 语义](galbot-sdk-doc-real-source.md) — **语义问题先查 `project_chn/GalbotSDK` 的 C++ 头文件**（部署 .pyi 是残缺版，勿据其判"未文档化"）；set_joint_commands=官方流式推荐（首令不插值、tfs=expected arrival time=标准关节唯一配速输入、0=fastest）、set_joint_positions=官方明示不适合流式（v3 的位）、batch 多点+Kp/Kd 未测。▸ 写/查 SDK 下发代码、判 API 行为前。
- [GalbotMotion 帧命名空间坑](galbot-motion-frame-namespace-pitfall.md) — IK/GET/SET 三套 API 帧名互不通用；set_end_effector_pose 要传链名+显式 `Parameter()`；桩文件不可信。▸ 任务空间运动、IK 结果莫名偏。
- [jetson_ik_move 工具](galbot-jetson-ik-move-tool.md) — IK 交互执行器用法；leg 只能前后/升降（y 是假解坑，配 FK 残差自检）；四元数顺序 [qx,qy,qz,qw]。▸ 手动挪机器人到某位姿/写腿动作。
- [G1 PVT 轨迹接口危险实录](galbot-g1-pvt-trajectory-hazard.md) — 零运动探针也致剧烈抖动，**traj/PVT 路线已封存**；提速加剧抖动（每点全停，冲击∝速度）；平滑走 track+合步。▸ 想走轨迹点路线时（答案：别）。
- [G1 臂控制模式与安全护栏](galbot-arm-control-mode-safety.md) — **臂=刚性位置伺服**（set_joint_positions 硬路由 id1 非柔顺，SDK 无切换入口）；**臂端 fault/堵转不回传恒 SUCCESS**；残差急停不可行（特征重叠）；压桌防护=关节包络护栏（d26f9ce，数据集构型盒截断；⚠ 10-07 修范畴错误——比对对象必须=BASE_ARM+delta 绝对构型，直接比 delta 必拦停，冒烟"0 误报"是绝对自比无效）；Motion.init 脱机挂死>120s FK 死路。▸ 臂安全、撞桌/fault、控制模式切换问题。
- [G1 数字孪生](galbot-g1-digital-twin-setup.md) — Jetson 推流 + Windows MuJoCo 渲染（jetson_sender.py TCP:9999 @30Hz）；只 mj_forward 不 mj_step；新 clone 的 XML 要 sed 修 `inertia="shell"`；LD_PRELOAD libgomp。▸ 数字孪生/MuJoCo 可视化。

## 4. 建图与导航

- [G1 建图/定位栈要点](galbot-g1-slam-mapping-stack.md) — FAST-LIO 风格 IESKF；engine_tools 菜单 1 存图 / 4+5 增量更新 / 2 发初始位姿；定位 Score≥0.8 看日志；updata_maps 工具集；bin 质心可判推车覆盖。▸ 建图、增量更新、查图质量。
- [G1 导航栈排障](galbot-g1-navigation-stack-troubleshooting.md) — pns 硬读 `/var/maps/cur/global_cloud_cleaned.pcd` 软链，缺则初始化死循环全拒（**换图必补**）；关节读不到=RT/急停侧（wbcs_test 诊断）；**kill launcher 管的服务会组杀兄弟进程且不再重启，恢复靠冷断电**；幽灵障碍坑（推车人影/地坪 multipath 伪点→start state collision）；换图五步标准流程。▸ 导航被拒、换图、关节数据断流。
- [G1 底盘朝向与导航 UI](galbot-g1-chassis-frame-nav-ui.md) — **车头=yaw 直接所指**（FRONT_OFF=0；曾误判 −x 已回退，此类问题一律以用户最新现场观察为准）；实时 2D 导航工具点障碍/膨胀区/图外一律拒绝、目标点不吸附；SDK 残留线程卡退出→`os._exit`。▸ 底盘导航、朝向争议、脚本退不死。

## 5. 数据集与模型判读

- [G1 微调数据集判读](galbot-pi05-g1-finetune-data.md) — pick_place_balence 16/23 维**右臂在前**（以 info.json 为权威，记反会把臂甩背后；9/30 时间线探针再证，USAGE.md 旧文档写反已纠）；800ms 慢推理已破案=state 文本进了 prompt，`FLASHRT_PI05_STATE_PROMPT_MODE=fixed` 根治；only_place 101 轨时间线：L 释放 3.7s → R 启动 +3.4s，**双臂严格串行**；**换新训练 ckpt 必查 action stats 量级对照前代**（10-08：框架版本会挪 stats 取样位置，only_pick 绝对角疑云已破案=targets 仍 delta；判定法与 out_proj bias 无效教训在案）；**10-10 重大：only_pick 视觉-臂接地微弱实测**（探针实证臂不靠视觉转向、爪相位靠视觉；训练侧排查序=dataloader 图像映射①/视觉塔 freeze②/同事配方对照③）。▸ 数据集维度判读、推理忽慢、抖动排查、换 ckpt 前、模型不看物件时。

## 6. 闭环执行器（run_g1_loop 调参）

- [G1 闭环速度调参定案](galbot-g1-loop-pace-tuning.md) — **统一速度律：臂速÷数据集原速 = n÷(30×pace×div)**，n=--n-action-steps（10-08 前名 steps-per-round）；**工作点 v3（10-07 晚定案）= spc25/pace 0.35/div 1.5（1.59×，前置 --nav-suspend，R 落盒 10.0s 纪录；代价=残差 p50 79/8 段≥100，持物复验前保留观察）**；v2 25/0.38/1.65 史档（1.33×，hold 兜底时代）；pace/div 的"死 knob/饱和"结论是水位函数（**水位≈0.34-0.35 崖边勿再压**）；spc=唯一结构杠杆（15 反例 0.95× 78 轮未落盒；30+ 勿试）；div 双语义（覆盖率 1/div、滞后滤波）；夹爪 0.15 m/s；**q 停=R 落盒才停（严格串行）**；梦游期签名+肥尾（spc20 出过 32.8s 大发作）；坡升保险丝 --hold-ramp（旧名 --rtc-ramp；≥2 窗 hold 才触发）；chunk 边界语义速查；**导航栈 SIGSTOP 搁置杠杆（10-07 定案）= −55ms 且 GR3D 50→7%，真机 hold 窗清零已验、催生 v3**；**10-09 native 孪生 `run_g1_loop_native.py` 落地+真机七跑定案**（固定 tick=1/30s 消 1 条 action、速度涌现 ≈1.0× 原速、水位换块弃尾、星饿=冻结停机；速度律对它无意义，v3 仍是现役默认；**工作点=cmd-every 6/arrive-div 2/rtc-horizon 10：抖感∝指令沿频率(30Hz≫10Hz>5Hz)、div 骑到点边界必锯齿、const-speed 证伪=配速律是防到点机制、SDK speed_rad_s=速度上限非配速**；接缝尖峰遥测=切换抖动量化，RTC on 五连绿全<块内均值；60 轮三态待跑）。▸ 调闭环速度、判读轮日志、归因复位/停走/摆动、跑 native 对照。

## 7. 仓库配套文档（不在记忆目录，在 ~/holy）

- `~/holy/DEPLOY.md` — 新机部署指南（纯环境包配方 v3，7.5G 不含 models）
- `~/holy/USAGE.md` — 操作手册（装机 + 日常使用）
- `~/holy/BENCHMARKS.md` — **基准数据报表**（**定位=论文级数据报表：只收数据表+测量条件+复现命令，零判读**；判读一律进记忆库。写作规范=**`.claude/skills/bench-report/SKILL.md`**：探针 stdout 必须当场落盘 `evidence/YYYYMMDD_主题/`、表格自足/单位进表头/口径注脚、新数据先落盘再进表）。2026-10-08 起：速览表置顶+§一延迟+§二模块耗时+**§三量化消融（含 10-06/07 逐层 K/V cos 表 int8/rot8、chan8/chan8all 按通道、dec8/w8a16 sanity，原始件在 `evidence/20261006-07_quant_kv_ablation/`）**+§四 W8A16+§五导航搁置+§六SDK 遥测+§七闭环工作点+**§八 RTC 引导消融（10-09，接缝 p50 57.2→20.7 mrad，原始件 `evidence/20261009_rtc_guided_ablation/`）**+§九复现索引；echo 机 09-21~09-24 旧数据与已推翻结论在 `BENCHMARKS_ARCHIVE.md`（顶部 ⚠ 标注）
- `~/holy/scripts/` — 2026-10-04 分层：`inference/` 直接入口（run_g1_loop=闭环定案入口（**10-08 命名正规化更名**，原 run_g1_loop_rtc；旗名/toml 键对照表=`docs/lerobot-alignment.md`）；**10-09 增 native 固定节拍孪生 run_g1_loop_native.py**（lerobot 调度对照，术语表 §九）/ run_g1_inference / g1_config.py 等，共享 config/g1.toml；原 A 脚本存档 `legacy/run_g1_loop_stale.py`）、`g1/` 整备标定、`dataset/` 数据集工具、`eval/` 精度评估（**10-09 增 rtc_ablation.py**）、`test/` 性能诊断、`probes/` 探针（g1_traj_probe / int8 三件套 / rot_equivalence / module_timing（10-08 从 transcript 抢救回）/ sdkfree）；`config/README.md` = g1.toml 导读
