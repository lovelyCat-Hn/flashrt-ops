---
name: galbot-s1-sdk-and-body-basics
description: S1 机 SDK/本体要点：SDK 1.10.0（galbot_sdk.s1，MachineType.S1）、状态 19 维/关节组含 torso、夹爪行程 0.007–0.11m、厂商 singorix 文档在 /data/galbot/share、同事可跑源码在 ~/yuqiz
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-28T06:57:28.073Z
---

**S1 SDK/本体基线（2026-10-27 盘点，机器 SN=R002AAFF21AA0063，ROBOT_SERIES=S1，HW 2.1，BODY_TYPE=A，双臂 A21）**。与 G1 线的关键差异（来源 [[galbot-s1-machine-state]] 机型差异的落地细节）：

- **SDK 版本**：本机 `/data/galbot/lib` = galbot_sdk **1.10.0**（G1 三台是 1.8.1），专用实现 `GalbotRobotS1`。pybind 全版本绑定（cp38~314，**含 3.11** → flash_pyrt311 可直接 import）；模块有 `s1`/`g1`/`g3` 三个子包，**用法 `galbot_sdk.s1` + `MachineType.S1`**。`s1/__init__.pyi` 类型存根在 `/data/galbot/lib/galbot_sdk/s1/`——完整 API 面可直接读，无需外部文档。
- **关节组差异**（同事 yuqiz 的 G1→S1 移植实测）：G1 = left/right_arm + leg + head → **S1 = left/right_arm + torso + head**；**状态 19 维**（双臂14 + 双夹爪2 + 躯干1 + 头2），G1 是 23 维。**夹爪行程 S1 = 0.007–0.11 m（反馈 0.055）**，G1 是 0–0.12 m（反馈 0.06）——pi0.5 夹爪标定/归一化不能照搬。
- **控制栈**：S1 走 **singorix WBC（全身控制器）**，target 话题 `singorix/wbcs/target`；主从臂遥操链 = galbot_upper(APP) 切 TELEOPERATION_MODE → hpu_launcher → singorix_wbcs_main → 主臂硬件写 wbcs/target。数采集要切 DATA_COLLECTION_MODE。G1 的 joint/PVT 那套接口不直接适用。
- **本地资料（无需下载 SDK 包）**：① 厂商文档 ~30 份 markdown 在 `/data/galbot/share/galbot/singorix_example/docs/`（`singorix_s1_config_reference_guide.md`、`singorix_nx1_python_sdk_guide.md`、quick_start、dexhand、endtool、chassis pose、PVT bypass 等）；② 同事源码 `~/yuqiz/s1_teleop/`（13KB s1_teleop.py + 配置）+ `~/yuqiz/s1_collect/`（采集/复位/读位姿/镜像/goto 全套，G1 without_hil 移植版，离线转 LeRobot v3，profile=s1_pick_3cam_14d/s1_place_4cam_21d）；③ 当日 evidence `~/yuqiz/evidence_20261027/`。
- **⚠ 安全教训（2026-10-27 手臂掉落事件，已写进 yuqiz README）**：切换模式/按接管键前后手臂可能瞬间失去力矩保持（WBCS 控制权切换异常→重力坠落）；接管时另一只手放急停旁；勿让手臂长时间大范围悬空。SDK 日志另有 endtool info 反复超时（`~/galbot_sdk_log/s1/`，左右端工具 info 不可用告警，原因待查）。
- **G1 记忆的适用性**：galbot-sdk-group-mode-ordering-pitfall（显式名字模式防乱序）、arm-control-mode-safety 等方法论仍适用；但关节名/组名/维数/行程全部要按 S1 重核，s1_collect 的 config（复位位姿等）是现成起点。

**SDK 源码包已到位（2026-10-27 下午，用户下载）**：`~/GalbotSDK/` = **1.10.0**，与本机 runtime 版本逐字匹配（VERSION 文件核对）。含 `examples/s1/python/`（galbot_robot/motion/navigation/perception/tutorials 六组，**tutorials/example5_execute_vla.py = VLA 集成模板**、example6_pick_and_place）、`docs/s1/zh/`（mkdocs HTML：api_python_reference/set_config_reference/tutorials）、`galbot_sdk/linux-aarch64-gcc1330/include/.../galbot_sdk_type.hpp`（C++ 头注释=API 语义权威）。`~/lerobot` 也已克隆（machine3 同款）。

**S1 关节速查表已入仓**：`~/holy/docs/s1-joint-api-quickref.md`（随 67655e7 已推 gitee）。核心事实：19 维布局/限位表出自 yuqiz config.py，源头=官方 `/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml`；**左右臂 j2/j4 限位不对称**（[-1.9808,1.7191] vs [-1.7191,1.9808]，映射/包络必须左右各建）；**头部限位两处口径不一致**（SDK 示例注释 ±0.7854/±0.6109 vs TOML ±0.7592/±0.5847）待实测定案；torso 限位 [0,1.0]；夹爪状态维 0–100% 与 pi05 同口径、物理行程 0.007–0.11 m 换算待标定；`set_joint_positions` 签名与 G1 相同，`set_joint_commands` 高频接口（100Hz 示例）+ `set_joint_commands_batch` 是 traj 执行器候选。另有 repo 级 skill `bench-report`（维护 BENCHMARKS.md 用）随 holy/.claude 就位。

**⚠ torso 运行时实名（2026-10-28 实测，get_joint_names 实读）**：= **`torso_lift_joint1`**，quickref/config 文档写的 `torso_joint1` 与 SDK 返回不符——按文档名读必 KeyError。执行器/限位表/状态布局常量已按实名修正（policies/base.py），quickref 已注 discrepancy。

**⚠ torso = 升降柱，位置语义 = 高度米非 rad（2026-10-28 用户提醒 + 官方实证）**：S1 与 G1 躯干本质不同——G1 无 torso（leg5 旋转关节），S1 torso 是**升降柱**，位置/限位 [0.0,1.0] 单位=**米**（升降高度）。证据：官方 `GalbotSDK/examples/s1/cpp/galbot_robot/src/request_target_example.cpp:268` `kTorsoJt = 0.60; // torso target height, meter` + yuqiz 位姿 torso 0.65/0.75 实证。勿按 rad 处理；限位表注释/S1Targets 文档/quickref 已全部改注 m。

**通用 S1 推理执行器已落地（2026-10-28，ops d03af19）**：`scripts/s1/inference/run_s1_loop.py`（policy 无关 + `policies/` 适配器解耦，详见 [[galbot-fineart-vla-flashrt-port]] ④段与 `scripts/s1/inference/README.md`）。dry-run 全链已验证的读路径：`get_joint_names(only_active_joint=True, joint_groups=[...])` + `get_joint_positions(joint_names=显式列表)`（group 模式坑同 G1）+ `get_gripper_state("<side>_gripper")`（String 名，width 属性）；**下发路径 dry-run 未触发，真机 --exec 前属未验证面**；`--exec` 现被 needs_mapping=False 结构性拒绝（ALOHA 14/G1 16→S1 19 语义映射未定义）。

**流式执行路线定案（2026-10-28 用户纠正，同日改毕）**：S1 执行器与 G1 traj 同哲学=**流式适配 lerobot 动作块**——`set_joint_positions` 勿用于执行（官方明示不适合流式），走 `set_joint_commands(_batch)`：batch=非阻塞多点轨迹，docstring 明示适用 "VLA inference batch output"；`Trajectory.joint_names` 优先于 groups；每点 `time_from_start_second`=相对轨迹起点累计时刻（官方示例 dt=0.008≈125Hz）；单点 `set_joint_commands` 高频口（标准关节只吃 position，**爪关节 position=宽度米且 velocity/effort 生效**——爪未来可并入流式待验）。⚠ G1 tfs 血泪史（tfs0 每点全停滞后 2.1×、tfs33.3 stall 判死）→ S1 真机首验先小步复验 tfs×WBC 语义。

**SDK 退出段错误（2026-10-28 replay 台实证）**：`GalbotRobot.init` 后不经优雅关停直接退出（SystemExit/正常 return 皆然）= 进程退出段 C++ 析构 **Segmentation fault**。正解 = 执行器 main 尾同款三连 `request_shutdown() → wait_for_shutdown() → destroy()`（.pyi :2716/:3125/:2201），放 `finally` 覆盖拒答/Ctrl-C/正常完成全路径；修后拒答路径 exit=1 干净。G1 的 "os._exit 硬退" 教条在 S1 的对应正解就是这三连（执行器一直带所以 dry-run 退出全干净，回放台漏了才暴露）。

**S1 臂 PVT 控制器约束面（2026-10-28 真机卡顿排查定位，权威=teleop toml）**：`/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/{left,right}_arm_pvt_ctrl.toml`——每关节 `limit_vel=3.0 rad/s`/`limit_acc=25`/pid kp100-kd40，**关键=target_pos 一阶 IIR 低通 cutoff 3Hz @1kHz**（τ≈53ms，阶跃走完 ~150-250ms，指令恒滞后 ~53ms）。**真机卡顿机制锚点**：fr=1 单点=阶跃灌 LP→坡+平台死区→步频力矩反复=卡卡声（5Hz 残差<1mrad=平台走死实证；30Hz 残差 1.5-30mrad=LP 尾巴非追踪误差）；多点 batch=WBC 插值连续目标流=静音（15Hz fr2 ✓）；30Hz fr12（窗 33.3ms≪LP 150ms，点距 2.78ms）仍卡卡。用户经验律自洽：低频×小导程 safe。窗口律初判（后被降级，见下段）：调用率/点距嫌疑排除，塌缩为命令窗口=R×行周期（30Hz 单行窗 33.3ms 仅『还好』、66.7ms 窗安静、点距细更佳）；`--rows-per-cmd` 已落地执行器/回放台（跨行 batch，窗口与行周期解耦零额外延迟，30Hz 实时配方=R=2×fr12；默认 1=逐行原路径逐位不变；smoke 三态绿+executor dry-run 绿）。遥测残差判读改口径：块尾残差主源=LP 滞后尾巴，勿当追踪误差。


**窗口律降级+真破案（10-28 深夜）**：上述窗口/点距 A/B 结论全被回放伪影污染——所有跑都带 `--execute-actions-per-chunk 20`（50 行块只放 20 行=每块界丢 30 行≈1s 动作→人工跳变 300-500 mrad 大/小逐块交替）；遥测铁证=钳位/残差与跳变幅度逐块相关（奇块钳位 68-82/残差 31-35=delta-max 顶棚追赶段=卡卡真源=速度/加速度饱和，用户『加速度』直觉对；偶缝 50.0mrad=整一钳位步=追赶未完铁证）；r/c=0 复测（30Hz R2fr12）钳位塌 0-48、迟到全 0、rep 界缝 2.1mrad=数据集闭环、用户判『挺不错』；残差逐块交替=LP 尾×数据速度相位非缺陷；**残留『切换一下卡顿』=块界重锚**（prev/EMA 锚到滞后 ~30mrad 的当前读数→目标流小反向×每 1.67s，平滑非齿轮——用户耳辨『非电机』正确；执行器同款行为=真换块必须重锚，安全正确）；回放台已加 `--seam-cont`（续锚上块末目标，数据连续口径消顿挫，A/B 用；executor 保持重锚，真闭环 A/B 后再议回移）。窗口/点距结论（33.3/66.7ms 窗、点距细佳）待干净口径重审；fr1 阶跃卡+LP 机制面不变。**SDK 下发语义（pyi 官方 docstring，10-28 查证）**：单点 set_joint_commands 标准关节只有 position 生效、velocity/effort 被忽略（爪例外：velocity+effort 生效=独立口发爪的依据）；batch 接口列 velocity/acc/Kp/Kd 未明示忽略=API 层配速唯一候选，但 pyi 无 Kp/Kd 字段且语义（前馈 vs 限幅）未验证，勿盲用。
