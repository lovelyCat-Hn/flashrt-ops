---
name: galbot-s1-pi05-inference
description: S1 pi05 真权重推理线——语义链/四坑/管线化/traj 直移植版执行器（现役主线，10-29 审查四处落地，R2×fr8 工作点待真机 exec）
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-29T11:11:58.521Z
---

# S1 pi05 推理线（2026-10-29 落地，接 [[galbot-fineart-vla-flashrt-port]] 的执行器底座）

**权重**：`/home/galbot/model_weights/pi05/S1_run_20261028_121132_b3f637a6_010000/pretrained_model`（同事 lerobot 管线微调，训练集=`~/datasets/clean_ready_224_1028` 同款，episodes 0-46）。lerobot 导出布局（config.json+model.safetensors+policy_pre/postprocessor）——flash_rt torch 前端直读，norm_stats 自动走 lerobot pre/post safetensors 对读兜底（`flash_rt/core/utils/norm_stats.py`，action 取 post/state 取 pre）。

**语义链（三源码交叉定案）**：
- action 16=[右臂7,右爪,左臂7,左爪]（数据集实名右臂在前）；state 27=前16与action同序+torso+head2+swerve×8
- use_relative_actions=True 排除爪 → 模型输出（unnorm 后）=**臂维 delta（相对 t0 state）+爪维绝对(0-100%)**
- 还原=lerobot `to_absolute_actions`：**absolute[k]=s_t0+delta[k] 逐行同锚广播，非 cumsum**（锚=喂 policy 的 state 前16位）；归一=QUANTILES 2(x−q01)/(q99−q01)−1 与 flash_rt q01_q99 同式
- 训练 task 字符串=**"test"**（tasks.parquet 唯一任务）→ 推理 prompt 用同串
- 训练帧口径=**4:3 letterbox 224²**（内容 224×168+上下 28px 黑条，数据集帧实测）；真机 S1 相机原生 4:3，letterbox224 后内容高 168 **精确匹配**（live dry-run 判据 ✓，适配器 load 时自动体检 `_check_frame_caliber`）
- swerve 8 维喂 0：训练里 1/3/5/7≈0、4≈307（常数→归一饱和与 0 同 bin，np.digitize 吸收）、2/6/8 狂转(spread 160-253 模型只能忽略)；torso(恒0.735)/head 喂真值

**三坑（都实咬过）**：
1. **VLAModel 包装层无 norm_stats**：load_model 返回 api.VLAModel，`norm_stats`/`calibrate_with_real_data` 住 `model.pipeline`（前端实例）——读包装层属性=静默空 dict
2. **toml ckpt 覆盖适配器 CKPT_DIR**：s1.toml 预扫填 argparse 默认，忘带 --ckpt 时加载 toml 里旧 policy 的权重（曾把 fineart 14 维 stats 喂给 pi05 护栏拦下）；现 toml 工作点已切 pi05+S1 路径+prompt="test"
3. **plan_row 块首锚定=龟速 bug**：delta-max 曾相对块首 state_raw 钳→整块 50 行钳死块首±0.05=臂速 0.05rad/块；G1 正主（run_g1_loop_traj:1830）=**链式限幅参考不重锚**（`prev+=clip(tgt−prev,±δ)`）。已修：plan_row 收 name→值 dict，执行器传演进链 chain（块界从新读数重启=重锚安全行为），回放台传每 tick 新读数字典

**状态**：dry-run 全绿（引擎加载 10-14s；infer **287-386ms**/块 50 步，exact 模式每轮 set_prompt 无重建税——state token 数恒定 prompt_len 恒定 pipeline 复用；若慢有 `FLASHRT_PI05_STATE_PROMPT_MODE=fixed` 环境钩子）。needs_mapping=True 闸②开。离线探针 `scripts/s1/probes/pi05_mapping_check.py`（stats 链+映射数学+帧口径判据）。

**待真机 exec**（用户在场手扶急停）：`PYTHONPATH=/data/galbot/lib ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/s1/inference/run_s1_loop.py --exec --live-frames --rounds 3 --rows-per-cmd 2 --frames-per-row 12`（toml 供 policy/ckpt/prompt/水位 0/爪 joint；**R2×fr12=四角闭环静配方**，toml 默认 R1fr1=①卡卡勿直接 exec；**--live-frames 必带**——10-29 首跑漏带=盲跑合成噪声图无任务意义，现确认框会警告）。链路=model delta→+t0 锚→绝对目标→链钳 0.05/步→LP；爪=关节流每拍无条件发。首跑核查：模型输出 delta 量级 vs 训练 stats（max≈0.53rad）、钳位计数、**round 行「消费≈10/50 弃尾 40 迟到 max≤~10ms」=管线健康**、grip 关节流 ControlStatus（真机爪组首次验证）。

**复位功能（10-29，ops 1316d1f）**：`--exec` 默认就位（开始前）+归位（结束/Ctrl-C 后，二次 Ctrl-C 跳过）到 **S1_HOME_POSE**（base.py 烘档=clean_ready_224_1028 全 47 集首帧逐关节中位：臂散布 ±0.1 rad 内/爪恒 100% 开/torso 0.735/head 0——in-distribution 起点）；home_to=回放台同款机制（每拍 fresh-read 链钳+2 点时间轴批，与 R/fr 解耦不卡卡；爪一次一包）；`--no-home-start/--no-return-home/--home-tol 0.03/--home-timeout 30` 可调。smoke 增 home_to stub 段（43 批收敛 5 mrad）。首跑日志判读：盲跑（无 --live-frames）3 轮全绿 infer 374-378ms/钳位 0——链路健康但无任务意义。

**管线化（10-29 用户定案，销 G1 异步预取 −12% 挂账）+ traj=行为标准**：用户明示「G1 traj 就是行为标准，写 traj 就是为了 S1 直接用，勿被 native 带偏」。①**水位=预取触发阈值**（traj 同名语义，非消费上限）：0=换块即预取（本块第 0 步下发后立即后台线程起跑下一块推理）、N=消费到第 K 步触发；**换块=推理落地即整块换入弃尾**（不等消费完）→ obs 年龄恒≈推理时延 0.3s（同步口径 0→1.67s 渐增），dry-run 实证消费 10/50、换块周期 0.37s、**迟到 max 6ms**（常驻遥测）；**星饿**（块耗尽推理未归）=不下发臂保持，超 `--starve-limit` 2.0s（traj 同名同默认）判挂死停循环；to_s1 锚=推理时读数（模型语义），链钳/EMA 锚=块界新读数；SDK 调用全主线程、工作线程只碰 torch；首块 obs 新抓（obs0=载入前校准帧隔着就位段已陈旧）；infer_pool.shutdown(wait=True) 在 SDK 三连前。②**爪=关节流**（`--grip-path joint` 默认=traj 标准；cmd=set_gripper_command 逃生门带 2mm 死区）：`set_joint_commands(joint_groups=["left_gripper","right_gripper"])` **position=宽度米、velocity=速度帽(toml 0.05)被尊重、effort=10**（pyi 明示爪维 velocity/effort 生效）；**每发送拍无条件发**——onset/吞令/chg 各门属 cmd 端点专属（G1 P6 三格），joint 路径密流免费；EMA 照带（ema_row 全滤臂/torso/head/爪，块界从 state_raw 爪 pct 维播种宽度米）；home 双爪一包一次。

**坑④（自咬）——绝对节拍基准必须块首取一次**：`tr=time.perf_counter()` 进拍内→`sleep_to(tr+period·(k+1))` 截止成**二次曲线**（拍间隔 33→67→100→133ms 递增，4 拍吞掉 290ms 推理窗=假象「GIL 争用」）；正解=块首 t0 一次、`sleep_to(t0+period·(k+1))`（G1 next_t 累计同款）。同轮另修：换循环时 r 忘自增=round0 死循环（dry-run 即现形）。

**traj 直移植版 run_s1_loop_traj.py（10-29 深夜落地，用户定案「直接把 G1 traj 复制过来换关节」，取代逐处打补丁路线）**：run_g1_loop_traj.py 整文件复制，**只换机体件**——①SDK 面 galbot_sdk.g1→s1（G1JointGroup 枚举→组名字符串 "left/right_gripper"，set_gripper_command 5 参同形）；②state 27 维=[右臂7,右爪%,左臂7,左爪%,torso,head2,swerve×8 恒0]，read_joints 双形态（ARM_NAMES=14 臂一次读；STATE_NAMES=臂+torso/head+get_gripper_state 宽度米拼装）；③取图 get_rgb_data(SensorType, RgbOutputFormat.JPEG, True)；④爪标定 ckpt 无 manifest→出厂缺省 0.007/0.11m+告警；⑤**发令批形态=唯一行为侧改动**：G1 逐帧单点 tfs=0 流→每拍一包 set_joint_commands_batch（tick 内 fr 帧并批、tfs 全批累计 fdt·(i+1)=墙钟节拍；③定案 tfs=0 ASAP=卡卡、时间轴是解药；四角④同款形状；--tfs-ms 保留为批内追加余量）；⑥**warm 内嵌**（G1 手跑 g1_pose_warmup→s1_warm()：set_joint_positions is_blocking 臂+头@0.15rad/s→1s→torso@0.2→双爪关节流一包全开，目标 S1_HOME_POSE；--exec 确认后就位+自然跑完归位；q/Ctrl-C 照 traj 立即 os._exit 不复位；10-29 适配器版 1.5rad/s 链钳被用户判「一瞬间复位」10× 超速=治愈动因）；⑦NavSuspend 整类删（旗标留档键 no-op）。水位/换块弃尾/星饿/链钳/EMA/融合/QuitWatcher/日志/汇总逐行未动。配套：s1_config.py（扁平键版 g1_config，读 config/s1.toml 三态回填）+s1_traj_interp.py（插值器逐行副本）+s1.toml 重构（frames_per_row 1→12、新增 skip_frames/aggregate_fn 等 traj 配方键、run_s1_loop 预扫描 `k in _known` 过滤不冲突）。配置链预 SDK 段全绿实测（toml 全键覆盖/banner 水位 0+fr12/爪缺省告警）。**两坑**：G1 traj 原版 help 串裸 `%`（`−56%）`）在 py≥3.11 `--help` 即炸=潜伏 bug 两机各修一处 `%%`；适配器版 home_to 批 tfs=period×(i+1) 但链钳 1.5rad/s=复位过速病根（机制对、速率错）。

**10-29 审查四处（另一 agent 复查，全落地+干跑实证）**：①**fresh_obs（S1 三相机 ~86ms p50）搬进 _PredictJob 工作线程**——主线程同步取图≈2.6 拍，水位 0 下每块预取拍必丢拍重锚=块频 ~2.5Hz 卡顿节奏（与真机卡卡同区间）；rtc_prefix 随 fresh_obs 在线程内取（读 chunk/BASE_ARM 只发生在换块间隙，主线程只推进 last_tgt/k 无写重叠）；干跑实证 tick 服务 max 130.8→22.6ms、丢拍 1→0。**坑：_run 内对闭包名 obs 赋值→整个函数域变局部，前置 `if obs is None` 必 UnboundLocalError；正解=def 时默认参绑定 `def _run(obs=obs,...)`**。SDK 双线程并发（worker 读 vs 主线程下发）未真机验证，`--no-obs-worker` 一键回退对照。②**--rows-per-cmd R 正式入 traj 版**（此前只适配器版有）：_rp_flush 攒 R 拍帧一包、tfs=fdt·(i+1) 铺满 R×tick 窗、每 R 拍一发；换块=半窗攒帧随旧块弃尾、星饿进入/自然收尾=先冲账（放完攒窗缓停）、护栏/abort 停机不冲账；R=1 与旧每拍一包逐字同款。③**fr 12→8**（4.17ms≈wbc_dt 4ms 采样周期；fr12 的 2.78ms 点距 ~1/3 被采样器跨过）。④**--cmd-every>1 S1 SystemExit 硬拦**（含 first-only/ramp：S1 WBC 直视计划断点，G1 伺服吞阶跃语义不成立；跨行平滑走 ②）。toml/s1_config 工作点=R2×fr8。**待真机判别（审查附问）**：每拍新建 Trajectory 无 target_ts → SDK 匹配 OVERRIDE（清队列重排）还是 PROVERRIDE（插种子续接），单点 tfs=2s 实验可辨（batch_tfs_probe.py 已有 stub 绿未真机）。

**10-30 换块融合真机 A/B 定案=双层胜单层（判读挂档）**：午间按用户要求把换块统一给 EMA（latest_only，缝只留 EMA 单层），真机对照用户判「**效果没有之前的好**」→ 当晚改回 **aggregate_fn=weighted_average（混接 crossfade 0.3旧/0.7新 + EMA 双层）现役**——缝上的旧流惯性项要留，EMA 单层判退（latest_only 仍可 `--aggregate-fn latest_only` 复现对照）。原单层改动细节：缝平滑只留 EMA 跨块连续（α=filter_alpha，换块从 last_tgt 重播起链）——缝首行等效新计划权重 0.7×0.1=7%→0.1=10%。代码侧：_ovp 门改开区间 0<α<1（恒等混接免重建帧免打印，与 --merge 互斥检查同口径）。干跑绿：横幅只显 EMA α=0.1、换块无 [重叠混接] 打印、缝首拍 0.6-1.3mrad/回走≤1.1mrad/丢拍 0（b9d2802）。**EMA 两问判读（10-30 答）**：traj 版 EMA=跨块连续非块内重启（块内重启版对接缝无效，10-12 对照判明）；水位 0 弃尾下换块 _EMA_STATE 重播为 last_tgt（否则状态记在未发送尾巴上=块首瞬移，161933 实证）。

**待真机 exec**：先无 --exec 干跑（SDK init+取图+推理+首块计划打印，零运动）再 --exec；就位 warm 与进环各一次回车确认。
