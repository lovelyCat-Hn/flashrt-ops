---
name: galbot-s1-pi05-inference
description: S1 pi05 真权重（lerobot 导出）接入 run_s1_loop 推理线——语义链/适配器/四坑/管线化（traj=行为标准）/爪关节流，dry-run 全绿待真机 exec
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-29T09:48:04.586Z
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
