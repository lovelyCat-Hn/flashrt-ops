# S1 通用推理执行路线（policy 无关）

两个执行器（2026-10-29 起）：

- **`run_s1_loop_traj.py` = G1 traj 直移植版【现役主线】**：用户定案「G1 traj
  就是行为标准，写 traj 就是为了 S1 直接用」——整文件复制
  `run_g1_loop_traj.py`，只换机体件（SDK 面/27 维 state/取图/爪标定缺省/
  内嵌 warm/导航栈删除）。行为逻辑（水位预取/换块弃尾/星饿/链钳/EMA/融合/
  QuitWatcher）逐行未动。唯一发令侧改形：G1 逐帧单点 tfs=0 流 → S1 每拍
  一包 `set_joint_commands_batch` 带时间轴批（S1 ③定案：tfs=0 ASAP=卡卡，
  时间轴是解药；与回放台四角闭环④同款形状）。配置=同目录 `s1_config.py`
  读 `config/s1.toml`（CLI > toml > BUILTIN 三态）。
- **`run_s1_loop.py` = 适配器版**：policy 解耦多policy 入口（fineart 线用），
  回放台/冒烟仍 import 它（`rsl`）。与新主线并行为 A/B 备胎，日常闭环走
  traj 移植版。

## 执行路线（流式，G1 traj 同款逐帧墙钟流）

lerobot 动作块 → 逐行动作 → 逐帧 `set_joint_commands` 直发（**tfs=0 立即
执行，行节拍=脚本绝对截止墙钟 tick，G1 同款：欠账不追发**；fr=1 默认=
不插值纯 30Hz，用户定案；fr>1=A/B 插值微流，行内小 batch 累计 tfs，
帧距 = ctrl_hz 倒数/帧数）。**节拍权在脚本，不整块打包给控制器**——
否则新计划只能等批播完、或赌 WBC 对播放中收到新 batch 的黑盒取代行为。
`set_joint_positions` 不用于执行（官方明示不适合流式；G1 traj 的立意就是
流式适配 lerobot 块语义，S1 同）。水位 `execute_actions_per_chunk`：
0=整块行流完（50 行≈1.67s）立即重推理；N=消费 N 行即重推理。
⚠ G1 的异步预取（换块即预取/星饿 hold）未带——同步循环下重推理 232ms
≪ 块长 1.67s，吞吐损失 ~12% 挂账。夹爪仍走独立 API（G1 `--grip-state-cmd`
同位）。⚠ tfs×singorix WBC 耦合语义未真机验证——G1 血泪史（tfs0 每点
全停滞后 2.1×、tfs33.3 stall 判死）上机前小步复验。

## 结构

```
run_s1_loop_traj.py   G1 traj 直移植版执行器（现役主线，见上）
run_s1_loop.py        适配器版执行器（多 policy；回放台/冒烟依赖）
s1_config.py          traj 版共享配置（config/s1.toml 三态回填）
s1_traj_interp.py     轨迹流插值器（g1_traj_interp.py 逐行副本）
policies/__init__.py  ADAPTERS 注册表 + create(name)
policies/base.py      PolicyAdapter 接口 + S1Targets + S1 关节/限位/夹爪换算
policies/fineart_vla.py  fineart 适配器（ALOHA 14 维；System 2 hook）
policies/pi05.py         pi05 适配器（G1 16/23 维语义；接口留位）
```

**新增 policy = `policies/` 加一个文件、注册进 `ADAPTERS`，执行器零改动。**
适配器四件套：`config_name / action_dim / state_dim / camera_map` +
`load(ckpt, task, calib_obs)` / `infer(obs, state_raw)`；可选 `subtask()`（System 2）
与 `to_s1(action_row, state_raw) → S1Targets`（动作语义映射）。

## 配置档与键名（对齐 G1 [lerobot] 体系）

默认读 `~/holy/config/s1.toml`（顶层键 = CLI dest 名，CLI 显式传参覆盖 toml；
`--config` 换档）。**键名继承 G1 `config/g1.toml [lerobot]` 的 lerobot/同事
对齐名**（命名档案 = `docs/lerobot-alignment.md`）：

| 键 | 语义 | G1 [lerobot] 对照 |
|---|---|---|
| `prompt` | System 2 任务提示 | 同名 |
| `execute_actions_per_chunk` | 水位：0=整块行流完立即重推理；N=消费 N 行即重推理（都逐帧墙钟直发） | 同名同款（现役 0） |
| `frames_per_row` | 每行帧数：1=不插值直发（默认，用户定案） | G1 traj 同名旗（cmd_every=1 同语义恒成立） |
| `rows_per_cmd` | 跨行打包行数 R：命令窗口=R×行周期。10-28 真机窗口律：单点/33.3ms 窗卡顿、66.7ms 窗安静；30Hz 实时配方=2×fr12 | S1 扩展键（G1 无） |
| `max_joint_delta_per_step` | 包络 0.05 rad/步（臂） | 同名同值 |
| `filter_alpha` | 行目标 EMA，0=关（G1 档 0.3 现役；S1 首跑先 0） | 同名（同事 action_filter.lowpass.alpha） |
| `subtask_every` | System 2 降频摊销（S1 扩展键） | G1 无此概念 |
| `skip_frames / aggregate_fn / grip_state_cmd / nav_suspend / execution_horizon / head_crop` | 刻意不带（判读见 s1.toml 注释） | N/A 或待真机证据 |

旧名 CLI 别名保留：`--n-action-steps` / `--delta-max` / `--ema-alpha` / `--task`。

## 用法

```bash
PYTHONPATH=/data/galbot/lib LD_LIBRARY_PATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python run_s1_loop.py --list-policies
# dry-run 全链（默认；配置走 ~/holy/config/s1.toml）
... run_s1_loop.py --policy fineart_vla --prompt "pick up the object" --rounds 3
# 真机相机取图
... run_s1_loop.py --policy fineart_vla --live-frames --subtask-every 5
```

## 安全设计（三重闸 + 零运动默认）

1. **默认 dry-run**：全链照跑（取图/推理/映射/打印），不发任何指令。
2. **`--exec` 需 `adapter.needs_mapping=True`**：fineart=ALOHA 14 维、pi05=G1 16 维，
   两者对 S1 19 维的语义映射均未定义（随微调线落地）→ `--exec` 一律结构性拒绝。
3. **`--exec` 交互确认** + `start_controller("all")` 显式开启 + 包络护栏
   （限位表逐关节钳位 + `--delta-max` 相对当前读数钳行目标，臂 only；
   插值帧由已钳目标构造，护栏不会被帧绕过）。流式构建器离线冒烟已验
   （Trajectory 构造/帧距/插值端点/delta-max 生效）。
4. **`--ema-alpha`（默认 0=关）**：行目标 EMA 低通最小版（G1 同名旋钮同语义，
   α=新样本权重；每块从当前读数重锚；G1 的接缝窗 α 调度/链钳/爪列 EMA 未带）。
   滤波在安全钳位**之前**，不会绕过限位；爪不滤（无翻转证据）。真机抖动露出
   再 A/B（G1 在役配方 EMA off，根治在训练侧）。

⚠ 2026-10-27 手臂掉落教训：模式切换/接管瞬间手臂可能失力矩——`--exec` 前手不离急停。

## 事实源与坑

- 关节/限位/夹爪口径：`~/holy/docs/s1-joint-api-quickref.md`；限位权威 =
  `/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml`。
- **读关节必须显式 `joint_names=`**（group 模式与 names 配对全错，G1 实证）。
- torso 运行时实名 `torso_lift_joint1`（config 文档写 `torso_joint1`，不符）；
  **升降柱，位置语义=高度米（0–1.0 m）非 rad**（官方 example kTorsoJt=0.60 "meter"）。
- 流式接口事实（.pyi 实证）：`set_joint_commands` 高频单点（标准关节只吃
  position；爪关节 position=宽度米、velocity/effort 生效）；
  `set_joint_commands_batch(Trajectory)` 非阻塞多点，`Trajectory.joint_names`
  优先于 groups，每点 `time_from_start_second` 累计时刻；官方示例 dt=0.008。
- 首次 `generate_subtask` 含一次性预热（本机实测 ~26s），其后 ~0.6–0.9s/次。
- 遥测落 `~/holy/evidence/s1_loop/run_<policy>_<ts>.json`（每轮 infer_ms/steps/
  sent/frames/clamped/S2 text + 运行配置，配置键名=上表正规名）。
