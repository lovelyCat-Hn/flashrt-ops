# S1 通用推理执行路线（policy 无关）

`run_g1_loop_traj.py` 的 S1 镜像，按 policy 适配器解耦：**执行器只做 S1 SDK 交互、
循环节拍、安全栏与遥测；policy 专属的一切（引擎加载/观测预处理/动作语义/System 2）
全部住 `policies/<name>.py`。**

## 执行路线（流式，对齐 G1 traj 哲学）

lerobot 动作块 → 逐行动作 → `--frames-per-row` 帧插值（**默认 1 = 不插值，
纯 30Hz 直发**，用户定案；>1 启用插值，帧距 = ctrl_hz 倒数/帧数）
→ `set_joint_commands_batch` 一次流式提交。batch 接口官方 docstring 明示适用
"VLA inference batch output"；每点 `time_from_start_second` = 相对轨迹起点的累计
时刻（官方示例口径，首点=dt）。`set_joint_positions` 不用于执行（官方明示不适合
流式；G1 traj 的立意就是流式适配 lerobot 块语义，S1 同）。夹爪仍走独立 API
（G1 `--grip-state-cmd` 同位）。⚠ tfs×singorix WBC 耦合语义未真机验证——
G1 血泪史（tfs0 每点全停滞后 2.1×、tfs33.3 stall 判死）上机前小步复验。

## 结构

```
run_s1_loop.py        执行器本体（policy 无关）
policies/__init__.py  ADAPTERS 注册表 + create(name)
policies/base.py      PolicyAdapter 接口 + S1Targets + S1 关节/限位/夹爪换算
policies/fineart_vla.py  fineart 适配器（ALOHA 14 维；System 2 hook）
policies/pi05.py         pi05 适配器（G1 16/23 维语义；接口留位）
```

**新增 policy = `policies/` 加一个文件、注册进 `ADAPTERS`，执行器零改动。**
适配器四件套：`config_name / action_dim / state_dim / camera_map` +
`load(ckpt, task, calib_obs)` / `infer(obs, state_raw)`；可选 `subtask()`（System 2）
与 `to_s1(action_row, state_raw) → S1Targets`（动作语义映射）。

## 用法

```bash
PYTHONPATH=/data/galbot/lib LD_LIBRARY_PATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python run_s1_loop.py --list-policies
# dry-run 全链（默认，零运动）
... run_s1_loop.py --policy fineart_vla --task "pick up the object" --rounds 3
# 真机相机取图
... run_s1_loop.py --policy fineart_vla --live-frames --subtask-every 5 --rounds 10
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
  sent/frames/clamped/S2 text + 运行配置）。
