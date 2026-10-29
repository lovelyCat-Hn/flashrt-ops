# S1 probes（回放台 / 执行链隔离工具）

## replay_stream_bench.py — 流式回放台

把轨迹数据喂回 **run_s1_loop 同一条下发路径**，隔离模型变量（G1
`scripts/g1/probes/replay_stream_bench.py` 的 S1 镜像，2026-10-28）。
用途：①真机首验 tfs×singorix WBC 语义的最安全入口（轨迹冻结、无模型方差、
delta-max 链钳自带慢速就位效应）；②量化执行链（节拍迟到/块缝/回读残差）；
③遥测落 `~/holy/evidence/s1_replay/` 为闭环对比预留同格式。

**同构保证**：直接 import run_s1_loop 的代码对象（stream_row/plan_row/
ema_row/send_grippers/sleep_to 原封复用），回放只换"动作来源"。
刻意比主执行器多一层：**每 tick 重读臂关节刷新链钳基准**（G1 traj 同款
per-tick re-read；主执行器现为每块读一次，真机 A/B 后再回移）。

### 数据格式（win 上传轨迹）

- `.npz`：`chunks [C,L,D]` 绝对目标 + `joint_names [D]`（S1 运行时实名）+
  可选 `grip [C,L,2]` 夹爪宽度 m〔left,right〕+ 可选 `hz`
- `.json`：同构 `{"chunks": …, "joint_names": […], "grip": null, "hz": 30}`
- `--synthetic N`：本地合成平滑正弦块（离线冒烟/节拍测试，零依赖）
- 支持关节族 = left/right_arm_joint1-7 / torso_lift_joint1（**米**）/
  head_joint1-2；未知关节名直接拒绝。lerobot parquet 转换器等样例到了再加
  （flash_pyrt311 无 pyarrow/pandas）。

生成 npz 示例：

```python
import numpy as np
np.savez("traj.npz", chunks=chunks, joint_names=names, grip=grip)  # 后两个可选
```

### lerobot 数据集直转：lerobot_to_traj.py

S1 遥测数据集（如 `~/datasets/clean_ready_224_1028`，galbot_one_s1，action
16 维右臂在前、爪=0–100%、fps=30）→ traj npz。**在 lerobot_cpu 环境跑**
（pyarrow 所在；flash_pyrt311 无）：

```bash
~/miniforge3/envs/lerobot_cpu/bin/python scripts/s1/probes/lerobot_to_traj.py \
  --dataset ~/datasets/clean_ready_224_1028 --episodes 0,1,2
```

列映射按 info.json names 按名对号（不依赖列序）；episode 过滤按 data parquet
的 `episode_index` 列（meta 的 dataset_from/to_index 是全库全局索引，勿当文件
内偏移）；爪 %→m 走 base.py 同常量同公式；每集按 50 行切伪块、尾块 hold-last。
⚠ 该数据集 meta/stats.json 臂值范围（±0.5）与 data 实测（±2.0 rad）矛盾
（全库量纲一致，stats 疑清洗前旧口径）——norm 判读勿引这份。

真数据首验读数（ep0-2，2026-10-28）：遥测增量几乎全 <0.05 rad/tick（钳位
≈0，30Hz 回放忠实复现）；chunk0 钳位 257=从零位到工作位姿的就位段（链钳
0.05/tick≈1.5 rad/s 渐进）；集界缝 15-31 mrad、集内伪块缝 1-7 mrad。

### A/B 维度（v1 精简；G1 台 stream/direct/stepwait/tfs 谱系按需再移植）

| 键 | 默认 | 说明 |
|---|---|---|
| `--frames-per-row` | 1 | 1=纯 30Hz 直发（执行器默认）；>1=行内插值微流 |
| `--rows-per-cmd` | 1 | 跨行打包行数 R：命令窗口=R×行周期（30Hz 实时窗口律配方=2×fr12） |
| `--filter-alpha` | **0** | 回放忠实默认 0（显式传才滤波；与执行器 toml 档无关） |
| `--max-joint-delta-per-step` | 0.05 | 臂链钳 rad/tick；0=关仅限位 |
| `--swap-gap-ms` | 0 | 块间 hold 模拟推理落地窗（G1 定案忠实节奏=0） |
| `--rounds` | 1 | 序列重复遍数 |

### 真机卡顿定案（10-28 深夜收官；早前"窗口律"结论已降级，见第 1 条）

S1 臂 PVT 控制器（`/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/
{left,right}_arm_pvt_ctrl.toml`）每关节 limit_vel=3.0 / limit_acc=25 /
**target_pos 3Hz 低通 @1kHz**（τ≈53ms，指令恒滞后 ~50ms）。排查史两段：

1. 早期 A/B 全带 `--execute-actions-per-chunk 20` 伪影（50 行块只放 20 行=
   每块界丢 30 行≈1s 动作→人工跳变 300-500 mrad 大/小逐块交替）；遥测铁证=
   钳位/残差与跳变幅度逐块相关——**delta-max 顶棚追赶段=卡卡真源**（速度/
   加速度饱和）。
2. `r/c=0` 复测（30Hz R2×fr12）：钳位塌缩 0-50、迟到全程 0、rep 界缝
   2.1 mrad（数据集首尾闭环），用户听感『挺不错』。

残留『切换顿挫』=块界重锚（prev/EMA 锚回滞后 ~30 mrad 的当前读数→目标流
小反向；平滑非齿轮；**执行器同款安全行为**——真换块必须从"现在在哪"重
规划）。`--seam-cont`（续锚上块末目标）仅属回放台；executor 接缝归 G1
工具箱线（异步预取/EMA 接缝调度/RTC 引导，闭环线挂账）。

**回放忠实定稿配方**：`--execute-actions-per-chunk 0 --rows-per-cmd 2
--frames-per-row 12 --seam-cont`。fr=1 单点直发仍真卡（阶跃灌 LP 的步频
力矩反复，与伪影无关）；R2fr1 对照骨架同款、密度二阶未再分。SDK 语义
（pyi 官方）：单点 set_joint_commands 标准关节只有 position 生效
（velocity/effort 忽略，爪例外）；batch 接口 velocity/acc 语义未验证勿盲用。

### 电机异响电流签名排查（10-29 工具落地；--log-hz + --marks + kaka_signature.py）

目的：定位**残留**异响（r/c=0 干净配方下偶发 + fr=1 单点直发的系统性卡卡）
的电机层根因。仪表盘已验证（10-28 零运动探针）：`get_joint_states` 的
`current` 字段=真力矩代理（静态重力基线可达 ±3A，纹波 ±0.05A）；
`effort` 恒 0 未用；`velocity/acceleration` 有但加速度噪声大（±0.1-0.17）；
timestamp_ns 可用。**电流签名判据表**：

| 假设 | 电流签名 | 辅证 |
|---|---|---|
| A 齿隙扫过 | 反向瞬间电流过零/大幅 \|Δcur\| 尖峰 | 误差换向点对齐 |
| B 加速度饱和（limit_acc=25） | 电流顶棚平台 + acc p95≈limit | 追赶段（钳位计数高）重合 |
| C 速度饱和急停（limit_vel=3） | \|vel\| p95→3 后电流骤降 | — |
| D hunting（kp100/kd40） | 静止段电流固定频率振荡（FFT 显著峰） | 已初排：静态保持无振荡 ✝ |
| E stick-slip | 低速段速度微脉冲 | — |

**跑法**（每跑一段，听到异响按回车打标；Ctrl-C/自然结束都落盘）：

```bash
# ① 激发跑（故意带 r/c=20 伪影=已知追赶卡卡源）+ 高频采样 + 耳打标：
bash ~/joint_observer/run_observed.sh replay_stream_bench.py \
  --data ~/datasets/clean_ready_224_1028_traj_ep0_1_2.npz --rounds 2 \
  --ctrl-hz 30.0 --execute-actions-per-chunk 20 --rows-per-cmd 2 \
  --frames-per-row 12 --filter-alpha 0.1 --max-joint-delta-per-step 0.05 \
  --swap-gap-ms 0 --log-hz 200 --marks --out ~/holy/evidence/s1_replay --exec
# ② 安静对照跑（定稿配方 + 同采样），残留异响也打标：
#    同上换 --execute-actions-per-chunk 0 --seam-cont
# ③ 分析（kaka 窗=标记±1.5s vs 静窗对照；FFT 只报显著峰>8×谱底）：
~/miniforge3/envs/flash_pyrt311/bin/python scripts/s1/probes/kaka_signature.py \
  ~/holy/evidence/s1_replay/kaka_*.npz
```

合成自测已验（注入 j3 12Hz ±2.5A：响窗 p95/尖峰率/速度全中、其余关节零
假阳性、FFT 独中 12Hz）。真机注意：log_hz=200 时 tick 内突发采样≈每 tick
3 读（30Hz R 大窗口时更多）；真实 SDK get_joint_states 单次 ~ms 级，
若迟到升高可降 `--log-hz 100`。

**⚠ npz err 列口径（10-29）**：`get_joint_states.position` 与指令系**符号相反**
（臂关节全体：同列相关 −0.97~−0.98、负镜像 +0.98、每关节偏移 c 跨跑≈常量；
该接口坑二，坑一=返回规范序）。kaka_signature 已修（镜像+逐关节 med 去偏，
修后 err p50 25-35 mrad 合理）；电流/vel/acc 绝对值与谱判读不受符号影响；
钳位/残差回读走 `get_joint_positions`（指令系，正确）不受影响。

**三格 A/B 定案（10-29，同一数据集 30Hz r/c=0）**：①R1fr1 tfs=0 踢率 j1
18.4/s、acc 触顶 23-30.5=卡卡；②R2fr12 时间轴批 acc 17-23 不触顶=安静；
③R2×fr1 全零 tfs 批（G1 式 ASAP 假设，`--tfs-zero`）**卡卡回归且更凶**
（踢率 30.5/s≈①1.65×、4 关节 16.7Hz 显著音调峰）→**时间轴必要性实锤，
因果链闭环：跳变参考→LP 速率需求→limit_acc=25 削顶→逐级力矩跳变∝Δ**。
文档语义（singorix_target_guide.md）：LinearInterpolate 按 tfs 时间轴自动
细插——33ms 时间轴是解药非病因；tfs=0 改变的是参考形状（斜坡→阶跃）。
原始件 `evidence/s1_replay/kaka_1793240742/1793240779/1793246486.npz`。
头探针 `batch_tfs_probe.py`（A 单点 tfs=2s 判 OVERRIDE/PROVERRIDE、B 批次
时间轴、C tfs=0、D 全零批；head_joint1 幅 0.03rad，stub 自测绿）挂账可选。

### 安全

dry-run 默认**零 SDK**（任何机器可跑；dry-run 状态走虚拟跟随，钳位计数=
真实链钳行为而非冻结态伪影）。`--exec` 需交互确认 + `start_controller`
（急停手不离，模式切换失力矩 10-27 教训）。包络 = S1_LIMITS 钳位 +
delta-max 臂链钳（每 tick 重读基准）。Ctrl-C 即退并落盘部分遥测。
SDK 会话退出走优雅三连 request_shutdown/wait_for_shutdown/destroy（缺它=
退出段 C++ 析构 Segmentation fault，10-28 实证）。

### 用法

```bash
# 离线冒烟（回归闸，stub 零运动）：
PYTHONPATH=/data/galbot/lib:~/holy/scripts/s1/inference \
  ~/miniforge3/envs/flash_pyrt311/bin/python scripts/s1/probes/smoke_offline.py
# 干跑（真机旁先看计划）：
~/miniforge3/envs/flash_pyrt311/bin/python scripts/s1/probes/replay_stream_bench.py --data traj.npz
# 真机回放：
PYTHONPATH=/data/galbot/lib ~/miniforge3/envs/flash_pyrt311/bin/python \
  scripts/s1/probes/replay_stream_bench.py --data traj.npz --exec
```

遥测（每块一行）：`rows | 帧流 | 钳位 | 缝 mrad（块界计划跳变）|
迟到 p50/max ms（节拍落后量；>1 拍=G1 铁律里的重锚场景）|
残差 max（块尾回读 vs 末行目标；臂 mrad / torso mm / head mrad）`。
