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

### A/B 维度（v1 精简；G1 台 stream/direct/stepwait/tfs 谱系按需再移植）

| 键 | 默认 | 说明 |
|---|---|---|
| `--frames-per-row` | 1 | 1=纯 30Hz 直发（执行器默认）；>1=行内插值微流 |
| `--filter-alpha` | **0** | 回放忠实默认 0（显式传才滤波；与执行器 toml 档无关） |
| `--max-joint-delta-per-step` | 0.05 | 臂链钳 rad/tick；0=关仅限位 |
| `--swap-gap-ms` | 0 | 块间 hold 模拟推理落地窗（G1 定案忠实节奏=0） |
| `--rounds` | 1 | 序列重复遍数 |

### 安全

dry-run 默认**零 SDK**（任何机器可跑；dry-run 状态走虚拟跟随，钳位计数=
真实链钳行为而非冻结态伪影）。`--exec` 需交互确认 + `start_controller`
（急停手不离，模式切换失力矩 10-27 教训）。包络 = S1_LIMITS 钳位 +
delta-max 臂链钳（每 tick 重读基准）。Ctrl-C 即退（遥测不落盘）。

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
