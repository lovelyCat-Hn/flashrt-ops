# fineart_vla e2e + 延迟三口径（协议③）

日期：2026-10-28　机器：S1（Jetson Orin, L4T R35.6.0）　env：flash_pyrt311
链路：`generate_subtask`（System 2，bf16 GPU）+ `infer`（System 1 flow，chunk 50 / step 10 / 14 维，graph on）
调用面 = parity 已证 surface（golden_engine / golden_subtask_engine 同款）；状态 = golden seed7 raw → q01/q99 归一化
SDK 口径调用形 = run_g1_loop_traj.py（`galbot_sdk.s1` GalbotRobot，init 相机订阅 + get_rgb_data 探针，**零 start_controller = 零运动**）

## 三口径结果（ms，20/20/60 轮）

| 口径 | S2 p50 | S1 p50 | Σ p50 | Σ p90 | 备注 |
|---|---:|---:|---:|---:|---|
| 空载（冻结合成帧） | 893.6 | 231.9 | 1125.7 | 1134.7 | 文本 20/20 同句（贪心确定） |
| SDK 在场（s1 相机订阅在场） | 905.0 | 232.6 | 1137.7 | 1146.5 | Δ(sdk−空载)：Σ +12.0（+1.1%），几乎全在 S2（+11.4） |
| 循环（每轮换帧+换噪声，60 轮） | 885.0 | 232.1 | 1116.7 | 1143.8 | 稳态服务节拍 **0.89 Hz**；S2 min 630.7 / max 1147.1 |

全部轮次 actions (50,14) 有限值 ✅。循环口径每轮换帧 → System 2 文本随观测变化
（60 轮内出现异句，如 `Prepare the table to the other table mat`）——S2 对观测
响应的机制证据；冻帧口径同输入同文本 = 确定性。

## 口径注

- Σ = S2+S1 串行全链；官方用法 System 2 可降频摊销（非每 chunk 必生成），0.89 Hz
  为每轮都生成文本的上界节拍；仅 S1 的节拍 = 1/0.232s ≈ 4.3 Hz（chunk 50 @30Hz
  ≈ 1.67s 执行窗，服务时间余量充足）。
- 与 pi05 线口径差：本台无真机取图（合成帧），SDK 成本只含「订阅栈在场」，
  不含 pi05 表 5-3 的采集守护/并行取图项。

## 文件与复现

| 文件 | 内容 |
|---|---|
| noload.json / sdk.json / loop.json | 逐轮 S2/S1/Σ 计时 + 判定字段 |

```bash
~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/policies/fineart_vla/bench_e2e_latency.py --mode noload
LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
  ~/holy/policies/fineart_vla/bench_e2e_latency.py --mode sdk
~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/policies/fineart_vla/bench_e2e_latency.py --mode loop
```
