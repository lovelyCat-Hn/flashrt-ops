# 2026-10-17 entry-match 回放台接缝消融（2×2）

**chunk 源**：`logs/chunks_traj_20261017_154417/`（27 块，30Hz traj 线裸配方 `--dump-chunks`
落盘的原始模型输出，无 RTC/无 EMA；chunk_028 被 q 截断弃用）
**回放**：`scripts/test/chunk_replay_test.py`，G1 右臂真机 open-loop，每块消费 11/50 行，
`set_joint_commands` tfs=0 直发 30Hz，链式限幅 ±0.05 rad/行。
**entry-match**（用户提案）：换块时读臂实时位姿 → 顺运动方向（主导关节不倒行，
frac=0.5）+ 首拍导程对齐稳态导程 g₀（近 4 次实发中位，夹紧带 [8,35] mrad）→
取 |gap−g₀| 最小的可行行 j 为新块首派行；无可行行回退反向量违约最小行。

## 文件清单

| 文件 | 配置 |
|---|---|
| chunk_replay_test_20261017_164248.log | v1（全关节方向门，g₀ 无夹紧，hold 330ms 固化），裸 |
| chunk_replay_test_20261017_164457.log | v1 + EMA 0.3 |
| chunk_replay_test_20261017_165343.log | v2（主导关节门 + g₀ 夹紧带 [8,35] + `--hold-ms 0`），裸；⚠ 无窗完整钳位（9 块消费不满，4 块仅 2 拍） |
| chunk_replay_test_20261017_165734.log | v2 + 窗完整钳位（`_jmax=len(rows)−consume`）+ EMA 0.3（=V6c，最优） |
| task_json_dump_source.txt | dump 目录 task.json 侧车（包络/prompt 溯源） |

## 命令

```
~/holy/run.sh ~/holy/scripts/test/chunk_replay_test.py \
  --chunks-dir ~/holy/logs/chunks_traj_20261017_154417 \
  --rows-per-block 11 --entry-match [--ema-alpha 0.3] [--hold-ms 0] --rounds 27 --exec
```

（前三跑为迭代版：v1→v2 间修复 argparse 误删 c8452882；窗钳位 2246e5b1）

## 结果速览（mrad，详见 BENCHMARKS §十一）

| 跑 | 残差 mean/p95/max | 入口行 p50/max | 满拍块 | 总耗时 |
|---|---|---|---|---|
| v1 裸 | 51.4 / 138.1 / 268.4 | 17 / 46（匹配 13/26） | 26/27 | 17.4 s |
| v1+EMA | 33.2 / 100.6 / 156.1 | 3 / 39（匹配 24/26） | 27/27 | 17.7 s |
| v2 裸 | 40.4 / 87.8 / 99.7 | 23 / 48（匹配 26/26） | 18/27 | 7.2 s |
| v2+钳位+EMA | **26.2 / 75.4 / 92.1** | **0 / 8**（26/26） | **27/27** | 9.1 s |

V6c 入口分布：j=0×21 / j=1×5 / j=8×1，部分违约回退×2（各 1 关节·行）；
块 4-9（轨迹快速段）g₀ 原始中位 60-88 顶夹紧 35、入口 gap 49-75；
尾段（块 12+）g₀ 稳态 8-16、入口 gap 8-15。
