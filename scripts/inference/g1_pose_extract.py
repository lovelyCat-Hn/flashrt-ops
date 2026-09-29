#!/usr/bin/env python
"""lerobot 数据集 → task 起始位姿 json（喂 g1_pose_warmup 的目标姿态）。

取数据集每个 episode 首帧（frame_index==0）的 observation.state 逐维中位数
——与 2026-09-23 pick_place_balence 的"分析脚本"同口径（98 条 task0 起始
中位），当时该脚本未入库，本文件是它的正式版。起始位姿是模型训练轨迹的
共同出发点，warmup 对齐它 = 对齐训练分布入口。

输出 json 与 warmup 既有约定兼容（start_pose 23 维 / n_episodes /
spread_max_dev），另附 task、source 便于人读。

用法（系统 python3，读 parquet 要 pandas+pyarrow；flash_pyrt311 无 pandas）:
  python3 ~/holy/scripts/inference/g1_pose_extract.py \
      --dataset ~/holy/datasets/only_place [--task 0] \
      [--out ~/holy/models/pi05_g1_place_deploy/episode_start_task0.json]

然后三选一让 warmup 用上（优先级从高到低）:
  ① g1_pose_warmup.py --pose-file <json>
  ② config/g1.toml [warmup].pose_file = "<json>"
  ③ 落到部署目录同名 episode_start_task0.json（零配置自动发现）
"""
import argparse
import datetime
import json
import pathlib

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", required=True, help="lerobot 数据集根目录（data/ + meta/）")
ap.add_argument("--task", type=int, default=0,
                help="task_index（-1 = 不分 task 全部起始帧；默认 0）")
ap.add_argument("--out", help="输出 json（默认 <dataset>/episode_start_task<T>.json）")
args = ap.parse_args()

ds = pathlib.Path(args.dataset)
pq = sorted(ds.glob("data/chunk-*/episode_*.parquet")) or sorted(ds.glob("data/chunk-*/*.parquet"))
if not pq:
    raise SystemExit(f"找不到 parquet：{ds}/data/chunk-*/")

frames = [pd.read_parquet(p, columns=["observation.state", "episode_index",
                                      "frame_index", "task_index"]) for p in pq]
df = pd.concat(frames, ignore_index=True)
first = df[df["frame_index"] == 0]
if args.task >= 0:
    first = first[first["task_index"] == args.task]
if first.empty:
    raise SystemExit(f"task_index={args.task} 无起始帧（总帧 {len(df)}）")

states = np.stack(first["observation.state"].to_numpy())
med = np.median(states, axis=0)
dev = np.abs(states - med)
# spread 只统计弧度维（dim7/15 夹爪是 0~100% 刻度，混入会虚大——2026-09-29
# only_place 实测 101 轨臂维最大波动 0.23 rad，夹爪 32~33.5% 贡献了 1.0）
RAD_DIMS = [i for i in range(states.shape[1]) if i not in (7, 15)]
out = pathlib.Path(args.out) if args.out else ds / f"episode_start_task{args.task}.json"

task_sentence = ""
tasks_p = ds / "meta" / "tasks.parquet"
if tasks_p.exists():
    tdf = pd.read_parquet(tasks_p)
    if "task" not in tdf.columns:          # 部分数据集把任务句放在索引上
        tdf = tdf.reset_index().rename(columns={tdf.index.name or "index": "task"})
    hit = tdf[tdf["task_index"] == args.task] if args.task >= 0 else tdf
    if not hit.empty:
        task_sentence = str(hit.iloc[0]["task"])

blk = {
    "start_pose": [round(float(v), 6) for v in med],
    "n_episodes": int(len(first)),
    "spread_max_dev": round(float(dev[:, RAD_DIMS].max()), 4),
    "task_index": int(args.task),
    "task": task_sentence,
    "source": str(ds),
    "generated": datetime.date.today().isoformat(),
    "dims": "23 维右臂在前: 右臂7/右爪0~100%/左臂7/左爪/腿5/头2",
}
out.write_text(json.dumps(blk, ensure_ascii=False, indent=1))

print(f"✓ {out}")
print(f"  task_index={args.task}  episodes={blk['n_episodes']}  "
      f"组内最大波动={blk['spread_max_dev']} rad")
print(f"  task: {task_sentence or '(未取到 tasks.parquet)'}")
print("  start_pose[0:8] =", blk["start_pose"][:8])
if args.task >= 0 and (ds / "episode_start_task0.json").exists() is False:
    print("  （warmup 零配置只认部署目录的 episode_start_task0.json；"
          "否则用 --pose-file 或 config [warmup].pose_file 指到本文件）")
