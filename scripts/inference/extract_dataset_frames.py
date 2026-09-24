#!/usr/bin/env python
"""pick_place_balence 数据集 → npz 预处理（系统 python3 跑，需 pandas+pyarrow）。

读 lerobot v3.0 的 parquet（data + meta/episodes + meta/tasks + meta/stats），
抽出指定 episode 的全帧 state(23)/action(16)/task 与视频寻址（哪个 mp4 的
第几帧），存成一个 npz 给 run_dataset_replay.py（flash_pyrt311，无 pandas——
numpy==1.26.4 钉版红线，勿往推理环境装 pandas/pyarrow）。

维序以 meta/info.json features.names 为唯一权威（历史教训：action 16 维
【右臂在前】= 右臂7+右夹爪%+左臂7+左夹爪%；state 23 维 = 16+leg5+head2；
曾记反把机械臂甩背后）。夹爪是 0~100% 训练原单位，npz 原样保存零换算。

用法:
  python3 ~/holy/scripts/inference/extract_dataset_frames.py \
      [--dataset ~/holy/datasets/pick_place_balence] \
      [--episodes 0,1,2] [--out ~/holy/datasets/replay_input.npz]
"""
import argparse
import glob
import json
import pathlib

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", default="~/holy/datasets/pick_place_balence")
ap.add_argument("--episodes", default="0", help="逗号分隔的 episode 号，或 all")
ap.add_argument("--out", default=None, help="npz 输出路径（默认 <dataset>/../replay_input.npz）")
args = ap.parse_args()

ds = pathlib.Path(args.dataset).expanduser()
info = json.loads((ds / "meta" / "info.json").read_text())
fps = int(info["fps"])
feat = info["features"]
ST_DIM = int(feat["observation.state"]["shape"][0])
AC_DIM = int(feat["action"]["shape"][0])
assert (ST_DIM, AC_DIM) == (23, 16), \
    f"预期 state 23/action 16，实际 {ST_DIM}/{AC_DIM}——维序判读结论已过时？"
CAM_KEYS = sorted(k for k in feat if k.startswith("observation.images."))
print(f"dataset: {ds.name} | fps={fps} | episodes={info['total_episodes']} "
      f"| frames={info['total_frames']} | cams={CAM_KEYS}")

# ── tasks（本数据集：句子在 DataFrame 索引、编号在 task_index 列）──
tasks_df = pd.read_parquet(ds / "meta" / "tasks.parquet")
if "task" in tasks_df.columns:
    order = tasks_df.sort_values("task_index")
    tasks = order["task"].astype(str).tolist()
else:  # sentences-in-index 布局（pick_place_balence 实际如此）
    order = tasks_df.sort_values("task_index")
    tasks = [str(x) for x in order.index]
print(f"tasks: {tasks}")

# ── episodes 表（v3.0：每相机独立寻址列 videos/<cam>/{chunk_index,file_index,from_timestamp}）──
ep_files = sorted(glob.glob(str(ds / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
assert ep_files, "meta/episodes 下没有 parquet？"
ep = pd.concat(pd.read_parquet(f) for f in ep_files).sort_values("episode_index")
need = {"episode_index", "length"} | {f"videos/{c}/{k}" for c in CAM_KEYS
                                      for k in ("chunk_index", "file_index", "from_timestamp")}
missing = need - set(ep.columns)
assert not missing, f"episodes 表缺列 {missing}（实际列：{list(ep.columns)}）——v3.0 布局假设失效"

sel = (list(range(int(ep["episode_index"].max()) + 1))
       if args.episodes == "all" else [int(x) for x in args.episodes.split(",")])
ep = ep[ep["episode_index"].isin(sel)]
# 每相机的 (chunk, file, 帧偏移=round(from_timestamp*fps))——表内权威值，零重构
ep_map = {}
for _, r in ep.iterrows():   # 列名带斜杠，itertuples 会改名，必须 iterrows
    ep_map[int(r["episode_index"])] = (
        int(r["length"]),
        [(int(r[f"videos/{c}/chunk_index"]), int(r[f"videos/{c}/file_index"]),
          int(round(float(r[f"videos/{c}/from_timestamp"]) * fps)))
         for c in CAM_KEYS])
print(f"选中 {len(ep_map)} 条: " + ", ".join(
    f"ep{e}(len {v[0]})" for e, v in sorted(ep_map.items())))

# ── 数据 parquet（多 episode 混在一个 file 里，按 episode_index 过滤）──
data_files = sorted(glob.glob(str(ds / "data" / "chunk-*" / "file-*.parquet")))
assert data_files, "data/ 下没有 parquet？"
frames = []
for f in data_files:
    df = pd.read_parquet(f)
    if "episode_index" not in df.columns:
        continue
    df = df[df["episode_index"].isin(ep_map)]
    if len(df):
        frames.append(df)
assert frames, "选中的 episode 在数据文件里一帧都没捞到？"
df = pd.concat(frames).sort_values(["episode_index", "frame_index"])

states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
actions = np.stack(df["action"].to_numpy()).astype(np.float32)
ep_id = df["episode_index"].to_numpy().astype(np.int32)
frame_idx = df["frame_index"].to_numpy().astype(np.int32)
task_idx = df["task_index"].to_numpy().astype(np.int32)
assert states.shape[1] == ST_DIM and actions.shape[1] == AC_DIM

# 每帧视频寻址（N × 相机数；列序 = CAM_KEYS）
vc = np.array([[ep_map[int(e)][1][k][0] for k in range(len(CAM_KEYS))] for e in ep_id], np.int32)
vf = np.array([[ep_map[int(e)][1][k][1] for k in range(len(CAM_KEYS))] for e in ep_id], np.int32)
vframe = np.array([[ep_map[int(e)][1][k][2] for k in range(len(CAM_KEYS))]
                   for e in ep_id], np.int32) + frame_idx[:, None]
# 与数据文件对照：长度守恒
for e, (ln, _) in ep_map.items():
    got = int((ep_id == e).sum())
    assert got == ln, f"ep{e} 数据帧数 {got} != episodes 表 length {ln}"

# ── action 统计（供回放脚本做落域/OOD 判读）──
st = json.loads((ds / "meta" / "stats.json").read_text())
def svec(key):
    a = np.asarray(st["action"][key], np.float32)
    return a
act_q01, act_q99 = svec("q01"), svec("q99")
act_mean = svec("mean")

out = pathlib.Path(args.out) if args.out else ds.parent / "replay_input.npz"
np.savez_compressed(
    out,
    states=states, actions=actions,
    ep_id=ep_id, frame_idx=frame_idx, task_idx=task_idx,
    vid_chunk=vc, vid_file=vf, vid_frame=vframe,
    tasks=np.array(tasks), cams=np.array(CAM_KEYS),
    act_q01=act_q01, act_q99=act_q99, act_mean=act_mean,
    state_names_json=json.dumps(feat["observation.state"].get("names", [])),
    action_names_json=json.dumps(feat["action"].get("names", [])),
    video_template=str(info.get("video_path", "")),
    dataset_root=str(ds),   # 回放脚本按它解析视频相对路径（npz 不在数据集目录里）
    fps=np.int32(fps),
    ep_list=np.array(sorted(ep_map)),
    ep_offsets=np.array([int((ep_id == e).argmax()) for e in sorted(ep_map)], np.int32),
)
print(f"\n✅ {len(df)} 帧 → {out} ({out.stat().st_size/1e6:.1f} MB)")

# 自检：episode0 第0帧 = 标准工作位（记忆锚点：leg 0.580/1.433/0.902/0/0 + head −0.061/0.330）
if 0 in ep_map:
    r0 = states[ep_id == 0][0]
    print(f"自检 ep0 第0帧 leg[16:21]={np.round(r0[16:21],3).tolist()} "
          f"head[21:23]={np.round(r0[21:23],3).tolist()}")
    print("（应≈ 0.580/1.433/0.902/0/0 与 −0.061/0.330；偏差大=维序错位，停！）")
