#!/usr/bin/env python
"""从 lerobot 数据集提取 warm 就位姿态 → config/s1.toml [warm_pose] 片段。

配方 = policies/base.py S1_HOME_POSE 同款（10-29 提取）：每集首帧的逐关节
跨集中位（抗单集异位；各关节跨集散布 = 起点一致性的直接读数）。
⚠ 用数据集首帧而非 ckpt stats：state 统计是全程分布（含抓取中姿态），
mean/q50 不对应任何任务起点（G1 g1_warm_core 2026-09-23 实证臂维差 0.83 rad）。

用法（工作人员，换任务/换数据集时重跑一遍；pyarrow 住 lerobot_cpu env，
与 lerobot_to_traj.py 同款解释器）：
  ~/miniforge3/envs/lerobot_cpu/bin/python \
    ~/holy/scripts/tools/warm_pose_from_dataset.py --dataset ~/datasets/<数据集>
把输出的 [warm_pose] 片段贴进 ~/holy/config/s1.toml 即生效（traj 入口启动会
打印覆盖来源，可核对吃了哪组键）。G1 侧同需求走
scripts/g1/dataset/g1_pose_extract.py（episode_start json +
config/g1.toml [warmup].pose_file，机制早已在）。

维序按 meta/info.json 的 feature names 对号（不写死列号，数据集重排免疫）。
"""
import argparse
import json
import pathlib

import numpy as np

AP = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--dataset", default="/home/galbot/datasets/clean_ready_224_1028",
                help="lerobot 数据集根目录（meta/info.json + data/…）")
AP.add_argument("--episodes", default=None,
                help="只用部分集提取（如 '0,1,5-10'；缺省=全部）——排查单集异位用")
A = AP.parse_args()

root = pathlib.Path(A.dataset).expanduser()
info = json.loads((root / "meta" / "info.json").read_text())
st_names = list(info["features"]["observation.state"]["names"])
n_state = len(st_names)
print(f"数据集 {root.name}：{info['total_episodes']} 集 / {info['total_frames']} 帧，"
      f"state {n_state} 维，fps={info['fps']}")

# episode → data file（同 lerobot_to_traj 口径：episodes meta 拿 chunk/file）
ep_meta = {}
import pyarrow.parquet as pq  # noqa: E402
for mf in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
    d = pq.ParquetFile(mf).read(columns=[
        "episode_index", "length", "data/chunk_index", "data/file_index"]
        ).to_pydict()
    for i, ep in enumerate(d["episode_index"]):
        ep_meta[ep] = (d["data/chunk_index"][i], d["data/file_index"][i])
eps = sorted(ep_meta)
if A.episodes:
    want = set()
    for part in A.episodes.split(","):
        if "-" in part:
            a, b = part.split("-")
            want.update(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
    eps = [e for e in eps if e in want]
print(f"提取集：{len(eps)} 集 {eps if len(eps) <= 10 else str(eps[:10]) + '…'}")

# 每集首帧 state（data file 按 episode_index 列过滤后取第 0 行）
firsts = np.empty((len(eps), n_state), dtype=np.float64)
for i, ep in enumerate(eps):
    ch, fi = ep_meta[ep]
    f = root / "data" / f"chunk-{ch:03d}" / f"file-{fi:03d}.parquet"
    t = pq.ParquetFile(f).read(columns=["observation.state", "episode_index"])
    mask = np.array(t.column("episode_index").to_pylist()) == ep
    S = np.array(t.column("observation.state").to_numpy().tolist(), dtype=np.float64)
    firsts[i] = S[mask][0]                     # 首帧=起始位姿
med = np.median(firsts, axis=0)                # 逐关节跨集中位
spr = firsts.max(axis=0) - firsts.min(axis=0)  # 逐关节跨集散布

# ── 按名分组（写死列号是排序坑的祸根，全部对号入座）────────────────────────
idx = {nm: i for i, nm in enumerate(st_names)}


def group(prefix, n, kind="joint"):
    """right_arm_joint1..7 → 7 维；head_joint1.. → n 维。"""
    out = []
    for k in range(1, n + 1):
        nm = f"{prefix}_joint{k}" if kind == "joint" else f"{prefix}{k}"
        if nm not in idx:
            raise SystemExit(f"[extract] state 缺 {nm}（names={st_names}）")
        out.append(idx[nm])
    return out


R_ARM = group("right_arm", 7)
L_ARM = group("left_arm", 7)
HEAD = group("head", 2)
TORSO = [i for nm, i in idx.items() if nm.startswith("torso_joint")]
if not TORSO:
    raise SystemExit(f"[extract] state 缺 torso_joint*（names={st_names})")
GRIP_R = idx.get("right_arm_gripper")
GRIP_L = idx.get("left_arm_gripper")
if GRIP_R is None:
    raise SystemExit(f"[extract] state 缺 right_arm_gripper（names={st_names})")

pose = {
    "right_arm": [round(float(med[i]), 4) for i in R_ARM],
    "left_arm": [round(float(med[i]), 4) for i in L_ARM],
    "torso": round(float(med[TORSO[0]]), 4),
    "head": [round(float(med[i]), 4) for i in HEAD],
    "grip_pct": round(float(med[GRIP_R]), 1),
}

# ── 报告：分组散布 = 起点一致性；双爪互检 = 单爪驱动的任务要人工看 ──────────
print("\n── 跨集首帧散布（max−min；大=各集起点不一致，warm 到中位≠处处 in-distribution）──")
for label, dims in (("right_arm", R_ARM), ("left_arm", L_ARM), ("head", HEAD)):
    print(f"  {label:10s} 散布 max {spr[dims].max():.3f} rad "
          f"（j{int(np.argmax(spr[dims])) + 1} 最大）")
print(f"  torso      散布 {spr[TORSO[0]]:.3f} m")
if GRIP_L is not None:
    gd = float(np.abs(firsts[:, GRIP_R] - firsts[:, GRIP_L]).max())
    print(f"  双爪差异   max {gd:.1f} pct"
          + ("" if gd <= 1.0 else "  ⚠ 左右爪首帧不一致（单爪任务？grip_pct 取了右爪）"))

print(f"\n── 贴进 ~/holy/config/s1.toml（覆盖 policies/base.py 烘档对应组）──\n")
print("[warm_pose]")
print(f"right_arm = {pose['right_arm']}")
print(f"left_arm  = {pose['left_arm']}")
print(f"torso     = {pose['torso']}          # 升降高度 m（非 rad）")
print(f"head      = {pose['head']}")
print(f"grip_pct  = {pose['grip_pct']}")
print("\n（全部组都贴；只想覆盖部分组就只贴那几行，未贴的组沿用烘档。"
      "生效核对=traj 启动打印 `[warm] 姿态覆盖 config/s1.toml [warm_pose]`）")
