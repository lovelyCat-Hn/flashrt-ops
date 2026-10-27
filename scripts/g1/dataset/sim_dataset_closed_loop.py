"""离线闭环复现：数据集图像 + 模拟 state 递进（真机 execute 语义），不碰真机。
对比开环探针（真值 state）与闭环递归（state 被自己的输出推动）的分叉。
用法： ~/holy/run.sh /tmp/sim_closed_loop.py
"""
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402

CKPT = "/home/galbot/holy/models/pi05_g1_deploy"
NPZ = pathlib.Path("~/holy/datasets/replay_input.npz").expanduser()
EP, SEED, ROUNDS, NSTEP = 0, 0, 10, 3
DELTA_MAX, BUDGET = 0.05, 0.15

z = np.load(NPZ, allow_pickle=False)
states, ep_id, frame_idx = z["states"], z["ep_id"], z["frame_idx"]
tasks = z["tasks"].tolist()
cams = z["cams"].tolist()
video_template = str(z["video_template"])
vb = (pathlib.Path(str(z["dataset_root"])) if "dataset_root" in z
      else NPZ.parent)
rows = np.where(ep_id == EP)[0]
base = int(rows[0])
SDK2DS = {"image": "observation.images.head_right",
          "wrist_image": "observation.images.left_arm",
          "wrist_image_right": "observation.images.right_arm"}

_caps = {}


def grab(fidx):
    r = base + int(np.where(frame_idx[rows] == fidx)[0][0])
    out = {}
    for mkey, dskey in SDK2DS.items():
        ci = cams.index(dskey)
        path = str(vb / video_template.format(
            video_key=dskey, chunk_index=int(z["vid_chunk"][r, ci]),
            file_index=int(z["vid_file"][r, ci])))
        cap = _caps.get(path)
        if cap is None:
            cap = _caps[path] = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(z["vid_frame"][r, ci])
                + (fidx - int(frame_idx[r])))
        ok, img = cap.read()
        if not ok:
            sys.exit(f"解码失败 frame{fidx} {path}")
        out[mkey] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out


model = flash_rt.load_model(CKPT, config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
ns = model._pipe.norm_stats
PROMPT = str(tasks[int(z["task_idx"][base])])
print(f"prompt={PROMPT!r}")

model.predict(grab(0), prompt=PROMPT, state=normalize_state(states[base], ns))


def _acts(x):
    if isinstance(x, dict):
        x = x.get("actions", x.get("raw_actions",
                    next(v for v in x.values() if hasattr(v, "shape"))))
    return np.asarray(x, dtype=np.float32)


def infer(obs, fidx, st23):
    model.set_prompt(PROMPT, state=normalize_state(st23, ns))
    gen = torch.Generator().manual_seed(SEED + fidx)
    return _acts(model.infer(obs, noise=torch.randn(10, 32, generator=gen)))


# 模拟 state：23 维，从 ep0 帧 0 出发；腿/头固定，臂/夹爪被自己的输出推动
sim = states[base].copy()
ARM = np.r_[0:7, 8:15]          # 14 臂维（跳过两个夹爪 %）
f0 = states[base]
names = ([f"Rj{i}" for i in range(1, 8)] + [f"Lj{i}" for i in range(1, 8)])

print("\n轮 | 帧 | 闭环Δ步0 max | 闭环合步后臂位 vs 帧0 (max@关节) "
      "| 开环Δ步0 max（真值state探针）")
for r in range(ROUNDS):
    f = r * NSTEP
    rr = base + int(np.where(frame_idx[rows] == f)[0][0])
    chunk_c = infer(grab(f), f, sim)             # 闭环：state=模拟
    chunk_o = infer(grab(f), f, states[rr])      # 开环：state=数据集真值
    d_c = chunk_c[:NSTEP].copy()
    d_o = chunk_o[:NSTEP].copy()
    arm_d_c = np.concatenate([d_c[:, :7], d_c[:, 8:15]], axis=1)
    arm_d_o = np.concatenate([d_o[:, :7], d_o[:, 8:15]], axis=1)
    cmd = np.clip(arm_d_c.sum(axis=0), -BUDGET, BUDGET)   # execute 语义
    sim[:7] += cmd[:7]
    sim[8:15] += cmd[7:]
    off = sim[ARM] - f0[ARM]
    k = int(np.argmax(np.abs(off)))
    print(f"{r:>2} | {f:>3} | {np.abs(arm_d_c).max():.3f} rad"
          f" | {np.abs(off).max()*1000:5.0f} mrad @{names[k]}"
          f" | {np.abs(arm_d_o).max():.3f} rad")

print("\n模拟终态 vs 数据集各帧最近匹配：")
mx = np.abs(states[rows][:, ARM] - sim[ARM]).max(axis=1)
best = int(np.argmin(mx))
print(f"  最近帧 {int(frame_idx[rows[best]])}：max {mx[best]*1000:.0f} mrad")
print("\n模拟终态逐关节 vs 帧0：")
for j, i in enumerate(ARM):
    if abs(sim[i] - f0[i]) > 0.05:
        print(f"  {names[j]:>4}: {f0[i]:+.3f} → {sim[i]:+.3f}"
              f"  ({(sim[i]-f0[i])*1000:+.0f} mrad)")
