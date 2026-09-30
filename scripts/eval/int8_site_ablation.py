#!/usr/bin/env python
"""INT8 病位消融（方案 A，2026-09-30）——encoder 逐层/逐单元选择性量化扫描。

背景：09-23 tf_matrix 判 INT8 毁动作质量（全 INT8 cos 0.15 / 仅编码器 0.27 /
bf16 0.98），但那是「全有或全无」的结论。FlashRT 已打 per-site 选择性量化
补丁（FVK_PI05_RTX_INT8_ENC_SKIP，层/单元粒度回退 bf16），本工具用它扫出
损伤到底住在哪几层/哪个单元，回答「留几层 bf16 能把质量救回来、还能剩多少
加速」。

基线 = enc-only 档（编码器 INT8 + 解码器 bf16）：解码器 M=10 上 INT8 本来
就慢又毒，部署相关的自由度只在编码器。参照系 = 同机 bf16 档（同输入同噪声
直接对比，9-21 合成 A/B 的教训：真实帧才算数）。验收口径 = teacher-forced
对数据集 delta 的 cos（9-23 教训：只看直接 A/B 会漏判）。

用法（npz 缺失时自动用系统 python3 跑 extract_dataset_frames.py 自举）:
  ~/holy/run.sh ~/holy/scripts/eval/int8_site_ablation.py \
      [--ep 0] [--frames 0,60,120,180,240,300] \
      [--configs bf16,enc8,no_l0-8,no_l9-17] \
      [--ckpt ~/holy/models/pi05_g1_place_deploy] \
      [--npz /tmp/ablation_only_place_ep0.npz]
内置 config（冒号后是 ENC_SKIP spec）:
  bf16      （FORCE_BF16 参照档）
  enc8:     （编码器全 INT8，预期复现 0.27 崩档）
  no_attn:  L0-17:attn     no_ffn:  L0-17:ffn
  no_l0-8:  L0-8           no_l9-17: L9-17
  也可直接写 name:SPEC 自定义（如 q13:L8-11,ffn 同款语法）。
"""
import argparse
import functools
import os
import pathlib
import subprocess
import sys
import time

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_place_deploy")
ap.add_argument("--dataset", default="/home/galbot/holy/datasets/only_place")
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--frames", default="0,60,120,180,240,300")
ap.add_argument("--configs", default="bf16,enc8,no_l0-8,no_l9-17,no_attn,no_ffn")
ap.add_argument("--npz", default=None)
ap.add_argument("--bench", type=int, default=0,
                help="每档追加 N 次稳态 infer 计时（末帧），得干净延迟")
ap.add_argument("--out", default="/tmp/int8_site_ablation.npz")
args = ap.parse_args()

os.environ["PI05_NO_GRAPH"] = "1"          # eager：确定性 + 消 graph 变量
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
          "FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_INT8_ENC_SKIP",
          "FVK_PI05_RTX_INT8_VISION"):
    os.environ.pop(k, None)

# ── npz 自举（系统 python3 跑 extract_dataset_frames，flash 环境无 pandas）──
DS = pathlib.Path(args.dataset).expanduser()
NPZ = pathlib.Path(args.npz) if args.npz else pathlib.Path(
    f"/tmp/ablation_{DS.name}_ep{args.ep}.npz")
if not NPZ.exists():
    print(f"npz 缺失，自举 {DS.name} ep{args.ep} → {NPZ} ...", flush=True)
    subprocess.run(["/usr/bin/python3",
                    "/home/galbot/holy/scripts/inference/extract_dataset_frames.py",
                    "--dataset", str(DS), "--episodes", str(args.ep),
                    "--out", str(NPZ)], check=True)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

z = np.load(NPZ, allow_pickle=False)
states, actions = z["states"], z["actions"]
ep_id, frame_idx, task_idx = z["ep_id"], z["frame_idx"], z["task_idx"]
vid_chunk, vid_file, vid_frame = z["vid_chunk"], z["vid_file"], z["vid_frame"]
tasks, cams = z["tasks"].tolist(), z["cams"].tolist()
video_template = str(z["video_template"])
video_base = (pathlib.Path(str(z["dataset_root"])) if "dataset_root" in z
              else NPZ.parent)

DS_OF = {"image": "observation.images.head_right",
         "wrist_image": "observation.images.left_arm",
         "wrist_image_right": "observation.images.right_arm"}
ARM_IDX = np.r_[0:7, 8:15]        # 14 臂维（dim7/15 夹爪 %；右臂在前，见 info.json）
GRIP_IDX = [7, 15]

sel = np.where(ep_id == args.ep)[0]
ep_len = int(frame_idx[sel].max()) + 1
FRAMES = [min(int(t), ep_len - 11) for t in
          (int(x) for x in args.frames.split(","))]
TASK = tasks[int(task_idx[sel[0]])]
print(f"{DS.name} ep{args.ep}（{ep_len} 帧）frames={FRAMES}\nTASK: {TASK}\n",
      flush=True)

caps = {}
def grab(t: int) -> dict:
    row = sel[int(np.searchsorted(frame_idx[sel], t))]
    out = {}
    for vkey, dskey in DS_OF.items():
        ci = cams.index(dskey)
        path = str(video_base / video_template.format(
            video_key=dskey, chunk_index=int(vid_chunk[row, ci]),
            file_index=int(vid_file[row, ci])))
        cap = caps.get(path)
        if cap is None:
            cap = caps[path] = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"取帧失败 ep{args.ep} t={t} {dskey}")
        out[vkey] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out

IMGS = [grab(t) for t in FRAMES]
RSTS = [states[sel[t]].astype(np.float32) for t in FRAMES]

# ── flash_rt ──
sys.path.insert(0, "/home/galbot/holy")
from cuda_warmup import cuda_warmup  # noqa: E402
cuda_warmup()
import flash_rt  # noqa: E402
import torch  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402

def fixed_noise(i: int):
    g = torch.Generator().manual_seed(20260930 + i)
    return torch.randn(10, 32, generator=g)

from flash_rt.frontends.torch.pi05_rtx import _parse_int8_enc_skip  # noqa: E402
def _skip_names(spec: str) -> set:
    """预期被 SKIP 的权重名数（与 FlashRT 同一 parser，交叉验证用）。"""
    return _parse_int8_enc_skip(spec, 18) if spec else set()

def parse_config(name_spec: str):
    if ":" in name_spec:
        name, spec = name_spec.split(":", 1)
    else:
        name, spec = name_spec, {"bf16": "", "enc8": "", "no_attn": "L0-17:attn",
                                 "no_ffn": "L0-17:ffn", "no_l0-8": "L0-8",
                                 "no_l9-17": "L9-17"}.get(name_spec, "")
    return name, spec

CKPT = str(pathlib.Path(args.ckpt).expanduser())
store, lat = {}, {}
for name_spec in args.configs.split(","):
    name, spec = parse_config(name_spec.strip())
    for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
              "FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_INT8_ENC_SKIP"):
        os.environ.pop(k, None)
    if name == "bf16":
        os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
    else:
        os.environ["FVK_PI05_RTX_INT8_ENCODER_ONLY"] = "1"   # 解码器恒 bf16
        if spec:
            # CLI 层 spec 内部分隔符用 +（逗号留给配置表），下发时转回
            os.environ["FVK_PI05_RTX_INT8_ENC_SKIP"] = spec.replace("+", ",")
    print(f"───── [{name}] ENC_SKIP='{spec}' 加载 {pathlib.Path(CKPT).name} ─────",
          flush=True)
    t0 = time.time()
    model = flash_rt.load_model(CKPT, config="pi05", num_views=3,
                                cache_frames=1, action_dim=16)
    print(f"load {time.time() - t0:.0f}s", flush=True)
    ns = model._pipe.norm_stats

    outs, ts = [], []
    n_enc8 = -1
    for i, imgs in enumerate(IMGS):
        stn = normalize_state(RSTS[i], ns)
        model.predict(imgs, prompt=TASK, state=stn)          # 设 prompt/state
        if n_enc8 < 0:    # 首次 predict 建好 pipeline，验证补丁生效
            n_enc8 = sum(1 for k in model._pipe.pipeline.weights.get("int8", {})
                         if k.startswith("encoder_"))
            want = 90 - len(_skip_names(spec.replace("+", ",")))
            print(f"编码器 INT8 位点 {n_enc8}/90（ENC_SKIP='{spec}' → 期望 {want}/90）",
                  flush=True)
        t1 = time.perf_counter()
        r = model.infer(imgs, noise=fixed_noise(i))
        ts.append((time.perf_counter() - t1) * 1000)
        outs.append(np.asarray(r["actions"], np.float32))
    outs = np.stack(outs)
    store[name] = outs
    lat[name] = float(np.median(ts))

    # 确定性自检（同噪声双跑应严格 0；非零则该档对比作废）
    model.predict(IMGS[0], prompt=TASK, state=normalize_state(RSTS[0], ns))
    d = np.abs(np.asarray(model.infer(IMGS[0], noise=fixed_noise(0))["actions"],
                          np.float32) - outs[0]).max()
    print(f"[{name}] p50 {np.median(ts):.0f} ms | 确定性 max_diff={d:.2e}"
          f"{' ⚠ 非零' if d != 0 else ''}", flush=True)

    if args.bench > 0:
        bt = []
        imgs_b = IMGS[-1]
        stn_b = normalize_state(RSTS[-1], ns)
        for i in range(args.bench):
            model.predict(imgs_b, prompt=TASK, state=stn_b)
            t1 = time.perf_counter()
            model.infer(imgs_b, noise=fixed_noise(i))
            bt.append((time.perf_counter() - t1) * 1000)
        lat[name] = float(np.median(bt))
        print(f"[{name}] bench N={args.bench} p50 {lat[name]:.0f} "
              f"(p10 {np.percentile(bt, 10):.0f} / p90 {np.percentile(bt, 90):.0f})",
              flush=True)
    np.savez(args.out, **{f"cfg_{k}": v for k, v in store.items()})
    del model
    torch.cuda.empty_cache()

# ── 汇总 ──
def cos(a, b):
    a, b = a.ravel(), b.ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else float("nan")

ref = store["bf16"]
print("\n===== ① vs bf16 直接对比（臂维 cos / 夹爪 MAE pp） =====")
print(f"{'config':<12} {'cos_arm':>8} {'cos_grip':>9} {'grip|d|pp':>10} "
      f"{'p50 ms':>7} {'tf-cos':>7}")
tf_all = {}
for name, outs in store.items():
    ca = [cos(outs[i][:, ARM_IDX].mean(axis=0),
              ref[i][:, ARM_IDX].mean(axis=0)) for i in range(len(FRAMES))]
    cg = [cos(outs[i][:, GRIP_IDX].mean(axis=0),
              ref[i][:, GRIP_IDX].mean(axis=0)) for i in range(len(FRAMES))]
    gd = [np.abs(outs[i][:, GRIP_IDX] - ref[i][:, GRIP_IDX]).mean()
          for i in range(len(FRAMES))]

    # teacher-forced：块均 delta vs 数据集 10 帧 delta（9-23 判决口径）
    tcs = []
    for i, t in enumerate(FRAMES):
        m10 = outs[i][:, ARM_IDX].mean(axis=0)
        ds10 = np.stack([actions[row][ARM_IDX] - states[row][ARM_IDX]
                         for row in sel[t:t + 10]]).mean(axis=0)
        tcs.append(cos(m10, ds10))
    tf_all[name] = float(np.mean(tcs))
    print(f"{name:<12} {np.mean(ca):>8.4f} {np.mean(cg):>9.4f} "
          f"{np.mean(gd):>10.2f} {lat[name]:>7.0f} {np.mean(tcs):>7.3f}")

print("""
判读:
  ① cos_arm 直接对 bf16——损伤定位信号。enc8 应显著低于 1；哪一档把 cos
    拉回 ≈1，损伤就住在被 SKIP 掉的那批位点里。
  tf-cos 对数据集（≥0.9 且夹爪正常才算「可上真机」，bf16 基线 ≈0.98）。
  两维交叉：tf-cos 高 + cos_arm 高 → 该档可作部署候选；再对它细扫半区定位。
  ⚠ 混布档的 bf16 回退 GEMM 未过 cuBLASLt autotune（启发式选型），延迟
    读数略吃亏；质量（cos）不受影响，选出候选档后再单独 bench 延迟。""")
