#!/usr/bin/env python
"""模块级耗时拆解探针（2026-10-07，W8A16 汇报用）——零 FlashRT 改动。

在 pipeline 相位边界打点（NO_GRAPH 模式下相位方法真实逐个调用）：
  图像预处理(前端 CPU) → 视觉编码器(3×SigLIP) → 编码器 18 层(产 KV)
  → 解码器(流匹配 10 步 × 18 层) → 拷贝/同步/后处理（余量）
graph-on 档相位方法被烘进图、重放期不触发，只报总时长（烟雾验证 W8A16 抓图可用）。

用法:
  ~/holy/run.sh /tmp/module_timing_probe.py --configs bf16_h50,w8a16_h50,g_w8a16_h50,bf16_h10,w8a16_h10
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
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_pick_deploy")
ap.add_argument("--dataset", default="/home/galbot/holy/datasets/only_place")
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--frames", default="0,60,120,180,240,300")
ap.add_argument("--configs", default="bf16_h50,w8a16_h50,g_w8a16_h50,bf16_h10,w8a16_h10")
ap.add_argument("--reps", type=int, default=18)
ap.add_argument("--npz", default="/tmp/ablation_only_place_ep0.npz")
args = ap.parse_args()

os.environ["PI05_NO_GRAPH"] = "1"          # 默认；g_* 档在循环里翻回 0
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
          "FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_INT8_ENC_SKIP",
          "FVK_PI05_RTX_W8A16_DECODER", "FVK_PI05_RTX_INT8_VISION"):
    os.environ.pop(k, None)

# ── npz 自举（与 rot 探针同源）──
DS = pathlib.Path(args.dataset).expanduser()
NPZ = pathlib.Path(args.npz)
if not NPZ.exists():
    print(f"npz 缺失，自举 {DS.name} ep{args.ep} → {NPZ} ...", flush=True)
    subprocess.run(["/usr/bin/python3",
                    "/home/galbot/holy/scripts/dataset/extract_dataset_frames.py",
                    "--dataset", str(DS), "--episodes", str(args.ep),
                    "--out", str(NPZ)], check=True)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

z = np.load(NPZ, allow_pickle=False)
states, actions = z["states"], z["actions"]
ep_id, frame_idx, task_idx = z["ep_id"], z["frame_idx"], z["task_idx"]
vid_chunk, vid_file = z["vid_chunk"], z["vid_file"]
tasks, cams = z["tasks"].tolist(), z["cams"].tolist()
video_template = str(z["video_template"])
video_base = (pathlib.Path(str(z["dataset_root"])) if "dataset_root" in z
              else NPZ.parent)

DS_OF = {"image": "observation.images.head_right",
         "wrist_image": "observation.images.left_arm",
         "wrist_image_right": "observation.images.right_arm"}

sel = np.where(ep_id == args.ep)[0]
FRAMES = [int(t) for t in args.frames.split(",")]
TASK = tasks[int(task_idx[sel[0]])]

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
from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline  # noqa: E402
from flash_rt.frontends.torch.pi05_rtx import (  # noqa: E402
    Pi05TorchFrontendRtx as _FE)

# ── 相位计时器（NO_GRAPH 下相位方法逐帧真实调用）──
T = {"vision": [], "enc": [], "dec": [], "fwd": [], "img": []}
_ORIG = {}

def _sync():
    torch.cuda.synchronize()

def _wrap_phase(key, name):
    orig = getattr(Pi05Pipeline, name)
    _ORIG[name] = orig
    def w(self, *a, **kw):
        _sync()
        t0 = time.perf_counter()
        r = orig(self, *a, **kw)
        _sync()
        T[key].append((time.perf_counter() - t0) * 1000)
        return r
    setattr(Pi05Pipeline, name, w)

_wrap_phase("vision", "vision_encoder")
_wrap_phase("enc", "transformer_encoder")
_wrap_phase("dec", "transformer_decoder")
_wrap_phase("fwd", "forward")

_orig_img = _FE._fill_img_buf
def _img_buf_w(self, observation):
    t0 = time.perf_counter()
    r = _orig_img(self, observation)
    T["img"].append((time.perf_counter() - t0) * 1000)
    return r
_FE._fill_img_buf = _img_buf_w

# chunk 注入（必须走原生 ctor 路径，api.load_model 按签名过滤 kwargs）
_ORIG_FE_INIT = _FE.__init__
_H = {"v": 0}

@functools.wraps(_ORIG_FE_INIT)
def _fe_init(self, *a, **kw):
    if _H["v"] > 0:
        kw["chunk_size"] = _H["v"]
    _ORIG_FE_INIT(self, *a, **kw)
_FE.__init__ = _fe_init

def fixed_noise(i: int, n: int):
    g = torch.Generator().manual_seed(20260930 + i)
    return torch.randn(n, 32, generator=g)

def med(x):
    return float(np.median(x)) if len(x) else float("nan")

RESULT = {}
for name in args.configs.split(","):
    name = name.strip()
    for k in ("FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_W8A16_DECODER"):
        os.environ.pop(k, None)
    graph_on = name.startswith("g_")
    if graph_on:
        name = name[2:]
    base = name.rsplit("_h", 1)
    variant, hor = base[0], int(base[1])
    _H["v"] = hor
    for k in T:
        T[k].clear()
    if graph_on:
        os.environ.pop("PI05_NO_GRAPH", None)
    else:
        os.environ["PI05_NO_GRAPH"] = "1"
    if variant == "bf16":
        os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
    elif variant == "w8a16":
        os.environ["FVK_PI05_RTX_W8A16_DECODER"] = "1"
    else:
        raise SystemExit(f"未知变体 {variant}")
    print(f"───── [{variant} h={hor} graph={'on' if graph_on else 'off'}] ─────",
          flush=True)

    t0 = time.time()
    model = flash_rt.load_model(args.ckpt, config="pi05", num_views=3,
                                cache_frames=1, action_dim=16)
    print(f"load {time.time() - t0:.0f}s chunk={model._fe_chunk if hasattr(model, '_fe_chunk') else '?'}",
          flush=True)
    if not graph_on:
        model._pipe.use_cuda_graph = False
    ns = model._pipe.norm_stats

    # 首帧预热（建管线/autotune；graph 档在此抓图）
    model.predict(IMGS[0], prompt=TASK, state=normalize_state(RSTS[0], ns))

    n = args.reps
    ts, ts_dec = [], []
    for r in range(n):
        i = r % len(IMGS)
        stn = normalize_state(RSTS[i], ns)
        for k in T:
            T[k].clear()
        t1 = time.perf_counter()
        model.infer(IMGS[i], noise=fixed_noise(r, hor))
        ts.append((time.perf_counter() - t1) * 1000)
    tot = med(ts)
    rec = {"total_p50": tot,
           "img": med(T["img"]), "vision": med(T["vision"]),
           "enc": med(T["enc"]), "dec": med(T["dec"]),
           "fwd": med(T["fwd"])}
    RESULT[(variant, hor, graph_on)] = rec
    phases = rec["img"] + rec["vision"] + rec["enc"] + rec["dec"]
    oth = rec["fwd"] - (rec["vision"] + rec["enc"] + rec["dec"])   # lang embeds 拷贝等
    rest = tot - rec["fwd"] - rec["img"]                           # 噪声填充/下载/后处理
    print(f"[{variant} h={hor} g={'on' if graph_on else 'off'}] "
          f"p50={tot:.1f} | 预处理CPU {rec['img']:.1f} | 视觉 {rec['vision']:.1f}"
          f" | 编码器 {rec['enc']:.1f} | 解码器 {rec['dec']:.1f}"
          f" | fwd内其他 {oth:.1f} | fwd外余量 {rest:.1f}", flush=True)
    del model
    torch.cuda.empty_cache()
    os.environ["PI05_NO_GRAPH"] = "1"   # 恢复默认，g_ 档只影响自己

print("\n===== 汇总（ms，p50，NO_GRAPH 相位打点 + graph-on 总时长）=====")
print(f"{'config':<18} {'total':>7} {'预处理':>7} {'视觉':>7} {'编码器':>7}"
      f" {'解码器':>7} {'fwd内其他':>9} {'fwd外余量':>9}")
for (variant, hor, g), r in sorted(RESULT.items(), key=lambda kv: str(kv[0])):
    oth = r["fwd"] - (r["vision"] + r["enc"] + r["dec"])
    rest = r["total_p50"] - r["fwd"] - r["img"]
    tag = f"{variant}_h{hor}" + ("_graph" if g else "")
    print(f"{tag:<18} {r['total_p50']:>7.1f} {r['img']:>7.1f} {r['vision']:>7.1f}"
          f" {r['enc']:>7.1f} {r['dec']:>7.1f} {oth:>9.1f} {rest:>9.1f}")
