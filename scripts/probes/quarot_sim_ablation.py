#!/usr/bin/env python
"""QuaRot 旋转 W8A8 仿真消融（Route B′ Phase 1，2026-09-30）——零 FlashRT 改动。

背景：09-30 消融判定编码器 W8A8 毁动作（cos 0.09），根因=Gemma per-channel
激活离群 × per-row 动态 scale。Route B′ 假设：每个 int8 GEMM 前对激活做
Hadamard 旋转 x' = x·H/√K（权重侧离线 W' = W·H/√K），把离群摊平到全通道，
per-row scale 就能活。数学精确：(xH)(WH)ᵀ = xWᵀ（H 对合正交）。

本脚本用 monkeypatch 仿真整条链路，不写任何 kernel：
  权重侧  load_model 后、首次 predict（惰性建管线+量化）前，把
          _pipe._ckpt_bf16 的编码器权重沿 dim0(K) 做 FWHT/√K ——
          _quantize_encoder_int8 照常消费旋转后的张量。
  激活侧  替换 Pi05Pipeline._enc_int8_gemm：cudaMemcpy D2H 取 x(M,K bf16)
          → torch FWHT/√K → per-row amax/127 int8（与 quantize_int8_rowwise
          同式）→ H2D 写回同一 scratch/scale buffer → 原样调 CUTLASS GEMM。
          同步拷贝保证时序正确；质量判决不看延迟（sim 档延迟必虚高）。

四档对照（同帧同噪声，验收口径=tf-cos，bf16 基线≈0.95）：
  bf16      参照档（FORCE_BF16）
  int8      kernel 原生量化（复现 0.09 崩档，对照组）
  int8_sim  torch 量化、无旋转（harness 契约自检：应≈int8 档）
  rot8      FWHT 旋转 + torch 量化（判决档）

判决：rot8 tf-cos ≥0.90（且夹爪不崩）→ 机制成立，进 Phase 2 写
FHT+int8 融合 kernel；仍崩 → Route B′ 出局，编码器维持 bf16。

用法:
  ~/holy/run.sh ~/holy/scripts/probes/quarot_sim_ablation.py \
      [--configs bf16,int8,int8_sim,rot8] [--ep 0] [--frames 0,60,120,180,240,300]
"""
import argparse
import ctypes
import math
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
ap.add_argument("--configs", default="bf16,int8,int8_sim,rot8")
ap.add_argument("--npz", default="/tmp/ablation_only_place_ep0.npz")
ap.add_argument("--out", default="/tmp/quarot_sim_ablation.npz")
args = ap.parse_args()

os.environ["PI05_NO_GRAPH"] = "1"
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
          "FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_INT8_ENC_SKIP",
          "FVK_PI05_RTX_INT8_VISION"):
    os.environ.pop(k, None)

# ── npz 自举（与 int8_site_ablation 同源）──
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
vid_chunk, vid_file, vid_frame = z["vid_chunk"], z["vid_file"], z["vid_frame"]
tasks, cams = z["tasks"].tolist(), z["cams"].tolist()
video_template = str(z["video_template"])
video_base = (pathlib.Path(str(z["dataset_root"])) if "dataset_root" in z
              else NPZ.parent)

DS_OF = {"image": "observation.images.head_right",
         "wrist_image": "observation.images.left_arm",
         "wrist_image_right": "observation.images.right_arm"}
ARM_IDX = np.r_[0:7, 8:15]
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
from flash_rt.core.cuda_buffer import _cudart as _rt, _check  # noqa: E402
from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline  # noqa: E402
from flash_rt.frontends.torch.pi05_rtx import (  # noqa: E402
    Pi05TorchFrontendRtx as _FE)

def fixed_noise(i: int):
    g = torch.Generator().manual_seed(20260930 + i)
    return torch.randn(10, 32, generator=g)

# ── FWHT（未归一化）+ 正确性自检 ──
def fwht(x: torch.Tensor) -> torch.Tensor:
    """x: (M, K) fp32，K 为 2 的幂。返回未归一化 Walsh-Hadamard 变换。"""
    M, K = x.shape
    h = 1
    while h < K:
        x = x.view(M, K // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2).view(M, K)
        h *= 2
    return x

_v = torch.randn(64, 128)
assert torch.allclose(fwht(fwht(_v)), _v * 128, atol=1e-3), "FWHT 对合性自检失败"
del _v

def rotate_encoder_weights(pipe) -> None:
    """编码器 2-D 权重沿 dim0(K) 旋转：W ← FWHT(W)/√K（(K,N) 布局，bf16_nn）。"""
    W = pipe._ckpt_bf16
    n_rot = 0
    for key in list(W.keys()):
        if not (key.startswith("encoder_") and key.endswith("_w")):
            continue
        lst = W[key]
        for i, w in enumerate(lst):
            if w.dim() != 2:
                continue
            K, N = w.shape
            # 权重存 (K, N)：旋转沿 K（输入通道）= 对转置后的 last dim 做 FWHT
            wr = fwht(w.float().t().contiguous()).t().contiguous() / math.sqrt(K)
            # 归一化 Hadamard 保 L2 范数：逐行能量不变 → 旋转正确性的强校验
            e0 = float(w.float().pow(2).sum())
            e1 = float(wr.pow(2).sum())
            assert abs(e1 - e0) / e0 < 1e-3, f"{key}[{i}] 旋转能量漂移 {e0}->{e1}"
            lst[i] = wr.to(w.dtype).contiguous()
            n_rot += 1
    print(f"  权重旋转完成：{n_rot} 个编码器张量", flush=True)

# 量化在前端构造函数内部执行（pi05_rtx.py 构造段 _quantize_encoder_int8），
# 旋转必须包在量化之前 —— post-load 旋转会得到「旋转激活×未旋转权重」的结构性
# 错值（首轮实测 cos −0.01 / grip 43pp 即此签名）。
_ORIG_QENC = _FE._quantize_encoder_int8

def _rot_qenc(self):
    rotate_encoder_weights(self)      # self = 前端，持 _ckpt_bf16
    print("  → _quantize_encoder_int8（消费旋转后权重）", flush=True)
    return _ORIG_QENC(self)

# ── 激活侧 monkeypatch（4 量化位点/层：qkv、o、gate_up、down）──
# 位点→路径：o/down 走 _enc_int8_gemm；qkv 走融合 fvk.rms_norm_int8_rowwise；
# gate_up 走融合 fvk.residual_add_rms_norm_int8_rowwise（gate/up 共享一份量化输入）。
_ORIG_ENC_I8 = Pi05Pipeline._enc_int8_gemm
_SIM_ROT = False
_SIM_CALLS = 0
_SIM_LOG = None        # 非空时记录 (位点, M, K) 诊断
_FKV = None            # flash_rt_kernels 模块对象（patch_sim 时绑定）
_ORIG_FUSED = {}

def _sim_quant_from_dev(src_ptr, i8_ptr, scale_ptr, M, K):
    """device bf16 (M,K) → [FWHT/√K] → per-row amax/127 int8 → 写回 device。"""
    torch.cuda.synchronize()          # 设备级同步：任意流上 src 均已就绪
    host = np.empty(M * K, dtype=np.uint16)
    _check(_rt.cudaMemcpy(host.ctypes.data, ctypes.c_void_p(src_ptr),
                          M * K * 2, 2), "sim D2H")
    x = torch.from_numpy(
        (host.astype(np.uint32) << 16).view(np.float32)).view(M, K).float().cuda()
    if _SIM_ROT:
        x = fwht(x) / math.sqrt(K)
    s = torch.clamp(x.abs().amax(dim=1) / 127.0, min=1e-12)
    q = torch.clamp(torch.round(x / s[:, None]), -127, 127).to(torch.int8)
    torch.cuda.synchronize()
    q_np = q.cpu().numpy().reshape(-1)
    s_np = s.cpu().numpy().astype(np.float32)
    _check(_rt.cudaMemcpy(ctypes.c_void_p(i8_ptr), q_np.ctypes.data,
                          M * K, 1), "sim H2D int8")
    _check(_rt.cudaMemcpy(ctypes.c_void_p(scale_ptr), s_np.ctypes.data,
                          M * 4, 1), "sim H2D scale")

def _sim_enc_int8(self, act_ptr, act_n, weight_name, out_ptr, M, N, K, stream):
    """o / down 位点：helper 路径——换量化，GEMM 原样。"""
    global _SIM_CALLS
    if not weight_name.startswith("encoder_") or \
            torch.cuda.is_current_stream_capturing():
        return _ORIG_ENC_I8(self, act_ptr, act_n, weight_name,
                            out_ptr, M, N, K, stream)
    _SIM_CALLS += 1
    if _SIM_LOG is not None:
        _SIM_LOG.append((weight_name, M, K))
    act_i8_ptr = self._pick_enc_int8_scratch(act_n)
    scale_buf = self._int8_scale_buf(weight_name, M)
    _sim_quant_from_dev(act_ptr, act_i8_ptr, scale_buf.ptr.value, M, K)
    self._int8_gemm_fused(act_i8_ptr, weight_name, out_ptr, M, N, K,
                          scale_buf.ptr.value, stream)

def _sim_qkv(x_ptr, rms_w, i8p, sp, seq, K, eps, stream=0):
    """qkv 位点：融合 rms+int8 → rms_norm 写 bf16 临时 + 仿真量化。"""
    global _SIM_CALLS
    if torch.cuda.is_current_stream_capturing():
        return _ORIG_FUSED["rms_norm_int8_rowwise"](
            x_ptr, rms_w, i8p, sp, seq, K, eps, stream=stream)
    _SIM_CALLS += 1
    if _SIM_LOG is not None:
        _SIM_LOG.append(("qkv", seq, K))
    tmp = torch.empty(seq, K, dtype=torch.bfloat16, device="cuda")
    _FKV.rms_norm(x_ptr, rms_w, tmp.data_ptr(), seq, K, eps, stream=stream)
    _sim_quant_from_dev(tmp.data_ptr(), i8p, sp, seq, K)

def _sim_gu(x_ptr, xnorm_ptr, rms_w, i8p, sp, seq, K, eps, stream=0):
    """gate/up 位点：融合 residual+rms+int8 → 非融合对（与 fp8 预校准路径
    同款调用）+ 仿真量化，x/x_norm 残差语义不变。"""
    global _SIM_CALLS
    if torch.cuda.is_current_stream_capturing():
        return _ORIG_FUSED["residual_add_rms_norm_int8_rowwise"](
            x_ptr, xnorm_ptr, rms_w, i8p, sp, seq, K, eps, stream=stream)
    _SIM_CALLS += 1
    if _SIM_LOG is not None:
        _SIM_LOG.append(("gu", seq, K))
    _FKV.residual_add(x_ptr, xnorm_ptr, seq * K, stream=stream)
    _FKV.rms_norm(x_ptr, rms_w, xnorm_ptr, seq, K, eps, stream=stream)
    _sim_quant_from_dev(xnorm_ptr, i8p, sp, seq, K)

def patch_sim(rot: bool, pipe) -> None:
    global _SIM_ROT, _FKV
    _SIM_ROT = rot
    _FKV = pipe.fvk
    _ORIG_FUSED["rms_norm_int8_rowwise"] = _FKV.rms_norm_int8_rowwise
    _ORIG_FUSED["residual_add_rms_norm_int8_rowwise"] = \
        _FKV.residual_add_rms_norm_int8_rowwise
    _FKV.rms_norm_int8_rowwise = _sim_qkv
    _FKV.residual_add_rms_norm_int8_rowwise = _sim_gu
    Pi05Pipeline._enc_int8_gemm = _sim_enc_int8

def unpatch_sim() -> None:
    if _FKV is not None:
        _FKV.rms_norm_int8_rowwise = _ORIG_FUSED["rms_norm_int8_rowwise"]
        _FKV.residual_add_rms_norm_int8_rowwise = \
            _ORIG_FUSED["residual_add_rms_norm_int8_rowwise"]
    Pi05Pipeline._enc_int8_gemm = _ORIG_ENC_I8

# ── 主循环 ──
CKPT = str(pathlib.Path(args.ckpt).expanduser())
store, lat = {}, {}
for name in args.configs.split(","):
    name = name.strip()
    for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
              "FVK_PI05_RTX_FORCE_BF16"):
        os.environ.pop(k, None)
    do_patch = name in ("int8_sim", "rot8")
    if name == "bf16":
        os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
    else:
        os.environ["FVK_PI05_RTX_INT8_ENCODER_ONLY"] = "1"
    print(f"───── [{name}] ─────", flush=True)
    if name == "rot8":
        _FE._quantize_encoder_int8 = _rot_qenc   # 旋转在 load 内部、量化之前
    t0 = time.time()
    model = flash_rt.load_model(CKPT, config="pi05", num_views=3,
                                cache_frames=1, action_dim=16)
    if name == "rot8":
        _FE._quantize_encoder_int8 = _ORIG_QENC
    print(f"load {time.time() - t0:.0f}s", flush=True)
    # 关 graph：calibrate 首帧会捕获 infer graph 并在推理期重放，
    # 重放会静默绕过 monkeypatch（PI05_NO_GRAPH 不被此前端消费，勿依赖）
    model._pipe.use_cuda_graph = False
    ns = model._pipe.norm_stats

    if do_patch:
        patch_sim(rot=(name == "rot8"), pipe=model._pipe)

    outs, ts = [], []
    n_enc8 = -1
    for i, imgs in enumerate(IMGS):
        stn = normalize_state(RSTS[i], ns)
        c0 = _SIM_CALLS
        model.predict(imgs, prompt=TASK, state=stn)
        c1 = _SIM_CALLS
        if n_enc8 < 0:
            n_enc8 = sum(1 for k in model._pipe.pipeline.weights.get("int8", {})
                         if k.startswith("encoder_"))
            print(f"编码器 INT8 位点 {n_enc8}/90", flush=True)
            if name == "rot8":
                assert n_enc8 == 90, "rot8 档必须全位点 INT8（否则 bf16 路会读到旋转权重）"
        t1 = time.perf_counter()
        if do_patch and i == 0:
            _SIM_LOG = []
        r = model.infer(imgs, noise=fixed_noise(i))
        ts.append((time.perf_counter() - t1) * 1000)
        outs.append(np.asarray(r["actions"], np.float32))
        if do_patch and i == 0:
            from collections import Counter
            cnt = Counter(e[0] if isinstance(e[0], str) and e[0] in
                          ("qkv", "gu") else "o/down" for e in _SIM_LOG)
            print(f"  sim patch 调用：predict(calibrate) +{c1 - c0}，"
                  f"infer +{_SIM_CALLS - c1}（期望 69=qkv18 + o/gu/down 各17，"
                  f"末层 early-return）| frame0 分布 {dict(cnt)}", flush=True)
            assert _SIM_CALLS - c1 == 69, \
                f"infer 期覆盖 { _SIM_CALLS - c1 } ≠ 69——有位点被绕过"
    outs = np.stack(outs)
    store[name] = outs
    lat[name] = float(np.median(ts))
    np.savez(args.out, **{f"cfg_{k}": v for k, v in store.items()})

    # 确定性自检
    model.predict(IMGS[0], prompt=TASK, state=normalize_state(RSTS[0], ns))
    d = np.abs(np.asarray(model.infer(IMGS[0], noise=fixed_noise(0))["actions"],
                          np.float32) - outs[0]).max()
    print(f"[{name}] p50 {np.median(ts):.0f} ms（sim 档虚高勿读）| "
          f"确定性 max_diff={d:.2e}{' ⚠ 非零' if d != 0 else ''}", flush=True)

    if do_patch:
        unpatch_sim()
    del model
    torch.cuda.empty_cache()

# ── 汇总 ──
def cos(a, b):
    a, b = a.ravel(), b.ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else float("nan")

ref = store["bf16"]
print("\n===== QuaRot 仿真消融（① vs bf16 直接对比 / ② tf-cos） =====")
print(f"{'config':<10} {'cos_arm':>8} {'cos_grip':>9} {'grip|d|pp':>10} {'tf-cos':>7}")
for name, outs in store.items():
    ca = [cos(outs[i][:, ARM_IDX].mean(axis=0),
              ref[i][:, ARM_IDX].mean(axis=0)) for i in range(len(FRAMES))]
    cg = [cos(outs[i][:, GRIP_IDX].mean(axis=0),
              ref[i][:, GRIP_IDX].mean(axis=0)) for i in range(len(FRAMES))]
    gd = [np.abs(outs[i][:, GRIP_IDX] - ref[i][:, GRIP_IDX]).mean()
          for i in range(len(FRAMES))]
    tcs = []
    for i, t in enumerate(FRAMES):
        m10 = outs[i][:, ARM_IDX].mean(axis=0)
        ds10 = np.stack([actions[row][ARM_IDX] - states[row][ARM_IDX]
                         for row in sel[t:t + 10]]).mean(axis=0)
        tcs.append(cos(m10, ds10))
    print(f"{name:<10} {np.mean(ca):>8.4f} {np.mean(cg):>9.4f} "
          f"{np.mean(gd):>10.2f} {np.mean(tcs):>7.3f}")

print("""
判读:
  int8   应复现崩档（cos_arm≪1，tf-cos≪0.9）——对照组不崩则本次全部作废。
  int8_sim ≈ int8 → monkeypatch 契约与 kernel 量化一致，仿真可信。
  rot8   判决档：tf-cos ≥0.90 且夹爪 MAE 可接受 → 机制成立，进 Phase 2
         （fht_int4.cu 改 FHT+int8 融合 kernel）；仍崩 → Route B′ 出局。
  ⚠ sim 档延迟含 D2H 同步 + CPU FWHT，仅供质量判决，不代表 Phase 2 延迟。""")
