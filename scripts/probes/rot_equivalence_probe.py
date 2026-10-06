#!/usr/bin/env python
"""旋转等价性 + 逐层 K/V 判读探针（Route B′ 复核，2026-10-06）——零 FlashRT 改动。

背景：quarot_sim_ablation 判 rot8（W8A8+Hadamard）tf-cos 0.264 后，外部复核
提出质疑：0.2 量级不像量化噪声，像实现 bug 或某层被毁。逐层 cos 正常应 0.99+。
本探针补上当时缺的三块判据：

  ① rot_bf16（只旋转、不量化）：编码器权重沿 K 旋转 + bf16 GEMM 输入同步旋转，
     输出应与 bf16 几乎一致（逐层 KV cos ≈ 1）。不过 → 旋转实现有 bug。
  ② 逐层 K/V cos：解码器实际消费的是 18 层 encoder KV cache（层 17 early-return
     前 qkv_split_rope 已写满）。逐层对照各档 vs bf16，定位第一层崩点。
     对照锚点：pi05_rtx.py:800 在码注释 int8 令 encoder cos 0.991→0.282，
     int8 档应复现 ~0.28 —— 对不上则采集管线自身有错。
  ③ 隐层 vs 动作拆分：同一档并报 KV cos（解码器入口）与动作端到端 cos，
     直接分辨「编码器被毁」还是「残差被流匹配头放大」。

四档：bf16（参照）/ rot_bf16（等价性门）/ int8（对照组，锚 0.282）/ rot8（判决档）。
rot_bf16 权重旋转在 load 后、首 forward 前就地对 pipe.weights 生效（bf16 档无
INT8 副本）；激活侧 patch gemm.bf16_nn，按权重张量 id 白名单只旋转 90 个编码器
GEMM 的输入（qkv/gate/up 的 GEMM 输入=rms_norm 出来后未旋转的 x_norm，o/down
输入=attn 出/hidden，均在 GEMM 入口统一旋转，数学恒等 (xH)(HW)ᵀ=xWᵀ）。

用法:
  ~/holy/run.sh ~/holy/scripts/probes/rot_equivalence_probe.py \
      [--configs bf16,rot_bf16,int8,rot8] [--ep 0] [--frames 0,60,120,180,240,300]
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
ap.add_argument("--configs", default="bf16,rot_bf16,int8,rot8")
ap.add_argument("--npz", default="/tmp/ablation_only_place_ep0.npz")
ap.add_argument("--out", default="/tmp/rot_equivalence_probe.npz")
args = ap.parse_args()

os.environ["PI05_NO_GRAPH"] = "1"
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
          "FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_INT8_ENC_SKIP",
          "FVK_PI05_RTX_INT8_VISION"):
    os.environ.pop(k, None)

# ── npz 自举（与 quarot_sim_ablation 同源）──
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
from flash_rt.models.pi05 import pipeline_rtx as _PR  # noqa: E402
from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline  # noqa: E402
from flash_rt.frontends.torch.pi05_rtx import (  # noqa: E402
    Pi05TorchFrontendRtx as _FE)

ENC_NKV, ENC_HD = _PR.ENC_NKV, _PR.ENC_HD

def fixed_noise(i: int):
    g = torch.Generator().manual_seed(20260930 + i)
    return torch.randn(10, 32, generator=g)

# ── FWHT（未归一化）+ 自检 ──
def fwht(x: torch.Tensor) -> torch.Tensor:
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

_ENC_FAMS = ("encoder_attn_qkv_w", "encoder_attn_o_w",
             "encoder_ffn_gate_w", "encoder_ffn_up_w", "encoder_ffn_down_w")

def rotate_pipe_weights(pipe) -> int:
    """管线 bf16 权重存的是设备指针（int，首帧 predict 时已上传）——
    就地 D2H → 沿 K FWHT/√K → H2D 写回原指针。"""
    SH = {
        "encoder_attn_qkv_w": (_PR.ENC_D, (_PR.ENC_NH + 2 * _PR.ENC_NKV) * _PR.ENC_HD),
        "encoder_attn_o_w":   (_PR.ENC_D, _PR.ENC_D),
        "encoder_ffn_gate_w": (_PR.ENC_D, _PR.ENC_H),
        "encoder_ffn_up_w":   (_PR.ENC_D, _PR.ENC_H),
        "encoder_ffn_down_w": (_PR.ENC_H, _PR.ENC_D),
    }
    n_rot = 0
    for fam, (K, N) in SH.items():
        lst = pipe.weights[fam]
        assert len(lst) == 18 and all(isinstance(p, int) for p in lst), \
            f"{fam} 布局异常：{type(lst)} len={len(lst)}"
        for i, ptr in enumerate(lst):
            host = np.empty(K * N, dtype=np.uint16)
            _check(_rt.cudaMemcpy(host.ctypes.data, ctypes.c_void_p(ptr),
                                  K * N * 2, 2), f"w D2H {fam}[{i}]")
            w = torch.from_numpy(
                (host.astype(np.uint32) << 16).view(np.float32)).view(K, N).float()
            wr = fwht(w.t().contiguous()).t().contiguous() / math.sqrt(K)
            e0 = float(w.pow(2).sum())
            e1 = float(wr.pow(2).sum())
            assert abs(e1 - e0) / e0 < 1e-3, f"{fam}[{i}] 能量漂移 {e0}->{e1}"
            raw = np.ascontiguousarray(
                wr.to(torch.bfloat16).contiguous()
                .view(torch.uint16).numpy().reshape(-1))
            _check(_rt.cudaMemcpy(ctypes.c_void_p(ptr), raw.ctypes.data,
                                  K * N * 2, 1), f"w H2D {fam}[{i}]")
            n_rot += 1
    print(f"  权重旋转完成：{n_rot} 个设备张量（应=90）", flush=True)
    assert n_rot == 90
    return n_rot

# ── bf16 GEMM 输入旋转（rot_bf16 档）：按权重设备指针白名单 ──
_ROT_W_IDS = set()
_SCRATCH = {"t": None, "n": 0}

def _rot_bf16_in(a_ptr: int, M: int, K: int) -> int:
    """device bf16 (M,K) → FWHT/√K → bf16，写 scratch，返回新指针。"""
    torch.cuda.synchronize()
    host = np.empty(M * K, dtype=np.uint16)
    _check(_rt.cudaMemcpy(host.ctypes.data, ctypes.c_void_p(a_ptr),
                          M * K * 2, 2), "rot D2H")
    x = torch.from_numpy(
        (host.astype(np.uint32) << 16).view(np.float32)).view(M, K).float()
    x = (fwht(x) / math.sqrt(K)).to(torch.bfloat16)
    if _SCRATCH["n"] < M * K:
        _SCRATCH["t"] = torch.empty(M * K, dtype=torch.bfloat16, device="cuda")
        _SCRATCH["n"] = M * K
    buf = _SCRATCH["t"]
    buf[:M * K].copy_(x.reshape(-1))
    torch.cuda.synchronize()
    return buf.data_ptr()

def patch_bf16_rot(pipe) -> None:
    g = pipe.gemm
    orig = g.bf16_nn
    for fam in _ENC_FAMS:
        for p in pipe.weights[fam]:
            _ROT_W_IDS.add(p)          # 权重设备指针值
    assert len(_ROT_W_IDS) == 90

    def _rot_nn(a_ptr, weight, out_ptr, M, N, K, *a, stream=0, **kw):
        if weight in _ROT_W_IDS:
            a_ptr = _rot_bf16_in(a_ptr, M, K)
        return orig(a_ptr, weight, out_ptr, M, N, K, *a, stream=stream, **kw)

    try:
        g.bf16_nn = _rot_nn
        patch_bf16_rot._restore = (g, orig, False)
    except (AttributeError, TypeError):
        type(g).bf16_nn = staticmethod(_rot_nn)
        patch_bf16_rot._restore = (type(g), orig, True)
    print("  bf16_nn 输入旋转 patch 已挂（90 权重白名单）", flush=True)

def unpatch_bf16_rot() -> None:
    obj, orig, is_cls = patch_bf16_rot._restore
    obj.bf16_nn = orig
    _ROT_W_IDS.clear()

# ── rot8 档复用 quarot_sim_ablation 的三件套 ──
_ORIG_QENC = _FE._quantize_encoder_int8

_FAM_OF = {
    "qkv": ["encoder_attn_qkv_w"],
    "gu": ["encoder_ffn_gate_w", "encoder_ffn_up_w"],
    "o": ["encoder_attn_o_w"],
    "down": ["encoder_ffn_down_w"],
}

def _chan_fold_qenc(tags):
    """chan 档 ctor 钩子：把静态通道 scale 折进权重（SmoothQuant 式），
    再照常走 per-output-channel INT8 量化——GEMM 契约不变。"""
    def _fold(self):
        for tag in tags:
            for fam in _FAM_OF[tag]:
                lst = self._ckpt_bf16[fam]
                for i, w in enumerate(lst):
                    sa = _CALIB.get((tag, i))
                    if sa is None:
                        continue   # 层17 的 o/gu/down 在推理中不跑（early-return）
                    sa = sa.to(w.device)
                    assert sa.numel() == w.shape[0], f"{fam}[{i}] scale 形状"
                    lst[i] = (w.float() * sa[:, None]).to(w.dtype).contiguous()
        print(f"  [chan] 通道 scale 折算完成：{sum(len(_FAM_OF[t]) for t in tags) * 18} 个权重", flush=True)
        return _ORIG_QENC(self)
    return _fold

def _rot_qenc(self):
    W = self._ckpt_bf16
    n_rot = 0
    for key in list(W.keys()):
        if not (key.startswith("encoder_") and key.endswith("_w")):
            continue
        lst = W[key]
        for i, w in enumerate(lst):
            if w.dim() != 2:
                continue
            K = w.shape[0]
            wr = fwht(w.float().t().contiguous()).t().contiguous() / math.sqrt(K)
            lst[i] = wr.to(w.dtype).contiguous()
            n_rot += 1
    print(f"  [rot8] 权重旋转（ctor 内）{n_rot} 个张量", flush=True)
    return _ORIG_QENC(self)

_ORIG_ENC_I8 = Pi05Pipeline._enc_int8_gemm
_SIM_ROT = False
_SIM_CALLS = 0
_SIM_LOG = None
_FKV = None
_ORIG_FUSED = {}

# ── 分布普查模式（prof 档）：在每个量化位点对真激活算四种 scale 方案的源头误差 ──
_PROF = None          # {(site,layer): [err_tok_amax, err_chan_amax, err_chan_holdout, err_tok_p99]}
_PROF_PHASE = "idle"  # predict 期不采
_PROF_LAYER = 0
_QKV_N = 0
_PROF_MODE = "err"    # "err" | "calib"（存每通道 amax 而非算误差）
_CALIB = None         # {(site,layer): per-channel amax (K,) fp32 tensor}
_CHAN_SA = None       # chan8 档加载的静态通道 scale
_CHAN_TAGS = ()       # 哪些位点走通道静态量化

def _prof_record(tag, layer, x):
    """x (M,K) fp32 torch——err 模式算四方案误差；calib 模式累积通道 amax。"""
    if _PROF_MODE == "calib":
        v = x.abs().amax(dim=0)
        key = (tag, layer)
        _CALIB[key] = (torch.maximum(_CALIB[key], v) if key in _CALIB
                       else v.clone())
        return
    M, K = x.shape
    ra = x.abs().amax(dim=1)                    # per-token amax
    ca = x.abs().amax(dim=0)                    # per-channel amax（oracle）
    ra9 = torch.quantile(x.abs(), 0.99, dim=1)  # per-token p99 截断
    # held-out per-channel：前半 token 定 scale，后半验误差（静态方案的泛化风险）
    h = max(M // 2, 1)
    cah = x[:h].abs().amax(dim=0)

    def rel(xx, s):
        s = torch.clamp(s, min=1e-12)
        d = torch.clamp(torch.round(xx / s), -127, 127) * s
        return float((xx - d).norm() / xx.norm())

    _PROF[(tag, layer)] = [rel(x, ra[:, None] / 127),
                           rel(x, ca[None, :] / 127),
                           rel(x[h:], cah[None, :] / 127),
                           rel(x, ra9[:, None] / 127)]

def _sim_quant_from_dev(src_ptr, i8_ptr, scale_ptr, M, K, tag=None):
    torch.cuda.synchronize()
    host = np.empty(M * K, dtype=np.uint16)
    _check(_rt.cudaMemcpy(host.ctypes.data, ctypes.c_void_p(src_ptr),
                          M * K * 2, 2), "sim D2H")
    x = torch.from_numpy(
        (host.astype(np.uint32) << 16).view(np.float32)).view(M, K).float()
    if _PROF_PHASE == "infer":
        layer = _PROF_LAYER
        if tag == "qkv":
            layer = _QKV_N - 1          # qkv 计数已含本次
        if _PROF is not None or _PROF_MODE == "calib":
            _prof_record(tag, layer, x)
        if _CHAN_SA is not None and tag in _CHAN_TAGS:
            # 通道静态：scale 折进权重，激活按定标量化，输出 scale 恒 1
            sa = _CHAN_SA[(tag, layer)]
            assert sa.numel() == K
            q = torch.clamp(torch.round(x / sa[None, :]), -127,
                            127).to(torch.int8)
            torch.cuda.synchronize()
            _check(_rt.cudaMemcpy(ctypes.c_void_p(i8_ptr),
                                  q.contiguous().numpy().reshape(-1).ctypes.data,
                                  M * K, 1), "chan H2D int8")
            ones = np.ones(M, dtype=np.float32)
            _check(_rt.cudaMemcpy(ctypes.c_void_p(scale_ptr), ones.ctypes.data,
                                  M * 4, 1), "chan H2D scale")
            return
    if _SIM_ROT:
        x = fwht(x) / math.sqrt(K)
    x = x.cuda()
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
    _sim_quant_from_dev(act_ptr, act_i8_ptr, scale_buf.ptr.value, M, K,
                        tag="down" if "down" in weight_name else "o")
    self._int8_gemm_fused(act_i8_ptr, weight_name, out_ptr, M, N, K,
                          scale_buf.ptr.value, stream)

def _sim_qkv(x_ptr, rms_w, i8p, sp, seq, K, eps, stream=0):
    global _SIM_CALLS, _QKV_N, _PROF_LAYER
    if torch.cuda.is_current_stream_capturing():
        return _ORIG_FUSED["rms_norm_int8_rowwise"](
            x_ptr, rms_w, i8p, sp, seq, K, eps, stream=stream)
    _SIM_CALLS += 1
    if _PROF_PHASE == "infer":
        _QKV_N += 1
        _PROF_LAYER = _QKV_N - 1
    if _SIM_LOG is not None:
        _SIM_LOG.append(("qkv", seq, K))
    tmp = torch.empty(seq, K, dtype=torch.bfloat16, device="cuda")
    _FKV.rms_norm(x_ptr, rms_w, tmp.data_ptr(), seq, K, eps, stream=stream)
    _sim_quant_from_dev(tmp.data_ptr(), i8p, sp, seq, K, tag="qkv")

def _sim_gu(x_ptr, xnorm_ptr, rms_w, i8p, sp, seq, K, eps, stream=0):
    global _SIM_CALLS
    if torch.cuda.is_current_stream_capturing():
        return _ORIG_FUSED["residual_add_rms_norm_int8_rowwise"](
            x_ptr, xnorm_ptr, rms_w, i8p, sp, seq, K, eps, stream=stream)
    _SIM_CALLS += 1
    if _SIM_LOG is not None:
        _SIM_LOG.append(("gu", seq, K))
    _FKV.residual_add(x_ptr, xnorm_ptr, seq * K, stream=stream)
    _FKV.rms_norm(x_ptr, rms_w, xnorm_ptr, seq, K, eps, stream=stream)
    _sim_quant_from_dev(xnorm_ptr, i8p, sp, seq, K, tag="gu")

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

# ── 逐层 K/V 采集：wrap _encoder_layer，层逻辑跑完读 KV cache ──
_ORIG_ENC_LAYER = Pi05Pipeline._encoder_layer
_CAP = False          # 仅 infer 期为 True
_KV = None            # dict[layer] -> (k fp32 np, v fp32 np)

def _d2h_bf16(ptr: int, n: int) -> np.ndarray:
    host = np.empty(n, dtype=np.uint16)
    torch.cuda.synchronize()
    _check(_rt.cudaMemcpy(host.ctypes.data, ctypes.c_void_p(ptr), n * 2, 2),
           "kv D2H")
    return (host.astype(np.uint32) << 16).view(np.float32)

def _cap_layer(self, i, seq, fuse_b1, stream):
    _ORIG_ENC_LAYER(self, i, seq, fuse_b1, stream)
    if _CAP and _KV is not None:
        n = seq * ENC_NKV * ENC_HD
        kp, vp = self._enc_kv_layer_ptrs(i, 0)
        _KV[i] = (_d2h_bf16(kp, n), _d2h_bf16(vp, n))

Pi05Pipeline._encoder_layer = _cap_layer   # 全档安装：bf16 参照也要采

# ── 主循环 ──
CKPT = str(pathlib.Path(args.ckpt).expanduser())
store, lat, kv_all = {}, {}, {}
for name in args.configs.split(","):
    name = name.strip()
    for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
              "FVK_PI05_RTX_FORCE_BF16"):
        os.environ.pop(k, None)
    is_sim = name in ("int8_sim", "rot8")
    is_rot_bf16 = name == "rot_bf16"
    is_prof = name == "prof"
    is_calib = name == "calib"
    is_chan = name in ("chan8", "chan8all")
    if name == "bf16" or is_rot_bf16:
        os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
    else:
        os.environ["FVK_PI05_RTX_INT8_ENCODER_ONLY"] = "1"
    print(f"───── [{name}] ─────", flush=True)
    if name == "rot8":
        _FE._quantize_encoder_int8 = _rot_qenc
    if is_calib:
        _PROF_MODE, _CALIB = "calib", {}
    if is_chan:
        zc = np.load("/tmp/pi05_enc_chan_calib.npz")
        _CALIB = {}
        for k, v in zc.items():
            tag, li = k.rsplit("_", 1)
            _CALIB[(tag, int(li))] = torch.from_numpy(
                np.ascontiguousarray(v)).float()
        _CHAN_SA = _CALIB
        _CHAN_TAGS = (("qkv", "gu") if name == "chan8"
                      else ("qkv", "gu", "o", "down"))
        _FE._quantize_encoder_int8 = _chan_fold_qenc(_CHAN_TAGS)
    t0 = time.time()
    model = flash_rt.load_model(CKPT, config="pi05", num_views=3,
                                cache_frames=1, action_dim=16)
    if name == "rot8" or is_chan:
        _FE._quantize_encoder_int8 = _ORIG_QENC
    print(f"load {time.time() - t0:.0f}s", flush=True)
    model._pipe.use_cuda_graph = False
    ns = model._pipe.norm_stats

    if is_sim or is_prof or is_calib or is_chan:
        patch_sim(rot=False, pipe=model._pipe)
    if is_prof:
        _PROF, _PROF_LAYER, _QKV_N, _PROF_PHASE = {}, 0, 0, "idle"
    if is_rot_bf16:
        # 管线在首次 predict 时惰性构建（FORCE_BF16 下无 int8 副本，权重未动），
        # 先建出来再旋转+打补丁
        model.predict(IMGS[0], prompt=TASK,
                      state=normalize_state(RSTS[0], ns))
        PL = model._pipe.pipeline
        assert PL is not None, "首次 predict 后管线仍未建——结构变了，停"
        rotate_pipe_weights(PL)
        patch_bf16_rot(PL)

    outs, ts = [], []
    n_enc8 = -1
    for i, imgs in enumerate(IMGS):
        stn = normalize_state(RSTS[i], ns)
        c0 = _SIM_CALLS
        _PROF_PHASE = "calib"
        _QKV_N, _PROF_LAYER = 0, 0
        model.predict(imgs, prompt=TASK, state=stn)
        c1 = _SIM_CALLS
        if n_enc8 < 0 and name in ("int8", "rot8"):
            n_enc8 = sum(1 for k in model._pipe.pipeline.weights.get("int8", {})
                         if k.startswith("encoder_"))
            print(f"编码器 INT8 位点 {n_enc8}/90", flush=True)
            if name == "rot8":
                assert n_enc8 == 90, "rot8 档必须全位点 INT8"
        t1 = time.perf_counter()
        if is_sim and i == 0:
            _SIM_LOG = []
        _PROF_PHASE = "infer"
        _QKV_N, _PROF_LAYER = 0, 0
        _KV = {}
        _CAP = True
        r = model.infer(imgs, noise=fixed_noise(i))
        _CAP = False
        ts.append((time.perf_counter() - t1) * 1000)
        outs.append(np.asarray(r["actions"], np.float32))
        kv_all[(name, i)] = _KV
        _KV = None
        if is_sim and i == 0:
            from collections import Counter
            cnt = Counter(e[0] if isinstance(e[0], str) and e[0] in
                          ("qkv", "gu") else "o/down" for e in _SIM_LOG)
            print(f"  sim patch 调用：predict +{c1 - c0}，infer +"
                  f"{_SIM_CALLS - c1}（期望 69）", flush=True)
            assert _SIM_CALLS - c1 == 69, f"infer 期覆盖 {_SIM_CALLS - c1} ≠ 69"
    outs = np.stack(outs)
    store[name] = outs
    lat[name] = float(np.median(ts))
    np.savez(args.out, **{f"cfg_{k}": v for k, v in store.items()})

    if not is_rot_bf16:   # rot_bf16 权重仍处旋转态，复跑必不同，自检无意义
        model.predict(IMGS[0], prompt=TASK, state=normalize_state(RSTS[0], ns))
        d = np.abs(np.asarray(model.infer(IMGS[0],
                                          noise=fixed_noise(0))["actions"],
                              np.float32) - outs[0]).max()
        print(f"[{name}] p50 {np.median(ts):.0f} ms | 确定性 max_diff={d:.2e}"
              f"{' ⚠ 非零' if d != 0 else ''}", flush=True)

    if is_sim:
        unpatch_sim()
    if is_rot_bf16:
        unpatch_bf16_rot()

    if is_calib:
        np.savez("/tmp/pi05_enc_chan_calib.npz",
                 **{f"{t}_{l}": v.numpy()
                    for (t, l), v in sorted(_CALIB.items())})
        print(f"校准完成：{len(_CALIB)} 位点 → /tmp/pi05_enc_chan_calib.npz",
              flush=True)
    _PROF_MODE, _PROF, _CHAN_SA, _CHAN_TAGS = "err", None, None, ()

    if is_prof:
        # ── 普查汇总：四方案源头重建误差（越小越好），按位点聚合 ──
        print("\n===== 分布普查：源头量化重建误差（相对 L2，单帧 infer） =====")
        print(f"{'site':<6}{'layer':>6} {'tok/amax':>9} {'chan/oracle':>12}"
              f" {'chan/holdout':>13} {'tok/p99':>9}")
        agg = {}
        for (tag, layer), e in sorted(_PROF.items(),
                                      key=lambda kv: (kv[0][0], kv[0][1])):
            print(f"{tag:<6}{layer:>6} {e[0]:>9.4f} {e[1]:>12.4f}"
                  f" {e[2]:>13.4f} {e[3]:>9.4f}")
            agg.setdefault(tag, []).append(e)
        print("\n-- 按位点中位数 --")
        for tag, es in agg.items():
            m = np.median(np.array(es), axis=0)
            print(f"{tag:<6} med  {m[0]:>9.4f} {m[1]:>12.4f}"
                  f" {m[2]:>13.4f} {m[3]:>9.4f}")
        print("""
判读: tok/amax=现行方案。chan/holdout 显著更小 → 通道结构主导，per-channel
  静态 scale（折进权重）有戏；tok/p99 显著更小 → 行内重尾，截断量化有戏。
  四列都 ≈1% 量级而端到端仍崩 → 源头无错可挤，放大在网络。""")
    del model
    torch.cuda.empty_cache()

# ── 汇总 1：逐层 K/V cos（vs bf16） ──
def cos(a, b):
    a, b = a.ravel(), b.ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else float("nan")

names = [n.strip() for n in args.configs.split(",")]
print("\n===== 逐层 encoder K/V cos（vs bf16，18 层 GQA cache = 解码器实际输入） =====")
hdr = "layer |" + "|".join(f" {n:>10} K" for n in names if n != "bf16") + \
      " ||" + "|".join(f" {n:>10} V" for n in names if n != "bf16")
print(hdr)
others = [n for n in names if n != "bf16"]
for li in range(18):
    rowk, rowv = [], []
    for n in others:
        cks = [cos(kv_all[(n, f)][li][0], kv_all[("bf16", f)][li][0])
               for f in range(len(FRAMES)) if ("bf16", f) in kv_all and (n, f) in kv_all]
        ckv = [cos(kv_all[(n, f)][li][1], kv_all[("bf16", f)][li][1])
               for f in range(len(FRAMES)) if ("bf16", f) in kv_all and (n, f) in kv_all]
        rowk.append(f"{np.mean(cks):12.4f}" if cks else "          —")
        rowv.append(f"{np.mean(ckv):12.4f}" if ckv else "          —")
    print(f"{li:5d} |" + "|".join(rowk) + " ||" + "|".join(rowv))

# ── 汇总 2：动作端到端（口径同 quarot_sim_ablation） ──
ref = store.get("bf16")
print("\n===== 动作端到端（① vs bf16 输出 / ② tf-cos vs 数据集真值位移） =====")
print(f"{'config':<10} {'cos_arm':>8} {'cos_grip':>9} {'grip|d|pp':>10} {'tf-cos':>7}")
for name, o in store.items():
    if ref is None and name != "bf16":
        print(f"{name:<10} （本进程无 bf16 参照，略）")
        continue
    ca = [cos(o[i][:, ARM_IDX].mean(axis=0),
              ref[i][:, ARM_IDX].mean(axis=0)) for i in range(len(FRAMES))]
    cg = [cos(o[i][:, GRIP_IDX].mean(axis=0),
              ref[i][:, GRIP_IDX].mean(axis=0)) for i in range(len(FRAMES))]
    gd = [np.abs(o[i][:, GRIP_IDX] - ref[i][:, GRIP_IDX]).mean()
          for i in range(len(FRAMES))]
    tcs = []
    for i, t in enumerate(FRAMES):
        m10 = o[i][:, ARM_IDX].mean(axis=0)
        ds10 = np.stack([actions[row][ARM_IDX] - states[row][ARM_IDX]
                         for row in sel[t:t + 10]]).mean(axis=0)
        tcs.append(cos(m10, ds10))
    print(f"{name:<10} {np.mean(ca):>8.4f} {np.mean(cg):>9.4f} "
          f"{np.mean(gd):>10.2f} {np.mean(tcs):>7.3f}")

print("""
判读:
  rot_bf16 逐层 KV cos 应 ≈1.000（<0.999 → 旋转实现有 bug，先修再谈）。
  int8 逐层 KV cos 应复现 ~0.28 量级（pi05_rtx.py:800 在码锚点）；
  对不上 → 本探针采集管线自错，全部作废。
  rot8 逐层 KV cos：≈0.99 而动作仍崩 → 无 bug，残差被流匹配头放大；
  仍 ~0.3 → 旋转在真实 kernel 路径上没生效/不成立，回仿真逐位点查。""")

# ── per-token 判读：崩是否集中在极少数海量 token（attention sink）──
# KV cache 布局 (total_kv_max, NKV, HD)，NKV=1 时每 token 一个 HD 维向量。
if all((n, 0) in kv_all for n in ("bf16", "int8", "rot8")):
    print("\n===== per-token 判读（frame0）=====")
    for li in (1, 17):
        for ki, kind in ((0, "K"), (1, "V")):
            bf = kv_all[("bf16", 0)][li][ki]
            n_tok = bf.shape[0] // (ENC_NKV * ENC_HD)
            bft = bf.reshape(n_tok, -1)
            bn = np.linalg.norm(bft, axis=1)
            top = np.argsort(bn)[::-1]
            print(f"\n-- layer{li} {kind}：token{n_tok}，"
                  f"norm p50={np.median(bn):.2f} max={bn[top[0]]:.2f}"
                  f"（{bn[top[0]] / np.median(bn):.0f}×）")
            for name in ("int8", "rot8"):
                xt = kv_all[(name, 0)][li][ki].reshape(n_tok, -1)
                pt = np.array([cos(xt[t], bft[t]) for t in range(n_tok)])
                line = f"   {name:<5} 崩token数(cos<0.8)={int((pt < 0.8).sum())}"
                line += f" top5norm位置={list(top[:5])}"
                for k in (1, 5, 20):
                    keep = np.ones(n_tok, bool)
                    keep[top[:k]] = False
                    a, b = xt[keep].ravel(), bft[keep].ravel()
                    line += f" 去top{k}:cos={cos(a, b):.3f}"
                print(line, flush=True)
