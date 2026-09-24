#!/usr/bin/env python
"""INT8 量化专项排查（2026-09-24）——独立文件，不改项目任何代码。

背景：09-21 ab_compare（base 权重 + 随机噪声图）int8 vs bf16 cos≈0.99999；
09-23 tf_matrix（微调权重 + 真实训练帧）int8 / enc_only 相对数据集 cos 崩到
0.15 / 0.27，bf16 0.98。两次测试【权重、输入分布、参照系】三个变量同时变了，
本工具把它们拆开归因：

  audit  — CPU 权重审计（无需 GPU）：忠实复刻 FlashRT 编码器/解码器
           per-output-channel INT8 权重量化公式（含 RMSNorm fold、bf16 圆整），
           逐张量对比 base vs 微调权重的量化相对误差与行离群度(absmax/rms)。
           回答：微调权重本身是否更难量化。
  ab     — GPU 端到端：同一权重、同一输入下 bf16 vs int8【直接】对比
           （补上 09-21 测试缺失的单元）， crossed 设计：
             {真实训练帧, 随机噪声帧} × {bf16, int8(, enc8)}
           另带输入敏感性探针（灰图/括号 prompt/腕交换/state 错配），
           检验各档位下模型还读不读输入。noise 用 infer(noise=) 显式固定
           （predict 同种子对不齐，见 ab_real_camera.py 教训）。

用法:
  ~/holy/run.sh ~/holy/scripts/eval/int8_quant_probe.py audit
  ~/holy/run.sh ~/holy/scripts/eval/int8_quant_probe.py ab [--ep 0] [--enc8] [--base]
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
sub = ap.add_subparsers(dest="cmd", required=True)

a_audit = sub.add_parser("audit", help="CPU 权重 INT8 量化误差审计（base vs ft）")

a_ab = sub.add_parser("ab", help="GPU 端到端 bf16 vs int8 直接 A/B + 敏感性探针")
a_ab.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_ft")
a_ab.add_argument("--ep", type=int, default=0)
a_ab.add_argument("--frames", default="0,50,150,300,450")
a_ab.add_argument("--enc8", action="store_true",
                  help="附加 enc_only 档（编码器 INT8 + 解码器 bf16）")
a_ab.add_argument("--base", action="store_true",
                  help="附加 base 模型对照（pi05_lerobot_base，7 维 libero 口径）")
a_ab.add_argument("--out", default="/tmp/int8_probe_acts.npz")
args = ap.parse_args()

# ════════════════════════════════════════════════════════════════════
# audit：CPU 权重审计
# ════════════════════════════════════════════════════════════════════
def cmd_audit(_args):
    import torch
    from safetensors import safe_open

    BF16 = torch.bfloat16
    CKPTS = [
        ("base", "/home/galbot/holy/models/pi05_lerobot_base/model.safetensors"),
        ("ft  ", "/home/galbot/holy/models/pi05_g1_ft/model.safetensors"),
    ]
    ENC_L = DEC_L = 18

    def quant_stats(w_engine):
        """w_engine: 引擎布局 [K, N]。复刻 pi05_rtx.py quant():转置→[N,K]→
        per-输出通道 scale=absmax/127→round。返回 (相对Frobenius误差, 行峰值张量)。"""
        x = w_engine.t().contiguous()                       # [N, K]
        scale = (x.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        q = torch.round(x / scale[:, None]).clamp_(-127, 127)
        err = x - q * scale[:, None]
        rel = (err.norm() / x.norm().clamp_min(1e-30)).item()
        peak = x.abs().amax(dim=1) / x.pow(2).mean(dim=1).sqrt().clamp_min(1e-30)
        return rel, peak

    results = {}      # (ckpt, comp) -> list[(rel, peak)]
    weight_rms = {}   # ckpt -> list[标量] 权重整体幅度
    for tag, path in CKPTS:
        t0 = time.time()
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            pfx = "model." if "model.action_out_proj.weight" in keys else ""
            vp = f"{pfx}paligemma_with_expert.paligemma.model"
            ep, dp = f"{vp}.language_model.layers", f"{pfx}paligemma_with_expert.gemma_expert.model.layers"

            def g(key):
                return f.get_tensor(key).to(BF16).float()

            def add(comp, w):
                rel, peak = quant_stats(w)
                results.setdefault((tag, comp), []).append((rel, peak))
                weight_rms.setdefault(tag, []).append(w.pow(2).mean().sqrt().item())

            for i in range(ENC_L):
                P = f"{ep}.{i}"
                # qkv：fp32 RMSNorm fold（跳过头交织——纯行重排，不改统计）
                fuse_attn = 1.0 + f.get_tensor(f"{P}.input_layernorm.weight").float()
                q = f.get_tensor(f"{P}.self_attn.q_proj.weight").float() * fuse_attn
                k = f.get_tensor(f"{P}.self_attn.k_proj.weight").float() * fuse_attn
                v = f.get_tensor(f"{P}.self_attn.v_proj.weight").float() * fuse_attn
                add("enc.qkv", torch.cat([q, k, v], 0).to(BF16).float())
                add("enc.o", g(f"{P}.self_attn.o_proj.weight").t())
                fuse_ffn = 1.0 + f.get_tensor(f"{P}.post_attention_layernorm.weight").float()
                add("enc.gate", (f.get_tensor(f"{P}.mlp.gate_proj.weight").float()
                                 * fuse_ffn).t().to(BF16).float())
                add("enc.up", (f.get_tensor(f"{P}.mlp.up_proj.weight").float()
                               * fuse_ffn).t().to(BF16).float())
                add("enc.down", g(f"{P}.mlp.down_proj.weight").t())

            for i in range(DEC_L):
                P = f"{dp}.{i}"
                q = g(f"{P}.self_attn.q_proj.weight")
                k = g(f"{P}.self_attn.k_proj.weight")
                v = g(f"{P}.self_attn.v_proj.weight")
                add("dec.qkv", torch.cat([q, k, v], 0))
                add("dec.o", g(f"{P}.self_attn.o_proj.weight").t())
                add("dec.gate", g(f"{P}.mlp.gate_proj.weight").t())
                add("dec.up", g(f"{P}.mlp.up_proj.weight").t())
                add("dec.down", g(f"{P}.mlp.down_proj.weight").t())

            add("action_out_proj", g(f"{pfx}action_out_proj.weight").t())
        print(f"[{tag}] 审计完成 {time.time() - t0:.0f}s", flush=True)

    comps = ["enc.qkv", "enc.o", "enc.gate", "enc.up", "enc.down",
             "dec.qkv", "dec.o", "dec.gate", "dec.up", "dec.down",
             "action_out_proj"]
    print(f"\n{'组件':<16} {'base relF%':>10} {'ft relF%':>10} "
          f"{'base peak p50/p95/max':>24} {'ft peak p50/p95/max':>24}")
    for comp in comps:
        rb = [r for r, _ in results[("base", comp)]]
        rf = [r for r, _ in results[("ft  ", comp)]]
        pb = torch.cat([p for _, p in results[("base", comp)]])
        pf = torch.cat([p for _, p in results[("ft  ", comp)]])
        qb = torch.quantile(pb, torch.tensor([0.5, 0.95]))
        qf = torch.quantile(pf, torch.tensor([0.5, 0.95]))
        print(f"{comp:<16} {100 * sum(rb) / len(rb):>10.3f} "
              f"{100 * sum(rf) / len(rf):>10.3f}   "
              f"{qb[0]:.1f}/{qb[1]:.1f}/{pb.max():.1f}"
              f"{'':<8}{qf[0]:.1f}/{qf[1]:.1f}/{pf.max():.1f}")
    for tag in ("base", "ft  "):
        import numpy as np
        print(f"[{tag}] 全 GEMM 权重 RMS mean = "
              f"{np.mean(weight_rms[tag]):.5f}（权重整体幅度对照）")

    print("""
判读:
  - relF% = INT8 权重量化相对误差（高斯行理论值≈0.09%；越大越难量化）
  - peak = 行内 absmax/rms（高斯行≈4；≫4 = 行内离群通道挤压同行其他权重）
  - 若 ft 各组件 relF/peak 与 base 同量级 → 权重不是「昨天才崩」的差异源，
    09-21 的高相似度是【输入盲区】（噪声图下输出不受调节支配），INT8 的
    编码器扰动一直都在（FlashRT 自测 encoder cosine 也只有 0.991）。
  - 若 ft 显著更差 → 微调权重离群度增长，属权重侧量化敏感。""")

# ════════════════════════════════════════════════════════════════════
# ab：GPU 端到端
# ════════════════════════════════════════════════════════════════════
def cmd_ab(args):
    for k in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
              "FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_INT8_ENCODER_STATIC",
              "FVK_PI05_RTX_INT8_VISION"):
        os.environ.pop(k, None)
    os.environ["PI05_NO_GRAPH"] = "1"
    os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")

    import numpy as np
    import cv2
    import torch

    sys.path.insert(0, "/home/galbot/holy")
    from cuda_warmup import cuda_warmup
    cuda_warmup()

    # ── 数据集（复用 tf_matrix 的 npz 缓存，只读） ──
    DS = pathlib.Path("/home/galbot/holy/datasets/pick_place_balence")
    CAM_KEY = {"image": "head_right", "wrist_image": "left_arm",
               "wrist_image_right": "right_arm"}
    ARM_DIMS = list(range(0, 7)) + list(range(8, 15))
    CACHE = pathlib.Path(f"/tmp/tf_matrix_ep{args.ep}.npz")
    if not CACHE.exists():
        subprocess.run(["/usr/bin/python3", "-c",
                        "import pathlib,numpy as np,pandas as pd,sys\n"
                        "ds=pathlib.Path(sys.argv[1]);out=sys.argv[2];ep=int(sys.argv[3])\n"
                        "data=pd.read_parquet(ds/'data/chunk-000/file-000.parquet')\n"
                        "meta=pd.read_parquet(ds/'meta/episodes/chunk-000/file-000.parquet')\n"
                        "row=meta[meta.episode_index==ep].iloc[0]\n"
                        "df=data[data.episode_index==ep].reset_index(drop=True)\n"
                        "np.savez(out,states=np.stack([np.asarray(s,np.float32) for s in df['observation.state']]),"
                        "actions=np.stack([np.asarray(a,np.float32) for a in df['action']]),"
                        "tasks=np.array([str(row['tasks'])]),"
                        "t0_head=float(row['videos/observation.images.head_right/from_timestamp']),"
                        "t0_left=float(row['videos/observation.images.left_arm/from_timestamp']),"
                        "t0_right=float(row['videos/observation.images.right_arm/from_timestamp']))",
                        str(DS), str(CACHE), str(args.ep)], check=True)
    z = np.load(CACHE, allow_pickle=False)
    states, actions = z["states"], z["actions"]
    T0 = {"head_right": float(z["t0_head"]), "left_arm": float(z["t0_left"]),
          "right_arm": float(z["t0_right"])}

    def frame_at(cam, t):
        p = DS / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_MSEC, (T0[cam] + t / 30.0) * 1000.0)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"取帧失败 ep{args.ep} t={t} {cam}")
        return np.ascontiguousarray(cv2.cvtColor(cv2.resize(img, (224, 224)),
                                                 cv2.COLOR_BGR2RGB))

    FRAMES = [min(int(x), len(states) - 11) for x in args.frames.split(",")]
    real_imgs = [{v: frame_at(c, t) for v, c in CAM_KEY.items()} for t in FRAMES]
    rs = np.random.RandomState(20260924)
    noise_imgs = [{v: rs.randint(0, 256, (224, 224, 3), np.uint8)
                   for v in CAM_KEY} for _ in FRAMES]
    real_states = [states[t].astype(np.float32) for t in FRAMES]

    print(f"ep{args.ep} 帧 {FRAMES} | {len(real_imgs)} 组真实帧 + "
          f"{len(noise_imgs)} 组噪声帧已就绪", flush=True)

    # ── flash_rt（无 graph monkeypatch，内存补丁不落盘） ──
    import flash_rt.frontends.torch.pi05_rtx as _fe
    _orig_init = _fe.Pi05TorchFrontendRtx.__init__

    @functools.wraps(_orig_init)
    def _no_graph_init(self, *a, **kw):
        kw["use_cuda_graph"] = False
        _orig_init(self, *a, **kw)

    _fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init
    import flash_rt
    from flash_rt.core.utils.actions import normalize_state

    TASK0 = "Left arm pick up A. Right arm pick up A."

    def fixed_noise(i):
        g = torch.Generator().manual_seed(20260924 + i)
        return torch.randn(10, 32, generator=g)

    # 单元定义：(名字, 图像列表, 状态列表, prompt, 参考帧号)
    cells = [("real", real_imgs, real_states, TASK0, FRAMES),
             ("noise", noise_imgs, real_states, TASK0, FRAMES)]

    store = {}
    tiers = [("bf16", {"FVK_PI05_RTX_FORCE_BF16": "1"},
              {"FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY"}),
             ("int8", {"FVK_PI05_RTX_FORCE_INT8": "1"},
              {"FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_INT8_ENCODER_ONLY"})]
    if args.enc8:
        tiers.append(("enc8", {"FVK_PI05_RTX_INT8_ENCODER_ONLY": "1"},
                      {"FVK_PI05_RTX_FORCE_BF16", "FVK_PI05_RTX_FORCE_INT8"}))

    ckpt = pathlib.Path(args.ckpt)
    for tier, env_on, env_off in tiers:
        for k in env_off:
            os.environ.pop(k, None)
        os.environ.update(env_on)
        print(f"\n───── [{tier}] 加载 {ckpt.name} ─────", flush=True)
        t0 = time.time()
        model = flash_rt.load_model(str(ckpt), config="pi05", num_views=3,
                                    cache_frames=1, action_dim=16)
        ns = model._pipe.norm_stats
        print(f"load {time.time() - t0:.0f}s", flush=True)
        # 建 pipeline + 预热（丢弃）
        model.predict(real_imgs[0], prompt=TASK0,
                      state=normalize_state(real_states[0], ns))
        model.predict(real_imgs[0], prompt=TASK0,
                      state=normalize_state(real_states[0], ns))

        acc = {}
        for name, imgs_list, sts, prompt, _ref in cells:
            outs, ts = [], []
            for i, imgs in enumerate(imgs_list):
                t1 = time.perf_counter()
                model.predict(imgs, prompt=prompt,
                              state=normalize_state(sts[i], ns))   # 设 prompt/state，丢弃
                r = model.infer(imgs, noise=fixed_noise(i))
                ts.append((time.perf_counter() - t1) * 1000)
                outs.append(np.asarray(r["actions"], np.float32))
            acc[name] = np.stack(outs)
            print(f"[{tier}] {name}: p50 {np.median(ts):.0f} ms", flush=True)

        # 确定性自检（同噪声双跑应严格 0）
        model.predict(real_imgs[0], prompt=TASK0,
                      state=normalize_state(real_states[0], ns))
        d = np.abs(np.asarray(model.infer(real_imgs[0],
                                          noise=fixed_noise(0))["actions"],
                              np.float32) - acc["real"][0]).max()
        print(f"[{tier}] 确定性自检 max_diff = {d:.2e} "
              f"({'OK' if d == 0 else '⚠ 非零，对比失效'})", flush=True)

        # 敏感性探针（帧 0；读输入能力检验）
        st0n = normalize_state(real_states[0], ns)
        probes = [
            ("gray", {v: np.full((224, 224, 3), 128, np.uint8) for v in CAM_KEY},
             real_states[0], TASK0),
            ("bracket", real_imgs[0], real_states[0], f"['{TASK0}']"),
            ("wrist_swap", {"image": real_imgs[0]["image"],
                            "wrist_image": real_imgs[0]["wrist_image_right"],
                            "wrist_image_right": real_imgs[0]["wrist_image"]},
             real_states[0], TASK0),
            ("state+300", real_imgs[0],
             states[min(FRAMES[0] + 300, len(states) - 11)].astype(np.float32), TASK0),
        ]
        for pname, imgs, st, pr in probes:
            model.predict(imgs, prompt=pr, state=normalize_state(st, ns))
            acc[f"probe_{pname}"] = np.asarray(
                model.infer(imgs, noise=fixed_noise(0))["actions"], np.float32)[None]

        store[tier] = acc
        np.savez(args.out, **{f"{t}_{k}": v for t, acc2 in store.items()
                              for k, v in acc2.items()})
        del model
        torch.cuda.empty_cache()

    # ── 汇总 ──
    def cos(a, b):
        a, b = a.ravel(), b.ravel()
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / n) if n > 0 else float("nan")

    ref = store["bf16"]
    print("\n===== ① 直接 A/B：int8 相对 bf16（与 09-21 的 0.99999 同口径） =====")
    print(f"{'单元':<12} {'cos':>8} {'max|d|':>8} {'|bf16|范数':>10}")
    for cell in [c[0] for c in cells]:
        a, b = ref[cell], store["int8"][cell]
        print(f"{cell:<12} {cos(a, b):>8.4f} "
              f"{np.abs(a - b).max():>8.4f} {np.linalg.norm(a[0]):>10.3f}")

    print("\n===== ② teacher-forced 口径：各档 vs 数据集 delta（块均） =====")
    print(f"{'单元':<12} " + " ".join(f"{t:>10}" for t in store))
    for cell in [c[0] for c in cells]:
        row = f"{cell:<12} "
        for tier, acc in store.items():
            cs = []
            for i in range(acc[cell].shape[0]):
                t = FRAMES[i]
                m10 = acc[cell][i][:10][:, ARM_DIMS].mean(axis=0)
                ds10 = np.stack([actions[tt][ARM_DIMS] - states[tt][ARM_DIMS]
                                 for tt in range(t, min(t + 10, len(states)))
                                 ]).mean(axis=0)
                cs.append(cos(m10, ds10))
            row += f"{np.mean(cs):>10.3f} "
        print(row)

    print("\n===== ③ 敏感性：探针输出 vs 同档 real 帧0 输出（读输入能力） =====")
    print(f"{'探针':<12} " + " ".join(f"{t:>18}" for t in store))
    for pname in ["gray", "bracket", "wrist_swap", "state+300"]:
        row = f"{pname:<12} "
        for tier, acc in store.items():
            p0, r0 = acc[f"probe_{pname}"][0], acc["real"][0]
            dev = np.linalg.norm(p0 - r0)
            row += f"{cos(p0, r0):>10.3f} " f"L2={dev:>6.3f} "
        print(row)

    print("\n===== ④ 跨帧自相似：档内 5 个不同输入帧两两 cos（越低=越读输入） =====")
    print(f"{'单元':<8} " + " ".join(f"{t:>8}" for t in store))
    for cell in [c[0] for c in cells]:
        row = f"{cell:<8} "
        for tier, acc in store.items():
            A = acc[cell]
            cs = [cos(A[i], A[j]) for i in range(len(A))
                  for j in range(i + 1, len(A))]
            row += f"{np.mean(cs):>8.3f} "
        print(row)
    print("""
判读:
  ① 若 int8 在 real 单元崩、noise 单元仍高 → 09-21 的高相似度 = 噪声图输入盲区
    （输出由初始噪声先验主导，编码器损坏不可见），与权重无关的测试方法缺陷。
  ① 若 int8 连 noise 单元也崩 → 权重/档位本身在微调权重上就不可用（结合 audit 归因）。
  ③ bf16 探针应显著偏离 real（cos 低、L2 大）＝模型在读输入；
    int8 若全部≈real（cos→1、L2→0）＝编码器条件通路被量化抹平，
    输出只随噪声走——「看起来稳定」其实是瞎的。""")

if args.cmd == "audit":
    cmd_audit(args)
else:
    cmd_ab(args)
