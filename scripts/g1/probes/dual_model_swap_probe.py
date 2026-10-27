#!/usr/bin/env python
"""FlashRT 同进程双模型共存探针（pick→place 单进程方案前置实验，零运动零 SDK）。

背景：编排器把 place 放进 pick 同一进程（折叠后原地换 prompt/stats/包络进
place 轮）的前提，是 FlashRT 允许同进程两个构造共存且交替推理不出错/不串味
——此前从未验证过（跨构造位等不可比是既有结论，但"同构造内位等"应当成立）。

验证四件事：
  ① 共存载入：onlypick + place 两个 bf16 构造同进程各 load 一次（显存账照影）
  ② 交替位等：载入 place 之后再跑 pick、再跑 place——每个模型与自己
    首轮记录逐位一致（同构造同输入必须 bit-equal；不等=graph/缓存互踩）
  ③ 延迟量级：交替后的 p50 与单模型在役基线同档（graph-on ~230ms）
  ④ 载入耗时：第二构造的 autotune 载入是否显著慢于首个

判定全绿 → run_g1_loop.py 二相改造（fold 后原地切 place 轮）开写；
任一红 → 记录现象，回退方案 A（早 spawn+握手）。

用法（不需要机器人，不碰 SDK；跑之前确认没有别的推理进程占 GPU）：
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/probes/dual_model_swap_probe.py
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, "/home/galbot/holy/FlashRT")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pick-ckpt", default="/home/galbot/holy/models/pi05_g1_onlypick_deploy")
ap.add_argument("--place-ckpt", default="/home/galbot/holy/models/pi05_g1_place_deploy")
ap.add_argument("--repeats", type=int, default=3, help="每模型记录/复验次数")
ap.add_argument("--views", type=int, default=3)
args = ap.parse_args()

PICK_PROMPT = "Left arm pick up A. Right arm pick up A."
PLACE_PROMPT = ("Left arm places A in the top-left corner. "
                "Right arm places A in the top-left corner.")


def free_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def action_dim_of(ckpt):
    mf_p = pathlib.Path(ckpt) / "flashrt_deploy.json"
    return int(json.loads(mf_p.read_text()).get("action_dim", 16)) if mf_p.exists() else 16


CAM_KEYS = ("image", "wrist_image", "wrist_image_right")   # 管线契约键名


def fixed_inputs(dim):
    """确定性输入：视角固定噪声图（键名跟管线契约）+ 固定 23 维归一化 state。"""
    import numpy as np
    rng = np.random.default_rng(0)
    obs = {k: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
           for k in CAM_KEYS[:args.views]}
    state = np.linspace(-0.5, 0.5, dim, dtype=np.float32)
    return obs, state


def load(ckpt):
    import flash_rt
    t0 = time.time()
    m = flash_rt.load_model(str(ckpt), config="pi05", num_views=args.views,
                            cache_frames=1, action_dim=action_dim_of(ckpt))
    return m, time.time() - t0


def run(m, obs, state, prompt, tag):
    import numpy as np
    t0 = time.time()
    out = np.asarray(m.predict(obs, prompt=prompt, state=state))
    return out, (time.time() - t0) * 1000.0


def main():
    import numpy as np
    print(f"MemAvailable 起始 {free_mb():.0f} MB", flush=True)
    rec = {}   # tag -> [out ×repeats]
    lat = {}   # tag -> [ms ×(repeats+1)]

    for name, ckpt, prompt in (("pick", args.pick_ckpt, PICK_PROMPT),
                               ("place", args.place_ckpt, PLACE_PROMPT)):
        m, t_load = load(ckpt)
        obs, state = fixed_inputs(action_dim_of(ckpt))
        _, ms = run(m, obs, state, prompt, name)          # 预热（引擎构建）
        lat.setdefault(name, []).append(ms)
        rec[name] = []
        for _ in range(args.repeats):
            out, ms = run(m, obs, state, prompt, name)
            rec[name].append(out)
            lat[name].append(ms)
        print(f"[{name}] load {t_load:.1f}s | 首推(建引擎) {lat[name][0]:.0f}ms"
              f" | 稳态 {[f'{x:.0f}' for x in lat[name][1:]]} ms"
              f" | MemAvailable {free_mb():.0f} MB", flush=True)
        if name == "pick":
            globals()["_m_pick"] = m
        else:
            globals()["_m_place"] = m

    # ── 交替复验：place 在场后重跑 pick、pick 在场后重跑 place，逐位比对 ──
    print("\n== 交替位等复验（同构造同输入必须 bit-equal）==", flush=True)
    bad = 0
    for name, prompt in (("pick", PICK_PROMPT), ("place", PLACE_PROMPT)):
        m = globals()[f"_m_{name}"]
        obs, state = fixed_inputs(action_dim_of(
            args.pick_ckpt if name == "pick" else args.place_ckpt))
        for r in range(args.repeats):
            out, ms = run(m, obs, state, prompt, name)
            lat[name].append(ms)
            ref = rec[name][r]
            if out.shape != ref.shape or not np.array_equal(out, ref):
                d = float(np.max(np.abs(out - ref))) if out.shape == ref.shape else -1
                print(f"⛔ [{name}] 第 {r} 次交替输出与首轮不一致 "
                      f"(shape {out.shape} vs {ref.shape}, max|diff| {d:.3e})",
                      flush=True)
                bad += 1
            else:
                print(f"✅ [{name}] 第 {r} 次交替 bit-equal（{ms:.0f}ms）", flush=True)

    print("\n== 延迟汇总（含交替后；首推建引擎已剔除口径见上）==", flush=True)
    for name in ("pick", "place"):
        ss = sorted(lat[name][1:])
        print(f"[{name}] 稳态 p50 {ss[len(ss)//2]:.0f} ms | max {ss[-1]:.0f} ms"
              f" | {len(ss)} 次", flush=True)
    print(f"MemAvailable 终态 {free_mb():.0f} MB", flush=True)

    if bad:
        print(f"\n⛔ 结论：双构造共存互踩（{bad} 处不一致）——同进程方案否决，"
              f"回退方案 A（早 spawn+握手）", flush=True)
        sys.stdout.flush()
        import os
        os._exit(1)
    print("\n✅ 结论：双构造共存+交替推理全绿——loop 二相改造可以开写", flush=True)
    sys.stdout.flush()
    import os
    os._exit(0)


main()
