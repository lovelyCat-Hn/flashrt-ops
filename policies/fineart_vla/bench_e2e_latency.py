#!/usr/bin/env python
"""fineart_vla e2e + 延迟三口径台（flash_pyrt311，零运动）。

三口径（镜像 pi05 表 1-1/5-3 方法论）：
  noload  空载：引擎独占进程，冻结合成帧，每轮 System 2+System 1 全链计时
  sdk     SDK 在场：galbot_sdk.s1 GalbotRobot 持相机订阅（调用形 =
          run_g1_loop_traj.py：init(sensors)+get_rgb_data，**无 start_controller
          =零运动**），模型输入仍冻结合成帧（与空载同输入，差值=SDK 栈成本）
  loop    循环：背靠背 60 轮，每轮换新合成帧+新噪声（稳态服务节拍）

调用面 = parity 已证 surface（golden_engine / golden_subtask_engine 同款）：
  load_model(config='fineart_vla') → set_prompt(task, state) → calibrate →
  generate_subtask(每轮) + infer(每轮)。

用法：
  ~/miniforge3/envs/flash_pyrt311/bin/python \\
      ~/holy/policies/fineart_vla/bench_e2e_latency.py --mode noload
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \\
      ~/miniforge3/envs/flash_pyrt311/bin/python \\
      ~/holy/policies/fineart_vla/bench_e2e_latency.py --mode sdk
落盘 ~/holy/evidence/fineart_e2e_latency/<mode>.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")

DEFAULT_CKPT = os.path.expanduser("~/holy/models/fineart_vla_midtrained")
DEFAULT_DIR = os.path.expanduser("~/holy/evidence/fineart_e2e_latency")
VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")
# SDK 相机映射（galbot_sdk.s1 SensorType，head+双腕 = fineart 3 视角）
SDK_CAMS = ("HEAD_RIGHT_CAMERA", "LEFT_ARM_CAMERA", "RIGHT_ARM_CAMERA")


def synth_frames(seed_base: int) -> list[np.ndarray]:
    return [np.random.default_rng(seed_base + i).integers(0, 256, (224, 224, 3),
                                                          dtype=np.uint8)
            for i in range(3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("noload", "sdk", "loop"), default="noload")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--task", default="pick up the object")
    ap.add_argument("--rounds", type=int, default=None,
                    help="默认：noload/sdk=20，loop=60")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    rounds = args.rounds or (60 if args.mode == "loop" else 20)

    # ── SDK 持栈（sdk 模式；调用形 = run_g1_loop_traj.py，零 start_controller） ──
    robot = None
    if args.mode == "sdk":
        from galbot_sdk.s1 import GalbotRobot, SensorType
        robot = GalbotRobot()
        sensors = {getattr(SensorType, c) for c in SDK_CAMS}
        if not robot.init(sensors):
            raise SystemExit("robot.init 失败（相机服务在跑吗？）——口径② 需要 SDK 栈在场")
        probe = robot.get_rgb_data(getattr(SensorType, SDK_CAMS[0]))
        print(f"[sdk] GalbotRobot init OK（{len(sensors)} 相机订阅在场，零运动）；"
              f"get_rgb_data 探针 {'OK' if probe is not None else '空'}")

    import torch  # noqa: E402
    from flash_rt.api import load_model

    # golden 同款状态：raw seed7 → q01/q99 归一化
    from golden_engine import build_inputs
    raw_state, norm_state, imgs0, _ = build_inputs(args.ckpt, 7, 123)
    obs0 = dict(zip(VIEW_KEYS, imgs0))

    print(f"[{args.mode}] load_model ← {args.ckpt}")
    t0 = time.time()
    model = load_model(args.ckpt, framework="torch", config="fineart_vla",
                       num_views=3, num_steps=10, action_dim=14)
    print(f"      {time.time() - t0:.1f}s")
    model.set_prompt(args.task, norm_state)
    model.pipeline.calibrate_with_real_data([obs0])

    rows = []
    rng = np.random.default_rng(2026)
    for i in range(rounds + args.warmup):
        imgs = synth_frames(8 + (i * 7) % 1000) if args.mode == "loop" else imgs0
        obs = dict(zip(VIEW_KEYS, imgs))
        noise = rng.standard_normal((50, 32)).astype(np.float32)
        t = time.perf_counter()
        text = model.pipeline.generate_subtask(args.task, obs)          # System 2
        t_s2 = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        out = model.infer(obs, noise=noise)                             # System 1
        t_s1 = (time.perf_counter() - t) * 1000
        acts = np.asarray(out["actions"] if isinstance(out, dict) else out)
        rec = {"round": i, "t_s2_ms": round(t_s2, 1), "t_s1_ms": round(t_s1, 1),
               "t_total_ms": round(t_s2 + t_s1, 1),
               "actions_shape": list(acts.shape),
               "finite": bool(np.isfinite(acts).all())}
        if i < args.warmup:
            print(f"  warm{i}: S2 {t_s2:.0f} / S1 {t_s1:.0f} ms（不计入）")
            continue
        rows.append(rec)
        print(f"  #{i - args.warmup + 1}: S2 {t_s2:.0f} / S1 {t_s1:.0f} / "
              f"Σ {t_s2 + t_s1:.0f} ms  {text!r}")

    def stat(key):
        v = np.array([r[key] for r in rows])
        return {"mean": round(float(v.mean()), 1),
                "p50": round(float(np.percentile(v, 50)), 1),
                "p90": round(float(np.percentile(v, 90)), 1),
                "min": round(float(v.min()), 1), "max": round(float(v.max()), 1)}

    # 循环口径的稳态服务节拍：末 30 轮 Σ 服务时间中位数倒数（Hz，不含轮间空隙）
    tail = np.array([r["t_total_ms"] for r in rows[-30:]])
    cadence_hz = round(1000.0 / float(np.median(tail)), 2)
    report = {"mode": args.mode, "rounds": len(rows), "task": args.task,
              "ckpt": args.ckpt, "sdk_present": robot is not None,
              "s2_stat": stat("t_s2_ms"), "s1_stat": stat("t_s1_ms"),
              "total_stat": stat("t_total_ms"),
              "loop_steady_hz": cadence_hz if args.mode == "loop" else None,
              "all_finite": all(r["finite"] for r in rows),
              "shapes_ok": all(r["actions_shape"] == [50, 14] for r in rows),
              "rows": rows}
    out = pathlib.Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.mode}.json").write_text(json.dumps(report, indent=2,
                                                      ensure_ascii=False))
    print(f"\n[{args.mode}] {len(rows)} 轮：S2 p50 {report['s2_stat']['p50']} / "
          f"S1 p50 {report['s1_stat']['p50']} / Σ p50 "
          f"{report['total_stat']['p50']} ms"
          + (f" | 稳态节拍 {cadence_hz} Hz" if args.mode == "loop" else ""))
    print(f"落盘 {out / (args.mode + '.json')}")

    if robot is not None:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
