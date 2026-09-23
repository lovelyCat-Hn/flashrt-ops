#!/usr/bin/env python
"""G1 双臂执行半环 smoke：pi0.5 动作 → SDK set_joint_positions → 机械臂（真机）。

⚠️ 与 run_g1_inference.py（只读）不同，本脚本加 --exec 后会真实驱动双臂！

安全设计（libero 权重动作对 G1 无语义，本脚本只验"推理→SDK→关节"链路）：
  1. 默认干跑：只打印计划目标，不发任何命令；--exec 才动
  2. 只动双臂 14 关节；腿/头维度永不下发；夹爪仅 --grip 显式开启时下发
     （0~100% → manifest 标定宽度，变化超 --grip-chg 才发，非阻塞）
  3. 每步目标 = 当前读数 ± delta-max 限幅（默认 0.05 rad ≈ 2.9°），模型说破天也只挪这些
  4. set_joint_positions 阻塞式 + speed 0.15 rad/s + timeout 兜底
  5. 执行前终端回车确认；每步回读关节对照 command vs achieved
用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/run_g1_execute.py \
      [--ckpt ~/holy/models/pi05_lerobot_base] [--exec] [--steps 3] \
      [--delta-max 0.05] [--force-state-dim] [--prompt "..."] [--grip]

真微调权重到位后: --ckpt <g1_ckpt_prep 产出的部署目录>，去掉 --force-state-dim，
语义即正确（夹爪维待标定钩子，见 run_g1_inference.py）。
"""
import argparse
import functools
import json
import os
import pathlib
import time

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_lerobot_base")
ap.add_argument("--prompt", default="Left arm pick up the block. Right arm pick up the block.")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实下发关节命令（默认干跑只打印）")
ap.add_argument("--steps", type=int, default=3, help="执行 chunk 前 K 步（≤10）")
ap.add_argument("--delta-max", type=float, default=0.05,
                help="每步相对当前读数的限幅（rad）")
ap.add_argument("--speed", type=float, default=0.15, help="关节速度上限 rad/s")
ap.add_argument("--force-state-dim", action="store_true",
                help="state/stats 维数不符时截断/零补适配（仅 smoke）")
ap.add_argument("--grip", action="store_true",
                help="启用夹爪下发（dim7/dim15 0~100%% → manifest 标定宽度）")
ap.add_argument("--grip-speed", type=float, default=0.05, help="夹爪速度 m/s")
ap.add_argument("--grip-effort", type=float, default=30, help="夹爪力矩 N")
ap.add_argument("--grip-chg", type=float, default=2.0,
                help="夹爪下发变化阈值 %%（小于它不重发）")
args = ap.parse_args()

# ── 部署清单（与 run_g1_inference.py 同规则）──
CKPT = pathlib.Path(args.ckpt)
mf_p = CKPT / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
ACTION_DIM = int(mf.get("action_dim", 16))
VIEWS = int(mf.get("views", 3))
CAM_MAP = mf.get("camera_map", {"image": "HEAD_RIGHT_CAMERA",
                                "wrist_image": "LEFT_ARM_CAMERA",
                                "wrist_image_right": "RIGHT_ARM_CAMERA"})

# ── 夹爪下发配置（--grip；宽度来自 manifest 标定，无标定拒绝开启）──
GRIP = None
if args.grip:
    g = mf.get("gripper", {})
    wmin, wmax = g.get("width_min"), g.get("width_max")
    if wmin is None or wmax is None:
        raise SystemExit("--grip 需要 manifest 夹爪标定；先重跑 "
                         "g1_ckpt_prep.py --grip-wmin/--grip-wmax（2026-09-23 实测 "
                         "0.0005/0.1200 m）")
    GRIP = {"names": (("right_gripper", 7), ("left_gripper", 15)),
            "wmin": float(wmin), "wmax": float(wmax),
            "speed": args.grip_speed, "effort": args.grip_effort,
            "chg": args.grip_chg, "sent": {}}
    print(f"夹爪下发开启: 0%→{wmin} m | 100%→{wmax} m | 速度 {args.grip_speed} m/s | "
          f"力矩 {args.grip_effort} N | 变化阈值 {args.grip_chg}%")

# ── 开关必须在 load_model 之前设 ──
os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")   # 全 INT8（echo 定档）
os.environ.setdefault("PI05_NO_GRAPH", "1")             # r35.5 驱动 graph 段错误绕法

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, SensorType, G1JointGroup  # noqa: E402

# ── 关节表（显式名——group 模式顺序不可信）──
# ⚠ 数据集/动作 chunk 维序【右臂在前】（meta/info.json 权威）：0-6 右臂 /
# 7 右夹爪 / 8-14 左臂 / 15 左夹爪。曾按左臂在前装配，右臂目标下给左臂
# （机械臂甩到背后）——2026-09-22 修正
LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT   # 与 chunk[0:7]+chunk[8:15] 逐位配对
STATE_NAMES = (RIGHT + ["right_gripper_joint1"]
               + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])
GRIP_DIMS = (7, 15)   # chunk 里的夹爪维（dim7 右 / dim15 左；--grip 开启才下发）


def read_joints(robot, names) -> np.ndarray:
    vals = robot.get_joint_positions([], names)
    if not vals or len(vals) != len(names):
        raise SystemExit(f"关节读取失败（{names[0]}...，返回 {len(vals) if vals else 0} 维）")
    return np.array(vals, dtype=np.float32)


def grab_views(robot) -> dict:
    out = {}
    for key, cam in list(CAM_MAP.items())[:VIEWS]:
        d = robot.get_rgb_data(getattr(SensorType, cam))
        if not d or not d.get("data"):
            raise SystemExit(f"取图失败: {cam}")
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"解码失败: {cam}")
        out[key] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out


def send_grip(robot, chunk_row):
    """chunk 行 dim7/dim15（0~100%）→ 标定宽度下发（超阈值才发，非阻塞）。

    ⚠ SDK 夹爪反馈滞后 ~6.3s 且 is_moving 恒 False——本函数只发不查，
    不等待（否则拖死控制环）；终态核对放执行结束后。
    """
    if GRIP is None:
        return None
    parts = []
    for name, dim in GRIP["names"]:
        p = float(np.clip(chunk_row[dim], 0.0, 100.0))
        last = GRIP["sent"].get(name)
        if last is not None and abs(p - last) < GRIP["chg"]:
            parts.append(f"{name[0].upper()} {p:.1f}%·hold")
            continue
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        st = robot.set_gripper_command(getattr(G1JointGroup, name),
                                       w, GRIP["speed"], GRIP["effort"], False)
        GRIP["sent"][name] = p
        parts.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm "
                     f"{str(st).replace('ControlStatus.', '')}")
    return "  ".join(parts)


# ── ① SDK 初始化（控制器前置，模式同 reset_pos.py 已验证用法）──
print("== ① SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
sensors = {getattr(SensorType, c) for c in list(CAM_MAP.values())[:VIEWS]}
if not robot.init(sensors):
    raise SystemExit("robot.init 失败（机器人上电？相机服务在跑？）")
time.sleep(5)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")
if not str(st).startswith("ControlStatus.SUCCESS"):
    print("⚠ 控制器启动非 SUCCESS，--exec 大概率会被拒，先干跑看数据")

# ── ② state + 模型 ──
state_raw = read_joints(robot, STATE_NAMES)
t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=VIEWS,
                            cache_frames=1, action_dim=ACTION_DIM)
ns = model._pipe.norm_stats
MODE = ns.get("norm_mode", "q01_q99") if isinstance(ns, dict) else "q01_q99"
try:
    state_n = normalize_state(state_raw, ns)
except ValueError as e:
    if not args.force_state_dim:
        raise SystemExit(f"state 维数与 stats 不符（{e}）；smoke 加 --force-state-dim") from e
    blk = ns.get("state") or ns["actions"]
    d = len(blk["q01" if MODE == "q01_q99" else "mean"])
    padded = np.zeros(d, np.float32)
    padded[:min(d, len(state_raw))] = state_raw[:d]
    state_n = normalize_state(padded, ns)
    print(f"[state] ⚠ 截断/零补 {len(state_raw)}→{d} 维适配 smoke stats")
print(f"tier=int8_full views={VIEWS} action_dim={ACTION_DIM} ckpt={CKPT.name} "
      f"| load {time.time() - t0:.1f}s | norm_mode={MODE}")

obs = grab_views(robot)
chunk = np.asarray(model.predict(obs, prompt=args.prompt, state=state_n))
print(f"predict → {chunk.shape}")
print(f"模型动作 chunk（前 3 步，未执行）:\n{np.round(chunk[:3], 3)}")
g = chunk[0][list(GRIP_DIMS)]
print(f"夹爪维首步值 {np.round(g, 3).tolist()}（0~100 语义"
      + ("，--grip 按标定下发" if GRIP else "，未开 --grip 不下发") + "）")

# ── ③ 计划 + 执行 ──
n_steps = max(1, min(args.steps, len(chunk)))
cur = read_joints(robot, ARM_NAMES)
print(f"\n== ③ 双臂执行计划（共 {n_steps} 步，每步限幅 ±{args.delta_max} rad）==")
print(f"当前读数: {np.round(cur, 3).tolist()}")
plan = []
for k in range(n_steps):
    tgt_raw = np.concatenate([chunk[k][:7], chunk[k][8:15]]).astype(np.float32)
    delta = np.clip(tgt_raw - cur, -args.delta_max, args.delta_max)
    plan.append(cur + delta)
    print(f"  步{k}: 模型目标 {np.round(tgt_raw, 3).tolist()}"
          f"\n      限幅后 {np.round(plan[-1], 3).tolist()}")
    if GRIP:
        for name, dim in GRIP["names"]:
            p = float(np.clip(chunk[k][dim], 0.0, 100.0))
            w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
            print(f"      夹爪[{name[0].upper()}]: {p:.1f}% → {w * 1000:.1f} mm")

if not args.do_exec:
    print("\n[干跑] 未下发任何命令。确认急停可及后，加 --exec 真实执行。")
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)

input(f"\n⚠ 将按 ≤{args.speed} rad/s 真实驱动双臂 {n_steps} 步"
      + ("并下发夹爪（标定宽度）" if GRIP else "") + "。急停就绪后回车开始，"
      f"Ctrl-C 中止...")

errs = []
try:
    for k in range(n_steps):
        cur_k = read_joints(robot, ARM_NAMES)          # 每步重读，限幅基准取实时值
        tgt_k = cur_k + np.clip(plan[k] - cur_k,
                                -args.delta_max, args.delta_max)
        t0 = time.perf_counter()
        st = robot.set_joint_positions(
            tgt_k.tolist(), joint_names=ARM_NAMES,
            is_blocking=True, speed_rad_s=args.speed, timeout_s=10.0)
        dt = (time.perf_counter() - t0) * 1000
        ach = read_joints(robot, ARM_NAMES)
        err = float(np.max(np.abs(ach - tgt_k)))
        errs.append(err)
        print(f"  步{k}: {st} | 耗时 {dt:.0f} ms | 回读最大偏差 {err * 1000:.1f} mrad")
        gp = send_grip(robot, chunk[k])
        if gp:
            print(f"      夹爪: {gp}")
        time.sleep(0.3)
finally:
    fin = read_joints(robot, ARM_NAMES)
    print(f"最终关节: {np.round(fin, 3).tolist()}")
    if GRIP:
        time.sleep(8.0)   # 夹爪反馈滞后 ~6.3s（实测），留足再读终态
        for name, _ in GRIP["names"]:
            gs = robot.get_gripper_state(getattr(G1JointGroup, name))
            if gs is not None:
                print(f"夹爪终态 {name}: {gs.width * 1000:.1f} mm "
                      f"(moving={gs.is_moving})")
    if errs:
        print(f"链路结论: {len(errs)} 步全部下发，回读最大偏差 "
              f"max {max(errs) * 1000:.1f} mrad（<20 mrad 视为到位）")
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    print("SDK 已关闭")
    os._exit(0)   # SDK 残留线程，干净退出（fleet 已知坑）
