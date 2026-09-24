#!/usr/bin/env python
"""G1 数据集观测回放真机执行：观测流=数据集（三相机帧+state），动作=模型推理，
驱动真机双臂+夹爪（bf16 定档）。

与 run_g1_loop.py 的区别：模型输入不走机器人相机/关节——全部来自数据集
episode（与 tf_rollout.py 验证过的输入完全一致），机器人只负责执行模型输出
（影子复现采集轨迹）。绕开场景摆位问题；真机侧关注点=执行跟踪质量。

安全设计（⚠ 加 --exec 会真实驱动双臂+夹爪！）：
  1. 只动双臂 14 关节 + 夹爪；腿/头永不下发
  2. 节拍执行（2026-09-24 衔接抖动定案）：每控制步仅 1 条臂目标=数据集
     参考位，速度=位移÷窗口（自适应 0.02~speed rad/s）——参考永远在臂
     前方，消除子步到点停车与"走完停等推理"（外部平滑方案根因 A/B 的
     SDK 等效实现：1ms RT 滤波层不可得，用速度匹配等效）；推理重叠在
     窗口内
  3. 漂移护栏：任一关节偏离起始位 > --max-excursion（config [loop].
     max_excursion=3.0，按 199 轨抓取包络标定）→ 立即停
  4. 轨迹锁定（2026-09-24 抖动定案）：到位确认 switch-dist 内才推进
     数据集时钟（超时 catch_timeout 警告推进），数据集自动慢放。纯时间
     锁会让参考以 ~0.23 rad/s 硬闯数据集 ~1.5 rad/s 快相位，臂满速追赶、
     滞后累计 1 rad 连续颤震；观测全来自数据集，慢放零语义损失
  5. 夹爪：模型 chunk dim7/15（绝对 %）→ manifest 标定宽度下发（非阻塞；
     用上一 chunk 末行，滞后 1 数据集帧）
  6. q 键即时退出；物理急停第一优先级；执行前回车确认

模型 state 输入用数据集值（含夹爪 33% 撑爪态），与训练/回放验证同源；
机器人夹爪反馈（滞后 ~6.3s）不进模型。

前置：g1_pose_warmup.py 预热到 task0 起始位（episode_start_task0.json）。
用法:
  ~/holy/run.sh ~/holy/scripts/inference/run_g1_replay.py [--ep 0] [--exec]
"""
import argparse
import ast
import functools
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--ep", type=int, default=0, help="数据集 episode 号（默认 0=task0）")
ap.add_argument("--stride", type=int, default=3, help="控制步长帧（3=数据集 10Hz）")
ap.add_argument("--start", type=int, default=0, help="起始帧")
ap.add_argument("--end", type=int, default=-1, help="结束帧（-1=到末尾）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实驱动（默认干跑：打印前 2 个控制步计划后退出；仅 CLI）")
ap.add_argument("--speed", type=float, default=None,
                help="关节速度上限 rad/s（config [replay].speed；节拍自适应下限 0.02）")
ap.add_argument("--pace", type=float, default=None,
                help="每控制步节拍窗口 s：本步位移摊满窗口发单条指令，推理重叠在内"
                     "（config [replay].pace）")
ap.add_argument("--max-excursion", type=float, default=None,
                help="偏离起始位护栏 rad（config [loop].max_excursion）")
ap.add_argument("--switch-dist", type=float, default=None,
                help="到位确认阈值 rad：回读误差收到该值即推进数据集时钟（config [loop].switch_dist）")
ap.add_argument("--catch-timeout", type=float, default=None,
                help="到位确认超时 s：超时警告并推进（config [replay].catch_timeout）")
ap.add_argument("--grip", action="store_true", default=None,
                help="夹爪下发（config [gripper].enabled）")
ap.add_argument("--no-grip", action="store_true", help="显式关闭夹爪下发")
ap.add_argument("--grip-speed", type=float, default=None, help="夹爪速度 m/s（config [gripper].speed）")
ap.add_argument("--grip-effort", type=float, default=None, help="夹爪力矩 N（config [gripper].effort）")
ap.add_argument("--grip-chg", type=float, default=None, help="夹爪重发阈值 %%（config [gripper].chg）")
ap.add_argument("--config", default=None, help="g1.toml 路径")
args = ap.parse_args()
sys.stdout.reconfigure(line_buffering=True)   # os._exit 不刷缓冲，全部行缓冲兜底

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import g1_config  # noqa: E402

g1_config.apply(args, {
    "ckpt": ("run", "ckpt"),
    "speed": ("replay", "speed"),
    "pace": ("replay", "pace"),
    "max_excursion": ("loop", "max_excursion"),
    "switch_dist": ("loop", "switch_dist"),
    "catch_timeout": ("replay", "catch_timeout"),
    "grip": ("gripper", "enabled"),
    "grip_speed": ("gripper", "speed"),
    "grip_effort": ("gripper", "effort"),
    "grip_chg": ("gripper", "chg"),
})
if args.no_grip:
    args.grip = False

# bf16 定档（2026-09-23 tf_matrix 实证：INT8 毁动作质量）——必须在 import 前
os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
os.environ.setdefault("PI05_NO_GRAPH", "1")
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")

import cv2  # noqa: E402
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, G1JointGroup  # noqa: E402

# ── 数据集（episode npz 缓存，与 tf_matrix/tf_rollout 共用）──
DS = pathlib.Path("/home/galbot/holy/datasets/pick_place_balence")
CAM = (("image", "head_right"), ("wrist_image", "left_arm"),
       ("wrist_image_right", "right_arm"))
ARM_DIMS = list(range(0, 7)) + list(range(8, 15))   # state 里的臂维（右 7 在前）

CACHE = pathlib.Path(f"/tmp/tf_matrix_ep{args.ep}.npz")
if not CACHE.exists():
    print(f"抽取 episode {args.ep} → {CACHE}（系统 python3，一次性）", flush=True)
    subprocess.run([
        "/usr/bin/python3", "-c",
        """
import pathlib, numpy as np, pandas as pd, sys
ds = pathlib.Path(sys.argv[1]); out = sys.argv[2]; ep = int(sys.argv[3])
data = pd.read_parquet(ds / "data/chunk-000/file-000.parquet")
meta = pd.read_parquet(ds / "meta/episodes/chunk-000/file-000.parquet")
row = meta[meta.episode_index == ep].iloc[0]
df = data[data.episode_index == ep].reset_index(drop=True)
np.savez(out,
         states=np.stack([np.asarray(s, np.float32) for s in df["observation.state"]]),
         actions=np.stack([np.asarray(a, np.float32) for a in df["action"]]),
         tasks=np.array([str(row["tasks"])]),
         t0_head=float(row["videos/observation.images.head_right/from_timestamp"]),
         t0_left=float(row["videos/observation.images.left_arm/from_timestamp"]),
         t0_right=float(row["videos/observation.images.right_arm/from_timestamp"]))
print(f"ep{ep}: {len(df)} 帧 → {out}", flush=True)
""",
        str(DS), str(CACHE), str(args.ep)], check=True)

_z = np.load(CACHE)
states, actions = _z["states"], _z["actions"]
try:
    TASK = ast.literal_eval(str(_z["tasks"][0]))[0]
except Exception:
    TASK = str(_z["tasks"][0])
T0_TS = {"head_right": float(_z["t0_head"]), "left_arm": float(_z["t0_left"]),
         "right_arm": float(_z["t0_right"])}
end = len(states) - 1 - args.stride if args.end < 0 else min(args.end, len(states) - 1 - args.stride)
ctrl_ts = list(range(args.start, end, args.stride))
print(f"ep{args.ep} {len(states)} 帧 | 回放 {args.start}~{end} stride={args.stride} "
      f"→ {len(ctrl_ts)} 控制步 | 任务: {TASK}", flush=True)


def frame_at(cam, t):
    p = DS / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_MSEC, (T0_TS[cam] + t / 30.0) * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise SystemExit(f"取帧失败: ep{args.ep} t={t} {cam}")
    return np.ascontiguousarray(
        cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))


# ── 关节表（数据集维序：右臂在前）──
LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT            # 与 chunk[0:7]+chunk[8:15] 逐位配对

# ── 夹爪下发配置（manifest 标定）──
CKPT = pathlib.Path(args.ckpt)
mf = json.loads((CKPT / "flashrt_deploy.json").read_text()) \
    if (CKPT / "flashrt_deploy.json").exists() else {}
GRIP = None
if args.grip:
    g = mf.get("gripper", {})
    wmin, wmax = g.get("width_min"), g.get("width_max")
    if wmin is None or wmax is None:
        raise SystemExit("--grip 需要 manifest 夹爪标定；先重跑 g1_ckpt_prep.py "
                         "--grip-wmin/--grip-wmax（2026-09-23 实测 0.0005/0.1200 m）")
    GRIP = {"names": (("right_gripper", 7), ("left_gripper", 15)),
            "wmin": float(wmin), "wmax": float(wmax),
            "speed": args.grip_speed, "effort": args.grip_effort,
            "chg": args.grip_chg, "sent": {}}
    print(f"夹爪下发开启: 0%→{wmin} m | 100%→{wmax} m | 速度 {args.grip_speed} m/s | "
          f"变化阈值 {args.grip_chg}%")
_gcal = mf.get("gripper", {})


def send_grip(robot, chunk_row):
    """chunk 行 dim7/dim15（0~100%）→ 标定宽度下发（超阈值才发，非阻塞）。

    反馈滞后 ~6.3s 只发不查（等反馈会拖死回放节奏）；终态核对放结束后。
    """
    if GRIP is None:
        return None
    parts = []
    for name, dim in GRIP["names"]:
        p = float(np.clip(chunk_row[dim], 0.0, 100.0))
        last = GRIP["sent"].get(name)
        if last is not None and abs(p - last) < GRIP["chg"]:
            continue
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        st = robot.set_gripper_command(getattr(G1JointGroup, name),
                                       w, GRIP["speed"], GRIP["effort"], False)
        GRIP["sent"][name] = p
        parts.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm "
                     f"{str(st).replace('ControlStatus.', '')}")
    return "  ".join(parts)


def read_joints(robot, names):
    vals = robot.get_joint_positions([], names)
    if not vals or len(vals) != len(names):
        raise SystemExit(f"关节读取失败（返回 {len(vals) if vals else 0} 维）")
    return np.array(vals, dtype=np.float32)


# ── q 键即时退出 ──
import atexit  # noqa: E402
import select  # noqa: E402
import termios  # noqa: E402
import tty  # noqa: E402


class QuitWatcher:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._old = None
        if os.isatty(self.fd):
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.restore)
            threading.Thread(target=self._loop, daemon=True).start()
            print("（回放期间随时按 q 退出；确认提示符处用回车；急停第一优先级）", flush=True)

    def restore(self):
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def _loop(self):
        while True:
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            if sys.stdin.read(1) in ("q", "Q"):
                print("\n⛔ 按下 q —— 立即退出（已下发目标可能仍在限速执行；"
                      "需立即断运动请拍急停）")
                self.restore()
                os._exit(2)


WATCH = QuitWatcher()


def _fatal_hook(t, v, tb):
    print(f"\n⛔ 异常退出: {t.__name__}: {v}", flush=True)
    rob = globals().get("robot")
    if rob is not None:
        try:
            rob.request_shutdown(); rob.wait_for_shutdown(); rob.destroy()
        except Exception:
            pass
    WATCH.restore()
    os._exit(1)


sys.excepthook = _fatal_hook

# ── ① SDK 初始化（不取相机流——模型观测全来自数据集）──
print("\n== ① SDK 初始化（确认急停可及！）==", flush=True)
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(5)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}", flush=True)
if not str(st).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止（先排查控制器）")

HOME = read_joints(robot, ARM_NAMES)   # 起始位 = 护栏基准（应为 task0 预热位）
ds0 = states[args.start][ARM_DIMS]
off = float(np.max(np.abs(HOME - ds0)))
print(f"起始位: {np.round(HOME, 3).tolist()}", flush=True)
print(f"与数据集 ep{args.ep} 帧起始臂位最大偏差 {off * 1000:.0f} mrad"
      + ("（⚠ >300 mrad：先跑 g1_pose_warmup.py 预热到 task0 起始位）" if off > 0.3 else ""))

# ── ② 模型 ──
print(f"\n== ② 加载模型 {CKPT.name}（bf16）==", flush=True)
t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
ns = model._pipe.norm_stats
print(f"模型就绪 load {time.time() - t0:.1f}s", flush=True)

SW = args.switch_dist
V_MIN = 0.02            # 节拍速度下限 rad/s（平稳档，实测冲击∝速度）


def predict_step(t):
    """数据集帧 t 的观测 → 模型 chunk（模型 state 输入=数据集原始值归一化）。"""
    imgs = {v: frame_at(c, t) for v, c in CAM}
    st_n = normalize_state(np.asarray(states[t], np.float32), ns)
    t1 = time.perf_counter()
    chunk = np.asarray(model.predict(imgs, prompt=TASK, state=st_n))
    return chunk, (time.perf_counter() - t1) * 1000


# ── ③ 干跑：前 2 个控制步计划，绝不下发 ──
if not args.do_exec:
    print(f"\n== [干跑] 节拍模式：每控制步 1 条臂目标（速度自适应 {V_MIN}~{args.speed} rad/s，"
          f"窗口 {args.pace:.2f} s，推理重叠在内）未下发任何命令 ==")
    prev_chunk = None
    for t in ctrl_ts[:2]:
        goal = np.asarray(states[min(t + args.stride, len(states) - 1)][ARM_DIMS], np.float32)
        cur = read_joints(robot, ARM_NAMES)
        gap = float(np.max(np.abs(goal - cur)))
        pace = float(np.clip(gap / args.pace, V_MIN, args.speed))
        chunk, ms = predict_step(t)
        grip_txt = "-"
        if GRIP and prev_chunk is not None:
            row = prev_chunk[min(args.stride, len(prev_chunk)) - 1]
            grip_txt = f"{np.clip(row[7], 0, 100):.1f}/{np.clip(row[15], 0, 100):.1f}%"
        print(f"  帧 {t}（推理 {ms:.0f} ms）: 目标 gap {gap * 1000:.0f} mrad → "
              f"节拍速度 {pace:.3f} rad/s"
              f"\n    离起始位峰值 {float(np.max(np.abs(goal - HOME))) * 1000:.0f} mrad | "
              f"夹爪(上chunk末行) {grip_txt}", flush=True)
        prev_chunk = chunk
    print("\n[干跑] 未下发任何命令。确认预热到位后加 --exec 真实回放。", flush=True)
    WATCH.restore()
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)

# ── ④ 回放执行 ──
input(f"\n⚠ 将按数据集 ep{args.ep} 节拍回放真机执行 {len(ctrl_ts)} 控制步"
      f"（每步 1 条目标、速度自适应 {V_MIN}~{args.speed} rad/s 摊满 {args.pace:.2f} s 窗口、"
      f"护栏 ±{args.max_excursion} rad"
      + ("，夹爪下发开启）" if GRIP else "）") + "。急停就绪后回车开始，随时按 q 退出...")

# ── ④ 节拍回放（2026-09-24 衔接抖动定案的 SDK 等效实现）：
# 每控制步仅 1 条臂目标=数据集参考位，速度=位移÷窗口（自适应 V_MIN~speed），
# 参考永远在臂前方 → 无"走完停等推理"的到点停车；推理与运动重叠在窗口内；
# 夹爪用上一 chunk 末行（滞后 1 数据集帧）。到位确认（switch-dist）保持
# 轨迹锁定：仅当节拍速度被 cap 时才需要等待，数据集自动慢放。
lag_l, ms_l, pace_l, gap_l, exc_max = [], [], [], [], 0.0
n_cmd = n_skip = 0
t_start = time.perf_counter()
aborted = False
prev_chunk = None
for t in ctrl_ts:
    # ① 本步参考位 + 护栏
    goal = np.asarray(states[min(t + args.stride, len(states) - 1)][ARM_DIMS], np.float32)
    drift = float(np.max(np.abs(goal - HOME)))
    exc_max = max(exc_max, drift)
    if drift > args.max_excursion:
        print(f"⛔ 漂移护栏：{drift * 1000:.0f} mrad > "
              f"{args.max_excursion * 1000:.0f}，停止（机械臂留在原地）", flush=True)
        aborted = True
        break
    # ② 节拍发令：速度把本步位移摊满窗口
    cur = read_joints(robot, ARM_NAMES)
    gap = float(np.max(np.abs(goal - cur)))
    gap_l.append(gap)
    if gap > 5e-3:
        pace = float(np.clip(gap / args.pace, V_MIN, args.speed))
        robot.set_joint_positions(goal.tolist(), joint_names=ARM_NAMES,
                                  is_blocking=False, speed_rad_s=pace)
        n_cmd += 1
        pace_l.append(pace)
    else:
        pace = 0.0        # 数据集驻停：不发令（臂保持）
        n_skip += 1
    # ③ 夹爪：上一 chunk 末行（滞后 1 数据集帧，时序精度够用）
    if GRIP and prev_chunk is not None:
        gp = send_grip(robot, prev_chunk[min(args.stride, len(prev_chunk)) - 1])
        if gp:
            print(f"  帧 {t} 夹爪: {gp}", flush=True)
    # ④ 推理与运动重叠
    chunk, ms = predict_step(t)
    ms_l.append(ms)
    # ⑤ 补满节拍窗口
    t_s = time.perf_counter()
    while time.perf_counter() - t_s < args.pace:
        time.sleep(0.05)
    # ⑥ 到位确认（节拍速度被 cap 时才需要等）
    t_g = time.perf_counter()
    while True:
        lag = float(np.max(np.abs(read_joints(robot, ARM_NAMES) - goal)))
        if lag <= SW:
            break
        if time.perf_counter() - t_g > args.catch_timeout:
            print(f"  ⚠ 到位超时 {args.catch_timeout:.0f} s（滞后 {lag * 1000:.0f} mrad），推进",
                  flush=True)
            break
        time.sleep(0.05)
    lag_l.append(lag)
    prev_chunk = chunk
    print(f"  帧 {t:>3}/{end} | gap {gap * 1000:3.0f} mrad | 速度 {pace:.3f} | "
          f"推理 {ms:4.0f} ms | 滞后 {lag * 1000:4.0f} mrad | "
          f"累计 {time.perf_counter() - t_start:5.1f} s", flush=True)

# ── ⑤ 汇总 ──
if GRIP:
    time.sleep(8.0)   # 夹爪反馈滞后 ~6.3s，留足再读终态
    for name, _ in GRIP["names"]:
        gs = robot.get_gripper_state(getattr(G1JointGroup, name))
        if gs is not None:
            print(f"夹爪终态 {name}: {gs.width * 1000:.1f} mm (moving={gs.is_moving})", flush=True)
a = np.asarray(lag_l) * 1000 if lag_l else np.array([0.0])
m = np.asarray(ms_l) if ms_l else np.array([0.0])
g = np.asarray(gap_l) * 1000 if gap_l else np.array([0.0])
p = np.asarray(pace_l) if pace_l else np.array([0.0])
print(f"""
== 回放汇总（{'⚠ 护栏中止' if aborted else '完成'}）==
控制步 {len(lag_l)}/{len(ctrl_ts)} | 臂指令 {n_cmd} 条（驻停跳过 {n_skip}） | 总时长 {time.perf_counter() - t_start:.0f} s
跟踪滞后（vs 数据集轨迹）: p50 {np.percentile(a, 50):.0f} / max {a.max():.0f} mrad
发令衔接间隙: p50 {np.percentile(g, 50):.0f} / p95 {np.percentile(g, 95):.0f} mrad（≈每步数据集位移，参考 <20）
节拍速度: p50 {np.percentile(p, 50):.3f} / max {p.max():.3f} rad/s（窗口 {args.pace:.2f} s）
推理: p50 {np.percentile(m, 50):.0f} / p95 {np.percentile(m, 95):.0f} ms（bf16）
离起始位峰值: {exc_max * 1000:.0f} mrad（护栏 {args.max_excursion * 1000:.0f}）""", flush=True)

WATCH.restore()
print("SDK 关闭中...", flush=True)
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
os._exit(0)   # SDK 残留线程，干净退出
