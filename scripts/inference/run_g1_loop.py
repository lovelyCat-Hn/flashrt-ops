#!/usr/bin/env python
"""G1 双臂闭环 receding-horizon：推理→执行→再观测 连续循环（真机）。

目的：实测执行链路的实时延迟与抖动（2026-09-22 闭环打通后的量化步骤）。
⚠ 加 --exec 会连续真实驱动双臂！安全设计：
  1. 只动双臂 14 关节；夹爪/腿/头维度永不下发
  2. 每步目标 = 当前读数 ± --delta-max 限幅（默认 0.05 rad），限速 0.15 rad/s
  3. 漂移护栏：任一关节偏离起始位超 --max-excursion（默认 0.25 rad）→ 立即
     停止循环（语义测试期防模型单向漂移拖走机械臂）
  4. q 键即时退出（后台监听线程，SDK 阻塞中也可退）；物理急停第一优先级
  5. 执行前回车确认

每轮遥测：推理 ms / 取图 ms / 每步执行 ms / 回读跟踪误差 mrad /
相邻步指令增量（抖动代理）/ 往返反转计数（振荡指标）/ 汇总分位数。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/run_g1_loop.py \
      [--ckpt ~/holy/models/pi05_g1_deploy] [--exec] \
      [--rounds 10] [--steps-per-round 3] [--delta-max 0.05] \
      [--speed 0.15] [--max-excursion 0.25] [--prompt "..."]
"""
import argparse
import atexit
import functools
import json
import os
import pathlib
import select
import sys
import termios
import threading
import time
import tty

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_deploy")
ap.add_argument("--prompt", default="Left arm pick up the block. Right arm pick up the block.")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实连续驱动双臂（默认干跑只打印首轮计划）")
ap.add_argument("--rounds", type=int, default=10, help="推理→执行循环轮数")
ap.add_argument("--steps-per-round", type=int, default=3,
                help="每轮执行 chunk 前 K 步（模型 10 步/次）")
ap.add_argument("--delta-max", type=float, default=0.05, help="每步限幅 rad")
ap.add_argument("--speed", type=float, default=0.15, help="关节速度上限 rad/s")
ap.add_argument("--max-excursion", type=float, default=0.25,
                help="偏离起始位护栏 rad（任一关节超限即停）")
args = ap.parse_args()


class QuitWatcher:
    """后台线程监听键盘 q：任何阶段即时退出（SDK 阻塞 C++ 调用吞 Ctrl-C）。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._old = None
        self.active = threading.Event()
        if os.isatty(self.fd):
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.restore)
            self.active.set()
            threading.Thread(target=self._loop, daemon=True).start()
            print("（循环期间随时按 q 退出；确认提示符处用回车/Ctrl-C；急停第一优先级）")

    def restore(self):
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def pause(self):
        self.active.clear()

    def resume(self):
        self.active.set()

    def _loop(self):
        while True:
            if not self.active.is_set() or not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            if sys.stdin.read(1) in ("q", "Q"):
                print("\n⛔ 按下 q —— 立即退出（已下发目标可能仍在限速执行）")
                self.restore()
                os._exit(2)


WATCH = QuitWatcher()


def _fatal_hook(t, v, tb):
    """任何未捕获异常：关 SDK + 还原终端 + 硬退。

    裸退会让 SDK 残留线程在解释器关闭时段错误（fleet 已知坑）。
    """
    print(f"\n⛔ 异常退出: {t.__name__}: {v}")
    rob = globals().get("robot")
    if rob is not None:
        try:
            rob.request_shutdown(); rob.wait_for_shutdown(); rob.destroy()
        except Exception:
            pass
    WATCH.restore()
    os._exit(1)


sys.excepthook = _fatal_hook

# ── 部署清单 ──
CKPT = pathlib.Path(args.ckpt)
mf_p = CKPT / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
ACTION_DIM = int(mf.get("action_dim", 16))
VIEWS = int(mf.get("views", 3))
CAM_MAP = mf.get("camera_map", {"image": "HEAD_RIGHT_CAMERA",
                                "wrist_image": "LEFT_ARM_CAMERA",
                                "wrist_image_right": "RIGHT_ARM_CAMERA"})

os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
os.environ.setdefault("PI05_NO_GRAPH", "1")   # r35.5 graph 段错误绕法

import numpy as np  # noqa: E402
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
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402

# ── 关节表（数据集维序：右臂在前）──
LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT            # 与 chunk[0:7]+chunk[8:15] 逐位配对
STATE_NAMES = (RIGHT + ["right_gripper_joint1"]
               + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])


def read_joints(robot, names) -> np.ndarray:
    vals = robot.get_joint_positions([], names)
    if not vals or len(vals) != len(names):
        raise SystemExit(f"关节读取失败（返回 {len(vals) if vals else 0} 维）")
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


def pstats(ms):
    a = np.array(ms)
    return (f"p50 {np.percentile(a, 50):.1f} | p95 {np.percentile(a, 95):.1f} | "
            f"max {a.max():.1f} ms")


# ── ① SDK 初始化 + 模型 ──
print("== ① SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
sensors = {getattr(SensorType, c) for c in list(CAM_MAP.values())[:VIEWS]}
if not robot.init(sensors):
    raise SystemExit("robot.init 失败")
time.sleep(5)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")
if not str(st).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止（先跑 run_g1_execute.py 排查）")

HOME = read_joints(robot, ARM_NAMES)   # 起始位 = 漂移护栏基准（应在预热工作位）
print(f"起始位（漂移护栏基准）: {np.round(HOME, 3).tolist()}")
print("⚠ 若当前不是预热工作位，先跑 g1_pose_warmup.py 再来")

t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=VIEWS,
                            cache_frames=1, action_dim=ACTION_DIM)
ns = model._pipe.norm_stats
print(f"tier=int8_full views={VIEWS} | load {time.time() - t0:.1f}s")

obs = grab_views(robot)
state0 = read_joints(robot, STATE_NAMES)
state_n0 = normalize_state(state0, ns)
chunk = np.asarray(model.predict(obs, prompt=args.prompt, state=state_n0))  # 建管线
# 首用引擎构建吸收：真实控制轮 0 曾撞 ~800ms 惰性构建（autotune 中途重现，
# 2026-09-22 tegrastats 已排除热/内存）。连做 3 次新鲜取图推理，把构建成本
# 烧在计时区外，避免首轮尖峰污染遥测
for _ in range(3):
    model.predict(grab_views(robot), prompt=args.prompt,
                  state=normalize_state(read_joints(robot, STATE_NAMES), ns))
print("预热推理 ×3 完成（吸收惰性引擎构建）")
n_steps = max(1, min(args.steps_per_round, 10))


def plan_step(k, cur):
    """chunk 第 k 步 → 限幅后目标（14 维，右臂在前）。"""
    arm_tgt = np.concatenate([chunk[k][:7], chunk[k][8:15]])
    return cur + np.clip(arm_tgt - cur, -args.delta_max, args.delta_max)


# ── ② 干跑：只打印首轮计划，绝不下发 ──
if not args.do_exec:
    print(f"\n== [干跑] 首轮 {n_steps} 步计划（未下发任何命令）==")
    for k in range(n_steps):
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(k, cur)
        drift = float(np.max(np.abs(tgt - HOME)))
        print(f"  步{k}: 限幅后 {np.round(tgt, 3).tolist()}"
              f"\n        离起始位峰值 {drift * 1000:.0f} mrad"
              f"（护栏 {args.max_excursion * 1000:.0f}）")
    print("\n[干跑] 未下发任何命令。加 --exec 真实执行。")
    WATCH.restore()
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)

# ── ③ 连续循环 ──
WATCH.pause()
input(f"\n⚠ 将连续 {args.rounds} 轮 × 每轮 {n_steps} 步真实驱动双臂"
      f"（限幅 ±{args.delta_max} rad，漂移护栏 ±{args.max_excursion} rad）。\n"
      "急停就绪后回车开始，循环期间随时按 q 退出...")
WATCH.resume()

infer_ms, grab_ms, round_ms, step_ms, track_err = [], [], [], [], []
cmd_hist = None
aborted = False

for r in range(args.rounds):
    if aborted:
        break
    t_r = time.perf_counter()
    t_g = time.perf_counter()
    obs = grab_views(robot)
    state_i = read_joints(robot, STATE_NAMES)
    state_n = normalize_state(state_i, ns)
    grab_ms.append((time.perf_counter() - t_g) * 1000)

    t_i = time.perf_counter()
    chunk = np.asarray(model.predict(obs, prompt=args.prompt, state=state_n))
    infer_ms.append((time.perf_counter() - t_i) * 1000)

    print(f"\n── 轮 {r} | 取图 {grab_ms[-1]:.0f} | 推理 {infer_ms[-1]:.0f} ms ──")
    for k in range(n_steps):
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(k, cur)
        drift = float(np.max(np.abs(tgt - HOME)))
        if drift > args.max_excursion:
            print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                  f"{args.max_excursion * 1000:.0f}，停止循环（机械臂留在原地）")
            aborted = True
            break
        cmd_delta = (tgt - cmd_hist) if cmd_hist is not None else np.zeros_like(tgt)
        cmd_hist = tgt.copy()
        t_s = time.perf_counter()
        st = robot.set_joint_positions(tgt.tolist(), joint_names=ARM_NAMES,
                                       is_blocking=True, speed_rad_s=args.speed,
                                       timeout_s=10.0)
        step_ms.append((time.perf_counter() - t_s) * 1000)
        ach = read_joints(robot, ARM_NAMES)
        err = float(np.max(np.abs(ach - tgt))) * 1000
        track_err.append(err)
        print(f"  步{k}: |Δcmd| {float(np.max(np.abs(cmd_delta))) * 1000:5.1f} mrad | "
              f"执行 {step_ms[-1]:5.0f} ms | 回读偏差 {err:4.1f} mrad | {st}")
        if not str(st).startswith("ControlStatus.SUCCESS"):
            print("⛔ 下发非 SUCCESS，停止循环")
            aborted = True
            break
    round_ms.append((time.perf_counter() - t_r) * 1000)
    print(f"  轮耗时 {round_ms[-1]:.0f} ms（重规划频率 {1000 / round_ms[-1]:.1f} Hz）")

# ── ④ 汇总 ──
fin = read_joints(robot, ARM_NAMES)
exc = float(np.max(np.abs(fin - HOME))) * 1000
print(f"\n== 汇总（{len(round_ms)} 轮 / {len(step_ms)} 步）==")
if infer_ms:
    print(f"推理: {pstats(infer_ms)}")
if grab_ms:
    print(f"取图+读关节: {pstats(grab_ms)}")
if step_ms:
    print(f"单步执行(阻塞): {pstats(step_ms)}")
if round_ms:
    print(f"整轮(重规划周期): {pstats(round_ms)}")
if track_err:
    print(f"跟踪误差: mean {np.mean(track_err):.1f} | max {max(track_err):.1f} mrad")
    print("抖动判读：|Δcmd| 快速变号=抖动；持续同号=漂移（由护栏兜底）")
print(f"最终偏离起始位: {exc:.0f} mrad（护栏 {args.max_excursion * 1000:.0f} mrad）"
      + (" ⛔ 护栏触发过" if aborted else ""))

WATCH.restore()
print("SDK 关闭中...")
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
os._exit(0)   # SDK 残留线程，干净退出
