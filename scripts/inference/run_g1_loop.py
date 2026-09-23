#!/usr/bin/env python
"""G1 双臂闭环 receding-horizon：推理→执行→再观测 连续循环（真机）。

目的：实测执行链路的实时延迟与抖动（2026-09-22 闭环打通后的量化步骤）。
⚠ 加 --exec 会连续真实驱动双臂！安全设计：
  1. 只动双臂 14 关节；夹爪/腿/头维度永不下发
  2. 每条指令目标 = 当前读数 ± steps-per-cmd×--delta-max 限幅（默认
     1×0.05 rad；合步 3 → 0.15 rad，与逐 3 步一轮的行程上限相同），限速 --speed
  3. 漂移护栏：任一关节偏离起始位超 --max-excursion（默认 0.25 rad）→ 立即
     停止循环（语义测试期防模型单向漂移拖走机械臂）
  4. q 键即时退出（后台监听线程，SDK 阻塞中也可退）；物理急停第一优先级
  5. 执行前回车确认

每轮遥测：推理 ms / 取图 ms / 每步执行 ms / 回读跟踪误差 mrad /
相邻步指令增量（抖动代理）/ 汇总分位数。
流水线：下一块推理在本轮首条指令执行期间后台完成（predict-only 入后台，
SDK 调用全留主线程）；配额步消费完而推理未就绪时，用旧 chunk 剩余步
"续航"追踪（限幅/护栏不变，观测滞后相应加长），推理再慢也不站桩，
chunk 耗尽才需等待。

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
ap.add_argument("--steps-per-cmd", type=int, default=1,
                help="合步：一条 SDK 指令跨 K 个 chunk 步（默认 1=逐步；"
                     "3=整轮一条平滑轮廓，起停次数 1/3，实测提速只会加剧"
                     "每点全停的冲击，减停顿才是平滑正解）")
ap.add_argument("--delta-max", type=float, default=0.05, help="每步限幅 rad")
ap.add_argument("--speed", type=float, default=0.15, help="关节速度上限 rad/s")
ap.add_argument("--max-excursion", type=float, default=0.25,
                help="偏离起始位护栏 rad（任一关节超限即停）")
ap.add_argument("--settle", action="store_true",
                help="步进-停走（阻塞等待到位，旧行为）；默认追踪式：误差收窄即发下一目标")
ap.add_argument("--settle-frac", type=float, default=0.3,
                help="追踪式换目标阈值：剩余误差 < frac×delta-max 即发下一步")
ap.add_argument("--switch-dist", type=float, default=0.0,
                help="提前换目标阈值 rad：剩余误差收到该值即重定向，滑行中转向、"
                     "速度不过零，消指令边界停顿；0=旧语义（近乎到位才换）")
ap.add_argument("--chunk-mode", choices=("track", "settle", "traj"), default="track",
                help="步进引擎：track=追踪单步 / settle=停走 / traj=整块轨迹流"
                     "（PVT 原生，最平滑；--steps-per-round 不适用，整 chunk 一次发）")
ap.add_argument("--traj-dt", type=float, default=0.1,
                help="traj 模式轨迹点周期 s（0.033=采集原速 30fps；默认 0.1=3 倍慢放）")
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
# state 以十进制文本拼进 prompt（format_pi05_prompt）：关节值一漂、bin 数位
# 变化 → token 数变；默认 exact 模式每种长度一条 pipeline，换长=整条重建+
# 重 autotune（~800ms，2026-09-23 合成实验实锤）。fixed=定长 200 一条
# pipeline 只换 embeds，实测含 state 切换恒定 240-253ms
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")

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
from galbot_sdk.g1 import (  # noqa: E402
    GalbotRobot, SensorType, Trajectory, TrajectoryPoint, JointCommand)

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
spc = max(1, min(args.steps_per_cmd, n_steps))   # 合步宽度（≤ 每轮步数）
BUDGET = spc * args.delta_max                    # 每条指令位移限幅


def plan_step(k, cur, budget=None):
    """chunk 第 k 步 → 限幅后目标（14 维，右臂在前）；budget=位移限幅。"""
    arm_tgt = np.concatenate([chunk[k][:7], chunk[k][8:15]])
    b = args.delta_max if budget is None else budget
    return cur + np.clip(arm_tgt - cur, -b, b)


def build_traj(chunk, cur0):
    """chunk → 整块 PVT 轨迹。逐点追踪式钳位推进；任一点越护栏即截断。

    返回 (traj, n_pts, final_p)；n_pts=0 表示首点就越护栏（勿下发）。
    """
    traj = Trajectory()
    traj.joint_names = ARM_NAMES
    pts, p = [], cur0.copy()
    for k in range(len(chunk)):
        arm_tgt = np.concatenate([chunk[k][:7], chunk[k][8:15]])
        new_p = p + np.clip(arm_tgt - p, -args.delta_max, args.delta_max)
        if float(np.max(np.abs(new_p - HOME))) > args.max_excursion:
            return traj, k, (p if k else None)   # p = 最后一个合法点
        p = new_p
        tp = TrajectoryPoint()
        tp.time_from_start_second = round((k + 1) * args.traj_dt, 4)
        vec = []
        for v in p:
            c = JointCommand()
            c.position = float(v)
            vec.append(c)
        tp.joint_command_vec = vec
        pts.append(tp)
    traj.points = pts
    return traj, len(chunk), p


# ── ② 干跑：只打印首轮计划，绝不下发 ──
if not args.do_exec:
    n_cmd = (n_steps + spc - 1) // spc
    print(f"\n== [干跑] 首轮 {n_steps} 步（合步 {spc} 步/指令 → {n_cmd} 条指令，"
          f"每条位移限幅 ±{BUDGET} rad）未下发任何命令 ==")
    for k in range(0, n_steps, spc):
        k_end = min(k + spc, n_steps)
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(k_end - 1, cur, BUDGET)
        drift = float(np.max(np.abs(tgt - HOME)))
        print(f"  指令[步{k}-{k_end - 1}]: 限幅后 {np.round(tgt, 3).tolist()}"
              f"\n        离起始位峰值 {drift * 1000:.0f} mrad"
              f"（护栏 {args.max_excursion * 1000:.0f}）")
    print("\n[干跑] 未下发任何命令。加 --exec 真实执行。")
    WATCH.restore()
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)

# ── ③ 连续循环（流水线：推理与执行重叠，消除轮间停顿）──
WATCH.pause()
input(f"\n⚠ 将连续 {args.rounds} 轮 × 每轮 {n_steps} 步真实驱动双臂"
      f"（合步 {spc} 步/指令，每条限幅 ±{BUDGET} rad，漂移护栏 ±{args.max_excursion} rad）。\n"
      "急停就绪后回车开始，循环期间随时按 q 退出...")
WATCH.resume()

infer_ms, grab_ms, round_ms, step_ms, track_err = [], [], [], [], []
cmd_hist = None
aborted = False
sustain_cmds = 0                     # 配额外"续航"指令条数（掩盖慢推理）


class _PredictJob:
    """后台推理作业：GPU 推理与 SDK 阻塞执行重叠。

    只有 predict 放后台；SDK 取图/读关节/下发全部留在主线程，不碰 SDK
    线程安全。同一时刻只有一个 predict 在飞（消费完才启动下一个）。
    """

    def __init__(self, model, prompt):
        self._model, self._prompt = model, prompt
        self._done = threading.Event()
        self._res = None
        self._err = None
        self.dur_ms = 0.0

    def start(self, obs, state_n):
        self._done.clear()
        self._err = None

        def _run():
            t = time.perf_counter()
            try:
                self._res = np.asarray(self._model.predict(
                    obs, prompt=self._prompt, state=state_n))
            except Exception as e:      # 主线程 result() 时再抛
                self._err = e
            self.dur_ms = (time.perf_counter() - t) * 1000
            self._done.set()

        threading.Thread(target=_run, daemon=True).start()

    def done(self):
        return self._done.is_set()

    def result(self):
        self._done.wait()
        if self._err is not None:
            raise self._err
        return self._res


def fresh_obs():
    """取图+读关节+归一化（主线程，~22ms），耗时计入 grab_ms。"""
    t_g = time.perf_counter()
    obs = grab_views(robot)
    st_n = normalize_state(read_joints(robot, STATE_NAMES), ns)
    grab_ms.append((time.perf_counter() - t_g) * 1000)
    return obs, st_n


job = _PredictJob(model, args.prompt)
job.start(*fresh_obs())              # 轮 0 的 chunk

for r in range(args.rounds):
    if aborted:
        break
    t_r = time.perf_counter()
    chunk = job.result()             # 上轮执行期间启动的推理——此刻早已就绪
    infer_ms.append(job.dur_ms)
    print(f"\n── 轮 {r} | 推理 {job.dur_ms:.0f} ms（与上轮执行重叠，零等待）──")
    if args.chunk_mode == "traj":
        cur0 = read_joints(robot, ARM_NAMES)
        traj, n_pts, final_p = build_traj(chunk, cur0)
        if n_pts == 0:
            print("⛔ 轨迹首点越护栏，停止循环")
            aborted = True
        else:
            print(f"  轨迹 {n_pts} 点 × {args.traj_dt * 1000:.0f} ms 下发...")
            t_s = time.perf_counter()
            st = robot.execute_joint_trajectory(traj, is_blocking=False)
            if r < args.rounds - 1:
                job.start(*fresh_obs())   # 推理藏进轨迹执行期（sleep 放 GIL）
            t_end = t_s + n_pts * args.traj_dt
            while time.perf_counter() < t_end:
                time.sleep(0.05)          # 主线程小睡，GIL 让给推理线程
            tss = robot.check_trajectory_execution_status([])
            step_ms.append((time.perf_counter() - t_s) * 1000)
            ach = read_joints(robot, ARM_NAMES)
            if final_p is not None:
                track_err.append(float(np.max(np.abs(ach - final_p))) * 1000)
            d_cmd = (float(np.max(np.abs(final_p - cmd_hist))) * 1000
                     if cmd_hist is not None and final_p is not None else 0.0)
            if final_p is not None:
                cmd_hist = final_p.copy()
            print(f"  轨迹完成: {st} | 执行 {step_ms[-1]:.0f} ms | "
                  f"末端偏差 {track_err[-1] if track_err else float('nan'):.1f} mrad | "
                  f"|Δcmd| {d_cmd:.1f} mrad | PVT 状态 "
                  f"{[s.name for s in tss] or '未上报'}")
            if any(s.value != 2 for s in tss):   # ≠ COMPLETED
                print("⛔ PVT 状态非 COMPLETED，停止循环")
                aborted = True
        round_ms.append((time.perf_counter() - t_r) * 1000)
        if round_ms[-1] > 1:
            print(f"  轮耗时 {round_ms[-1]:.0f} ms（重规划频率 {1000 / round_ms[-1]:.1f} Hz）")
        continue
    gate = args.settle_frac * args.delta_max   # 换目标阈值按单步尺度（到位未停透）
    # 消费配额 n_steps 步后若新块未就绪 → 用旧 chunk 剩余步"续航"追踪，
    # 轮间零站桩（推理 825ms 实测 > 滑行 539ms 的对策）；chunk 耗尽才需等待。
    # 每条指令仍受 ±BUDGET 限幅 + 漂移护栏，续航不放大行程风险
    k = 0
    while k < len(chunk):
        k_end = min(k + spc, len(chunk))   # 本条指令消费 chunk 步 [k, k_end)
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(k_end - 1, cur, BUDGET)
        drift = float(np.max(np.abs(tgt - HOME)))
        if drift > args.max_excursion:
            print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                  f"{args.max_excursion * 1000:.0f}，停止循环（机械臂留在原地）")
            aborted = True
            break
        cmd_delta = (tgt - cmd_hist) if cmd_hist is not None else np.zeros_like(tgt)
        cmd_hist = tgt.copy()
        t_s = time.perf_counter()
        if args.settle:
            # 旧步进-停走：阻塞到位再发下一条 → 速度曲线锯齿（"卡卡的"根因）
            st = robot.set_joint_positions(tgt.tolist(), joint_names=ARM_NAMES,
                                           is_blocking=True, speed_rad_s=args.speed,
                                           timeout_s=10.0)
        else:
            # 追踪式（合步）：一条指令跨 spc 步，SDK 内部平滑轮廓一次滑到位
            # ——每轮起停 1/spc 次。0.6 提速实测更抖（每点全停，冲击∝速度），
            # 平滑靠减停顿次数而非提速
            robot.set_joint_positions(tgt.tolist(), joint_names=ARM_NAMES,
                                      is_blocking=False, speed_rad_s=args.speed)
            if k == 0 and r < args.rounds - 1:
                # 下一块推理藏进本指令执行期（观测取自滑行中途，即真实当前态）
                job.start(*fresh_obs())
            # 提前换目标：剩余误差收到 switch-dist（>0）即重定向——机械臂
            # 滑行中转向、速度不过零，消指令边界的停-走；0 = 旧近停透语义。
            # 最短驻留 0.15s 防止目标太近时连环刷指令
            sw_gate = args.switch_dist if args.switch_dist > 0 else gate
            min_dwell = 0.15 if args.switch_dist > 0 else 0.0
            deadline = t_s + max(2.0, 1.5 * BUDGET / max(args.speed, 0.01))
            while True:
                ach = read_joints(robot, ARM_NAMES)
                if (time.perf_counter() - t_s >= min_dwell
                        and float(np.max(np.abs(ach - tgt))) <= sw_gate):
                    break
                if time.perf_counter() > deadline:   # 兜底（慢速时按预算放宽）
                    break
                time.sleep(0.02)
            st = "ControlStatus.SUCCESS(tracked)"
        step_ms.append((time.perf_counter() - t_s) * 1000)
        ach = read_joints(robot, ARM_NAMES)
        err = float(np.max(np.abs(ach - tgt))) * 1000
        track_err.append(err)
        tag = f"指令[步{k}-{k_end - 1}]" + ("·续航" if k >= n_steps else "")
        if k >= n_steps:
            sustain_cmds += 1
        print(f"  {tag}: |Δcmd| "
              f"{float(np.max(np.abs(cmd_delta))) * 1000:5.1f} mrad | "
              f"执行 {step_ms[-1]:5.0f} ms | 回读偏差 {err:4.1f} mrad | {st}")
        if not str(st).startswith("ControlStatus.SUCCESS"):
            print("⛔ 下发非 SUCCESS，停止循环")
            aborted = True
            break
        k = k_end
        # 配额消费完：新块就绪（或已是最后一轮）→ 立即换块零等待；否则续航
        if k >= n_steps and (r == args.rounds - 1 or job.done()):
            break
    round_ms.append((time.perf_counter() - t_r) * 1000)
    if round_ms[-1] > 1:
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
    print(f"单指令执行({spc} 步合步): {pstats(step_ms)}")
if round_ms:
    print(f"整轮(重规划周期): {pstats(round_ms)}")
if track_err:
    print(f"跟踪误差: mean {np.mean(track_err):.1f} | max {max(track_err):.1f} mrad")
    print("抖动判读：|Δcmd| 快速变号=抖动；持续同号=漂移（由护栏兜底）")
if sustain_cmds:
    print(f"续航指令: {sustain_cmds} 条（配额外消费旧 chunk 步，掩盖慢推理轮间站桩）")
print(f"最终偏离起始位: {exc:.0f} mrad（护栏 {args.max_excursion * 1000:.0f} mrad）"
      + (" ⛔ 护栏触发过" if aborted else ""))

WATCH.restore()
print("SDK 关闭中...")
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
os._exit(0)   # SDK 残留线程，干净退出
