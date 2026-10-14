#!/usr/bin/env python
"""单发大步长速度探针——钉死臂速天花板是"每窗规划压制"还是"硬速度帽"。

背景（2026-10-12）：30Hz set_joint_commands 位置流（tfs=0）实测臂巡航只有
~0.5-0.8 rad/s（分箱：步长 20-30 mrad/拍时臂动 14，30+ 压平台），计划速度
跟不上。头文件警告"large jumps may cause excessively fast motion"暗示伺服
单发大目标时能跑很快——若是，则天花板是流式 regimen 的逐窗规划压制而非硬帽。

判别设计（同距离 A/B，同一跑内完成）：
  单发腿：一条 set_joint_commands(position=cur+d, tfs=0) → 高频回读采样到
          收敛，测到达时间 + 峰值关节速度 + 全程速度廓线（落盘 npz）
  流式腿：同距离走 30Hz 链限幅流（delta-max 0.05，回放/闭环同款）→ 同测
判读：单发峰值速度 >> 0.8 rad/s（且到达时间 ≫ 短于流式）→ 每窗规划压制
      （流式可以救：微流/批接口有戏）；单发也 ~0.6-0.8 → 硬速度帽（执行器
      地板，只能训练侧/任务侧绕）。

⚠ 加 --exec 会真实驱动双臂！安全：干跑默认零下发；目标逐关节包络钳位；
   漂移护栏；每腿出发-返回净位移≈0；q/Ctrl-C 随退；结束 0.2 rad/s 官方路径
   回起始位；物理急停第一优先级。

用法:
  ~/holy/run.sh ~/holy/scripts/test/single_shot_speed_probe.py            # 干跑
  ~/holy/run.sh ~/holy/scripts/test/single_shot_speed_probe.py \
      --steps 0.05,0.1,0.2,0.3 --exec
"""
import argparse
import atexit
import json
import os
import pathlib
import select
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "FlashRT"))

from galbot_sdk.g1 import GalbotRobot, JointCommand  # noqa: E402

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真机下发（不带=干跑：打印计划即退）")
ap.add_argument("--steps", type=str, default="0.05,0.1,0.2,0.3",
                help="逗号分隔的单发步长 rad 序列（每步长：出发+返回两腿均测量）")
ap.add_argument("--stream-contrast", type=float, default=0.2,
                help="流式对照腿的距离 rad（0=不做；30Hz 链限幅 0.05 同款回放口径）")
ap.add_argument("--poll-hz", type=int, default=100, help="到达监测回读频率 Hz")
ap.add_argument("--tol", type=float, default=0.005,
                help="收敛判据 rad（最大关节距，连续 8 拍稳定）")
ap.add_argument("--timeout", type=float, default=6.0, help="单腿到达超时 s")
ap.add_argument("--delta-max", type=float, default=0.05,
                help="流式对照腿链限幅 rad/拍（在役值）")
ap.add_argument("--max-excursion", type=float, default=3.0,
                help="偏离起始位护栏 rad")
ap.add_argument("--envelope", type=str,
                default=str(_ROOT / "models" / "pi05_g1_place_deploy"
                            / "joint_envelope.json"),
                help="包络护栏 json（臂 14 维钳位；本探针无 chunk，默认 place 盒）")
args = ap.parse_args()

# ── 日志全文落盘 ──
_log_path = str(_ROOT / "logs" / time.strftime("singleshot_probe_%Y%m%d_%H%M%S.log"))
pathlib.Path(_log_path).parent.mkdir(parents=True, exist_ok=True)
_log_fh = open(_log_path, "a", buffering=1, encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except ValueError:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except ValueError:
                pass


sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)
print(f"[log] 全文日志 → {_log_path}")


class QuitWatcher:
    """后台线程监听键盘 q：任何阶段即时退出（SDK 阻塞 C++ 调用吞 Ctrl-C）。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._old = None
        self._wake_r = None
        self.active = threading.Event()
        if os.isatty(self.fd):
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.restore)
            r, w = os.pipe()
            os.set_blocking(w, False)
            signal.set_wakeup_fd(w)
            signal.signal(signal.SIGINT, lambda *_: None)
            signal.signal(signal.SIGTERM, lambda *_: None)
            self._wake_r = r
            self.active.set()
            threading.Thread(target=self._loop, daemon=True).start()
            print("（循环期间随时按 q 或 Ctrl-C 退出；确认提示符处用回车；急停第一优先级）")

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
            try:
                fds = [self._wake_r] if self._wake_r is not None else []
                if self.active.is_set():
                    fds.append(sys.stdin)
                if not fds or not select.select(fds, [], [], 0.2)[0]:
                    continue
                if (self._wake_r is not None
                        and select.select([self._wake_r], [], [], 0)[0]):
                    os.read(self._wake_r, 1)
                    print("\n⛔ 收到中断信号 —— 立即退出")
                    self.restore()
                    os._exit(130)
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出")
                    self.restore()
                    os._exit(2)
            except Exception:
                self.restore()
                os._exit(3)


robot = None
WATCH = QuitWatcher()

# 采样落盘目录（速度廓线离线分析用）
_out_dir = pathlib.Path(_log_path).with_name(
    _log_path.split("/")[-1].replace("singleshot_probe", "singleshot_data"))
_out_dir.mkdir(parents=True, exist_ok=True)


def _teardown():
    rob = globals().get("robot")
    if rob is not None:
        try:
            rob.request_shutdown()
            rob.wait_for_shutdown()
            rob.destroy()
        except Exception:
            pass
    WATCH.restore()
    try:
        _log_fh.close()
    except Exception:
        pass


def _fatal_hook(t, v, tb):
    print(f"\n⛔ 异常退出: {t.__name__}: {v}")
    _teardown()
    os._exit(1)


sys.excepthook = _fatal_hook

# ── 包络 ──
ENV_LO14 = ENV_HI14 = None
if os.path.exists(args.envelope):
    with open(args.envelope) as f:
        _e = json.load(f)
    _inf = float("inf")
    lo = [(-_inf if v is None else float(v)) for v in _e["action_lo"]]
    hi = [(_inf if v is None else float(v)) for v in _e["action_hi"]]
    ENV_LO14 = np.array(lo[:7] + lo[8:15], dtype=np.float32)
    ENV_HI14 = np.array(hi[:7] + hi[8:15], dtype=np.float32)
    print(f"[护栏] 包络开：{args.envelope}（目标逐关节钳位）")
else:
    print(f"[护栏] ⚠ 包络文件缺失：{args.envelope}（无钳位，风险自担）")

STEPS = [float(x) for x in args.steps.split(",") if x.strip()]
print(f"[plan] 步长序列 {STEPS} rad | 流式对照 {args.stream_contrast} rad"
      f" | 收敛 {args.tol * 1000:.0f} mrad×8 拍 | 超时 {args.timeout}s"
      f" | 回读 {args.poll_hz} Hz")

if not args.do_exec:
    print("\n== [干跑] 零下发。真机将执行：==")
    print(f"  每步长 d∈{STEPS}: 单发 cur+d（包络钳位）→ 测到达 → 单发返回 → 测到达")
    if args.stream_contrast > 0:
        print(f"  流式对照：距起始 {args.stream_contrast} rad，30Hz 链限幅流到达，同测")
    print("  结束后 0.2 rad/s 官方路径回起始位。加 --exec 执行。")
    sys.exit(0)


def read_joints():
    v = robot.get_joint_positions([], ARM_NAMES)
    if not v or len(v) != len(ARM_NAMES):
        raise SystemExit(f"关节读取失败（返回 {len(v) if v else 0} 维）")
    return np.asarray([float(x) for x in v], dtype=np.float32)


def clip_env(tgt):
    if ENV_LO14 is None:
        return tgt
    return np.clip(tgt, ENV_LO14, ENV_HI14)


def send_oneshot(tgt):
    _cmds = []
    for _ji in range(14):
        _c = JointCommand()
        _c.position = float(tgt[_ji])
        _cmds.append(_c)
    return robot.set_joint_commands(_cmds, joint_names=ARM_NAMES,
                                    time_from_start_s=0.0)


def run_leg(tag, tgt):
    """下发 + 高频回读到收敛。返回 (到达 s, 峰值关节速度 rad/s, 采样数)。"""
    t0 = time.perf_counter()
    _st = send_oneshot(tgt)
    if not str(_st).startswith("ControlStatus.SUCCESS"):
        print(f"⛔ [{tag}] 下发非 SUCCESS（{_st}），终止")
        _teardown()
        os._exit(4)
    dt = 1.0 / args.poll_hz
    samples = [(time.perf_counter(), read_joints())]
    stable = 0
    arrive = None
    while time.perf_counter() - t0 < args.timeout:
        time.sleep(dt)
        cur = read_joints()
        now = time.perf_counter()
        samples.append((now, cur))
        gap = float(np.max(np.abs(cur - tgt)))
        if gap < args.tol:
            stable += 1
            if stable >= 8:
                arrive = now - t0
                break
        else:
            stable = 0
        drift = float(np.max(np.abs(cur - START)))
        if drift > args.max_excursion:
            print(f"⛔ [{tag}] 漂移护栏 {drift * 1000:.0f} mrad 超限，终止")
            _teardown()
            os._exit(5)
    if arrive is None:
        arrive = float("nan")
        print(f"  [{tag}] ⚠ 超时未收敛（末gap "
              f"{float(np.max(np.abs(samples[-1][1] - tgt))) * 1000:.1f} mrad）")
    # 峰值关节速度（有限差分；SDK 读延迟使其为下界估计）
    peak_v = 0.0
    for i in range(1, len(samples)):
        dts = samples[i][0] - samples[i - 1][0]
        if dts > 1e-4:
            v = float(np.max(np.abs(samples[i][1] - samples[i - 1][1]))) / dts
            peak_v = max(peak_v, v)
    np.savez(str(_out_dir / f"{tag}.npz"),
             t=np.array([s[0] - t0 for s in samples]),
             q=np.stack([s[1] for s in samples]).astype(np.float32),
             tgt=tgt.astype(np.float32), arrive=np.float64(arrive),
             peak_v=np.float64(peak_v))
    print(f"  [{tag}] 到达 {arrive * 1000:6.0f} ms | 峰值关节速 {peak_v:5.2f} rad/s"
          f" | 采样 {len(samples)}")
    return arrive, peak_v, len(samples)


def run_stream_leg(tag, tgt):
    """流式对照：30Hz 链限幅直发到目标（回放/闭环同款口径）。"""
    t0 = time.perf_counter()
    dt = 1.0 / 30
    _next_t = t0
    samples = []
    stable = 0
    arrive = None
    while time.perf_counter() - t0 < args.timeout:
        now = time.perf_counter()
        if now < _next_t:
            time.sleep(_next_t - now)
        _next_t += dt
        cur = read_joints()
        step = np.clip(tgt - cur, -args.delta_max, args.delta_max)
        if float(np.max(np.abs(step))) < 1e-4 and float(
                np.max(np.abs(cur - tgt))) < args.tol:
            arrive = time.perf_counter() - t0
            samples.append((time.perf_counter(), cur))
            break
        _st = send_oneshot(cur + step)
        if not str(_st).startswith("ControlStatus.SUCCESS"):
            print(f"⛔ [{tag}] 流式下发非 SUCCESS（{_st}），终止")
            _teardown()
            os._exit(4)
        samples.append((time.perf_counter(), cur))
        drift = float(np.max(np.abs(cur - START)))
        if drift > args.max_excursion:
            print(f"⛔ [{tag}] 漂移护栏超限，终止")
            _teardown()
            os._exit(5)
    peak_v = 0.0
    for i in range(1, len(samples)):
        dts = samples[i][0] - samples[i - 1][0]
        if dts > 1e-4:
            v = float(np.max(np.abs(samples[i][1] - samples[i - 1][1]))) / dts
            peak_v = max(peak_v, v)
    np.savez(str(_out_dir / f"{tag}.npz"),
             t=np.array([s[0] - t0 for s in samples]),
             q=np.stack([s[1] for s in samples]).astype(np.float32),
             tgt=tgt.astype(np.float32), arrive=np.float64(arrive or float("nan")),
             peak_v=np.float64(peak_v))
    print(f"  [{tag}] 到达 {(arrive or float('nan')) * 1000:6.0f} ms"
          f" | 峰值关节速 {peak_v:5.2f} rad/s | 采样 {len(samples)}")
    return arrive, peak_v


print("\n== SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(2)
_cst = robot.start_controller("all")
print(f"start_controller('all') → {_cst}")
if not str(_cst).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止")

START = read_joints()
print(f"起始位: {[round(float(v), 3) for v in START]}")

WATCH.pause()
input(f"\n⚠ 将执行 {len(STEPS)}×2 单发腿"
      + (" + 1×流式对照腿" if args.stream_contrast > 0 else "")
      + f"（每腿出发即返回，净位移≈0；急停就绪后回车开始）...")
WATCH.resume()

results = []   # (标签, 距离, 到达 s, 峰值 v)
for d in STEPS:
    tgt_out = clip_env(START + d)
    tgt_back = clip_env(START)
    eff = float(np.max(np.abs(tgt_out - START)))
    if eff < 0.3 * d:
        print(f"\n-- 步长 {d}: 包络内有效位移仅 {eff * 1000:.0f} mrad（<30%），跳过")
        continue
    print(f"\n── 步长 {d} rad（包络后有效 {eff * 1000:.0f} mrad）──")
    a1, v1, _ = run_leg(f"out_d{int(d * 1000):03d}", tgt_out)
    results.append((f"单发出 {d}", eff, a1, v1))
    _back_from = read_joints()          # 返程实际位移按出发时位置量
    a2, v2, _ = run_leg(f"back_d{int(d * 1000):03d}", tgt_back)
    results.append((f"单发返 {d}",
                    float(np.max(np.abs(tgt_back - _back_from))), a2, v2))

if args.stream_contrast > 0:
    d = args.stream_contrast
    tgt = clip_env(START + d)
    eff = float(np.max(np.abs(tgt - START)))
    if eff >= 0.3 * d:
        print(f"\n── 流式对照 {d} rad（30Hz 链限幅 ±{args.delta_max}）──")
        a, v = run_stream_leg("stream", tgt)
        results.append((f"流式 {d}", eff, a, v))

print(f"\n== 汇总（判定基准：流式巡航天花板 ~0.5-0.8 rad/s）==")
for tag, d, a, v in results:
    print(f"  {tag:<12} 距离 {d * 1000:5.0f} mrad | 到达 {a * 1000:6.0f} ms"
          f" | 峰值 {v:5.2f} rad/s | 等效均速 {(d / a if a and a == a else 0):5.2f} rad/s")
oneshot_v = [v for tag, _, _, v in results if tag.startswith("单发") and v == v]
if oneshot_v:
    best = max(oneshot_v)
    print(f"\n单发腿峰值速度最高 {best:.2f} rad/s → "
          + ("远超流式天花板 0.8 → 【每窗规划压制】，流式侧有救（微流/批接口）"
             if best > 1.2 else
             ("与流式天花板同量级 → 【硬速度帽】，执行器地板" if best < 1.0
              else "临界（1.0-1.2）→ 建议加大 --steps 再测一档")))
print(f"\n速度廓线数据 → {_out_dir}/（npz：t/q/tgt，可离线画 profile）")

# 回起始位（官方低频路径）
_gst = robot.set_joint_positions([float(v) for v in START], [], ARM_NAMES,
                                 True, 0.2, 30.0)
print(f"回起始位 → {_gst}")
_teardown()
os._exit(0)
