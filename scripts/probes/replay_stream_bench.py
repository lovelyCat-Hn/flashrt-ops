#!/usr/bin/env python
"""set_joint_commands 流式回放台——把保存的推理 chunk 喂回同一条下发路径，隔离控制器变量

目的（2026-10-10 排查设计）：机械臂抖动/往复的三个嫌疑——①模型输出增量特征、
②插值倍频、③控制器（含 tfs 配速语义）——本台隔离 ②③：同一份数据、同一条
插值/限幅/突发代码路径（与 run_g1_loop_traj 逐处同构），无推理无模型，
逐变量 A/B 真机手感。遥测三线（|Δcmd|/导程/残差）格式与主脚本一致，
同一解析器可直接对比闭环日志。

数据源：run_g1_loop_traj --dump-chunks 落盘的 chunk_*.npz（原始 (50,16) 模型输出
+BASE_ARM）。臂 14 维回放（爪维不喂）。语义与主脚本一致：绝对目标=BASE_ARM+chunk，
链式限幅从**当前实际关节**出发，块间 hold --swap-gap-ms 模拟推理落地窗。

A/B 维度：
  --mode stream   frames-per-row 帧插值突发（=主脚本同路径；240Hz 口径）
  --mode direct   每行一次 set_joint_commands（30Hz 口径，天然配对 tfs≈33.3）
  --tfs-ms        0=fastest arrival（现行为）| 帧节奏≈4.2 | 行节奏≈33.3
  --interp cubic|linear|quintic / --ema-alpha / --delta-max / --frames-per-row

安全：包络钳位（joint_envelope.json；stream 钳帧 direct 钳行，同主脚本帧级钳位）、
漂移护栏（偏离起始位超限停）、q/Ctrl-C 随退（主脚本同款 QuitWatcher）、
起始慢速就位（set_joint_positions 0.2 rad/s 官方低频路径）、
--nav-suspend 导航栈搁置（同主脚本，默认关）。纯执行无推理。

用法：
  ~/holy/run.sh ~/holy/scripts/probes/replay_stream_bench.py \
      --chunks-dir logs/chunks_traj_XXXX --rounds 3 --mode stream
  # 干跑（零下发，打印计划即退）：去掉 --exec
"""
import argparse
import atexit
import glob
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "FlashRT"))
sys.path.insert(0, str(_ROOT / "scripts" / "inference"))

from galbot_sdk.g1 import GalbotRobot, JointCommand  # noqa: E402
from g1_traj_interp import create_interpolator       # noqa: E402

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT
GAP_EPS = 1e-4

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--chunks-dir", required=True, help="--dump-chunks 落盘目录")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真机下发（不带=干跑：打印计划即退）")
ap.add_argument("--rounds", type=int, default=3, help="回放块数上限（0=全部）")
ap.add_argument("--fps", type=int, default=30, help="行节拍 Hz")
ap.add_argument("--mode", choices=["stream", "direct"], default="stream",
                help="stream=帧插值突发（默认，同主脚本）；"
                     "direct=每行一令（30Hz 口径）")
ap.add_argument("--frames-per-row", type=int, default=8)
ap.add_argument("--interp", choices=["linear", "cubic", "quintic"], default="cubic")
ap.add_argument("--delta-max", type=float, default=0.05,
                help="链式限幅 rad/行（与主脚本同义）")
ap.add_argument("--ema-alpha", type=float, default=0.0, help="行目标 EMA（0=关）")
ap.add_argument("--tfs-ms", type=float, default=0.0,
                help="time_from_start_s 毫秒：0=fastest（默认）；"
                     "stream 帧节奏≈4.2 | direct 行节奏≈33.3")
ap.add_argument("--burst-reserve-ms", type=float, default=6.0,
                help="stream 突发预算=tick−该值（0=铺满整拍）")
ap.add_argument("--rows-per-block", type=int, default=0,
                help="每块消费行数（0=全部 50；35=模拟水位换块节奏）")
ap.add_argument("--swap-gap-ms", type=float, default=330.0,
                help="块间 hold 毫秒（模拟推理落地窗，主脚本节奏=330）")
ap.add_argument("--goto-start", dest="goto_start", action="store_true", default=True)
ap.add_argument("--no-goto-start", dest="goto_start", action="store_false",
                help="跳过起始慢速就位（臂已在轨迹起点附近时）")
ap.add_argument("--envelope", type=str,
                default=str(_ROOT / "models" / "pi05_g1_onlypick_deploy"
                            / "joint_envelope.json"),
                help="包络护栏 json（臂 14 维钳位）")
ap.add_argument("--max-excursion", type=float, default=3.0,
                help="偏离起始位护栏 rad（任一关节超限即停）")
ap.add_argument("--nav-suspend", action="store_true",
                help="循环期 SIGSTOP 冻结导航栈七进程（同 run_g1_loop_traj；默认关）")
args = ap.parse_args()

TICK = 1.0 / args.fps

# ── 日志 ──
_log_path = str(_ROOT / "logs" / time.strftime("replay_stream_%Y%m%d_%H%M%S.log"))
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
                pass    # teardown 已关日志文件后的残余 flush

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except ValueError:
                pass


sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)


# ── q 退出（主脚本同款：pause/resume 让确认提示符处回车可用）──
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
            r, w = os.pipe()                # 信号直通管道：C 层收信号即写字节，
            os.set_blocking(w, False)       # 主线程卡死在 SDK C++ 调用也能退出
            signal.set_wakeup_fd(w)
            signal.signal(signal.SIGINT, lambda *_: None)     # 退出走管道，别靠
            signal.signal(signal.SIGTERM, lambda *_: None)    # 会被推迟的异常
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
                    print("\n⛔ 收到中断信号 —— 立即退出"
                          "（已下发目标可能仍在限速执行）")
                    self.restore()
                    os._exit(130)
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出（已下发目标可能仍在限速执行）")
                    self.restore()
                    os._exit(2)
            except Exception:
                self.restore()
                os._exit(3)   # 监听线程死了比静默更危险：宁可误退不可失控


robot = None
WATCH = QuitWatcher()


# ── 导航栈 SIGSTOP 搁置（--nav-suspend，run_g1_loop_traj 同款移植）──
# A/B 对比要求与闭环跑同背景条件：导航栈停/不停会改变 tick 服务负载。
NAV_SUSPEND_NAMES = (
    "localization_server", "galbot_fusion_main", "service_navigation_plan",
    "galbot_vtn", "surround_cameras_capture", "swallows", "service_lidar_capture",
)
_NAV_GUARD_SRC = """
import os, sys, time, signal
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
ppid = os.getppid()
pids = [int(x) for x in sys.argv[1].split(',') if x]
while True:                      # 父进程（本脚本）消失才往下走
    time.sleep(0.3)
    try:
        os.kill(ppid, 0)
    except OSError:
        break
for p in pids:
    try:
        os.kill(p, signal.SIGCONT)   # 最后一道网：主进程任何死法都恢复导航栈
    except OSError:
        pass
"""


class NavSuspend:
    """只认领自己 STOP 的 pid（resume 只 CONT 这些），对别人冻结的不碰。"""

    def __init__(self):
        self.stopped = {}                     # pid -> name
        self._guard = None

    def _find_pids(self):
        found = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    arg0 = f.read().split(b"\x00")[0].decode("utf-8", "replace")
            except OSError:
                continue
            base = os.path.basename(arg0)
            if base in NAV_SUSPEND_NAMES:
                found[int(pid)] = base
        return found

    def suspend(self):
        cands = self._find_pids()
        missing = [n for n in NAV_SUSPEND_NAMES if n not in cands.values()]
        if missing:
            print(f"[nav-suspend] ⚠ 未找到（跳过）: {missing}")
        self._guard = subprocess.Popen(
            [sys.executable, "-c", _NAV_GUARD_SRC,
             ",".join(str(p) for p in sorted(cands))],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for pid, name in sorted(cands.items()):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    if f.read().split()[2] == "T":   # 已冻结（非我们所停）：不碰
                        continue
            except (OSError, IndexError):
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
                self.stopped[pid] = name
            except OSError as e:
                print(f"[nav-suspend] ⚠ STOP {name}({pid}) 失败: {e}")
        print(f"[nav-suspend] 已冻结 {len(self.stopped)} 进程: "
              + ", ".join(f"{n}({p})" for p, n in sorted(self.stopped.items()))
              + f" | 守护网 pid={self._guard.pid}")
        print("[nav-suspend] ⚠ 冻结期底盘勿动（localization 位姿停更）；"
              "相机 capture 不在冻结集，取图不受影响")

    def resume(self):
        if not self.stopped:
            return
        for pid in sorted(self.stopped):
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        print(f"[nav-suspend] 已恢复 {len(self.stopped)} 进程"
              "（CONT 幂等；守护网发现主进程退出后自退）")
        self.stopped.clear()


NAV = None   # suspend 后指向 NavSuspend 实例；_teardown 统一 resume


def _teardown():
    if globals().get("NAV") is not None:
        try:
            NAV.resume()
        except Exception:
            pass
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
    """任何未捕获异常：关 SDK + 还原终端 + 硬退（SDK 残留线程裸退会段错误）。"""
    print(f"\n⛔ 异常退出: {t.__name__}: {v}")
    _teardown()
    os._exit(1)


sys.excepthook = _fatal_hook

# ── 数据装载（chunk delta → 绝对目标，主脚本 build_rows 同一换算）──
files = sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.npz")))
if not files:
    raise SystemExit(f"目录无 chunk_*.npz：{args.chunks_dir}")
BLOCKS = []
for f in files:
    d = np.load(f)
    ch = np.asarray(d["chunk"], dtype=np.float32)         # (50,16)
    base = np.asarray(d["base_arm"], dtype=np.float32)    # (14,)
    BLOCKS.append(np.concatenate([ch[:, :7], ch[:, 8:15]], axis=1)
                  + base[None, :])                        # (50,14) 绝对目标
n_blocks = len(BLOCKS) if args.rounds <= 0 else min(args.rounds, len(BLOCKS))
BLOCKS = BLOCKS[:n_blocks]
per = args.frames_per_row if args.mode == "stream" else 1
INTERP = (create_interpolator(args.interp, dim=14, input_hz=args.fps,
                              output_hz=args.fps * per)
          if args.mode == "stream" else None)

# ── 包络（与主脚本同源 json；stream 钳帧，direct 钳行）──
ENV_LO14 = ENV_HI14 = None
if os.path.exists(args.envelope):
    _env = json.load(open(args.envelope))
    _inf = float("inf")
    lo = [(-_inf if v is None else float(v)) for v in _env["action_lo"]]
    hi = [(_inf if v is None else float(v)) for v in _env["action_hi"]]
    ENV_LO14 = np.array(lo[:7] + lo[8:15], dtype=np.float32)
    ENV_HI14 = np.array(hi[:7] + hi[8:15], dtype=np.float32)
    print(f"[护栏] 包络开：{args.envelope}（臂 14 维钳位）")
else:
    print(f"[护栏] ⚠ 包络文件缺失，跳过钳位：{args.envelope}")


def build_rows(cur, tgt_blk):
    """链式限幅 + 可选 EMA（主脚本 build_rows 同构：每行 ≤±delta-max）。"""
    rows = np.empty_like(tgt_blk)
    prev = np.asarray(cur, dtype=np.float32)
    for kk in range(len(tgt_blk)):
        prev = prev + np.clip(tgt_blk[kk] - prev,
                              -args.delta_max, args.delta_max)
        rows[kk] = prev
    if args.ema_alpha > 0:
        a = float(args.ema_alpha)
        acc = rows[0]
        for kk in range(1, len(rows)):
            acc = a * rows[kk] + (1.0 - a) * acc
            rows[kk] = acc
    return rows


def read_joints():
    v = robot.get_joint_positions([], ARM_NAMES)
    return np.asarray([float(x) for x in v], dtype=np.float32)


def send_burst(frames_idx_list, frames_all):
    """帧突发：绝对截止配速（预算 tick−reserve 匀速铺满），主脚本同款。"""
    fdt = (TICK - args.burst_reserve_ms / 1000.0) / len(frames_idx_list)
    _t_f = time.perf_counter()
    for _bi, _fi in enumerate(frames_idx_list):
        _fr = frames_all[_fi]
        _cmds = []
        for _ji in range(14):
            _c = JointCommand()
            _c.position = float(_fr[_ji])   # 标准关节只消费 position
            _cmds.append(_c)
        _st = robot.set_joint_commands(_cmds, joint_names=ARM_NAMES,
                                       time_from_start_s=args.tfs_ms / 1000.0)
        if not str(_st).startswith("ControlStatus.SUCCESS"):
            return _st
        _t_f += fdt
        if _bi < len(frames_idx_list) - 1:
            _sl = _t_f - time.perf_counter()
            if _sl > 0:
                time.sleep(_sl)
    return None


print(f"[replay-bench] 模式 {args.mode} | tick {TICK*1000:.1f} ms（{args.fps} Hz）"
      + (f" | {args.frames_per_row} 帧/行（≈{args.fps*per:.0f}Hz {args.interp}）"
         f" | 突发预算 {TICK*1000 - args.burst_reserve_ms:.1f} ms/行"
         if args.mode == "stream" else " | 每行一令")
      + f" | 链式限幅 ±{args.delta_max} | EMA α={args.ema_alpha:g}"
      + f" | tfs {args.tfs_ms:g} ms | 块间 hold {args.swap_gap_ms:.0f} ms"
      + f" | 回放 {n_blocks}/{len(files)} 块")
if args.mode == "direct" and args.tfs_ms <= 0:
    print("[提示] direct 口径天然配对 tfs≈33.3 ms（--tfs-ms 33.3）；"
          "当前 fastest arrival")
if args.mode == "stream" and args.frames_per_row > 1:
    _wp = np.zeros((3, 14), dtype=np.float32)
    _t0 = time.perf_counter()
    INTERP.interpolate(_wp)
    print(f"[样条] 预热完成（首次构造 {1000*(time.perf_counter()-_t0):.0f} ms 计入）")

if not args.do_exec:
    cur = np.zeros(14, dtype=np.float32)
    for bi, tgt in enumerate(BLOCKS):
        rows = build_rows(cur, tgt)
        d = np.abs(np.diff(np.vstack([cur[None], rows]), axis=0))
        lead0 = float(np.max(np.abs(rows[0] - cur))) * 1000
        print(f"[干跑] 块{bi+1}: 行{len(rows)} |Δcmd| mean {d[1:].mean()*1000:.1f} "
              f"p95 {np.percentile(d[1:],95)*1000:.1f} max {d[1:].max()*1000:.1f} mrad "
              f"| 首行导程 {lead0:.1f} mrad"
              + (f" | 帧 {INTERP.interpolate(np.vstack([cur[None], rows])).shape[0]}"
                 if INTERP is not None else ""))
        cur = rows[-1]
        if ENV_LO14 is not None:
            cur = np.clip(cur, ENV_LO14, ENV_HI14)
    print("[干跑] 零下发，计划如上。")
    sys.exit(0)

# ── 真机路径 ──
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(2)
_cst = robot.start_controller("all")
print(f"start_controller('all') → {_cst}")

if args.nav_suspend:
    NAV = NavSuspend()
    NAV.suspend()

start_pos = read_joints()
print(f"起始位: {[round(float(v), 3) for v in start_pos]}")

# 起始就位：慢速到首块行0绝对目标（官方低频路径 0.2 rad/s）
first_abs = BLOCKS[0][0]
_dist = float(np.max(np.abs(first_abs - start_pos)))
if args.goto_start and _dist > 0.02:
    print(f"[就位] 距首块行0 {_dist*1000:.0f} mrad → 0.2 rad/s 慢速移动…")
    robot.set_joint_positions([float(v) for v in first_abs],
                              [], ARM_NAMES, True, 0.2, 30.0)
    time.sleep(0.5)
elif _dist > 0.3:
    print(f"⚠ 未就位且 --no-goto-start：距首块行0 {_dist*1000:.0f} mrad（风险自担）")

WATCH.pause()
input(f"\n⚠ 将回放保存轨迹 {n_blocks} 块（无推理；急停就绪后回车开始）...")
WATCH.resume()

t0 = time.perf_counter()
n_env_clip = 0
n_skip = 0
for bi, tgt_blk in enumerate(BLOCKS):
    cur = read_joints()
    rows = build_rows(cur, tgt_blk)
    n_rows = (len(rows) if args.rows_per_block <= 0
              else min(args.rows_per_block, len(rows)))
    rows = rows[:n_rows]
    if args.mode == "direct" and ENV_LO14 is not None:
        rows = np.clip(rows, ENV_LO14, ENV_HI14)    # direct 的"帧"=行本身
    if args.mode == "stream":
        # 换块时一次性：[cur]+行目标 样条 → 逐行分桶帧 + 帧级包络钳位（主脚本同）
        frames_all = INTERP.interpolate(
            np.vstack([cur[None].astype(np.float32), rows]))
        if ENV_LO14 is not None:
            _pre = frames_all.copy()
            np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :],
                    out=frames_all)
            n_env_clip += int(np.sum(np.any(_pre != frames_all, axis=1)))
        bursts = [list(range(kk * per, (kk + 1) * per)) for kk in range(n_rows)]
    else:
        frames_all = rows
        bursts = [[kk] for kk in range(n_rows)]
    d = np.abs(np.diff(np.vstack([cur[None], rows]), axis=0))
    t_blk = time.perf_counter()
    print(f"\n── 回放块 {bi+1}/{n_blocks}（行 {n_rows}）| "
          f"|Δcmd| mean {d[1:].mean()*1000:.1f} max {d[1:].max()*1000:.1f} mrad ──")

    last_tgt = None
    _next_t = time.perf_counter()
    for kk in range(n_rows):
        _now = time.perf_counter()
        if _now < _next_t:
            time.sleep(_next_t - _now)
        _next_t += TICK
        cur = read_joints()                      # 拍首读实际（主脚本同序）
        tgt = rows[kk]
        cmd_delta = ((tgt - last_tgt) if last_tgt is not None
                     else np.zeros_like(tgt))
        d_mrad = float(np.max(np.abs(cmd_delta))) * 1000
        gap0 = float(np.max(np.abs(tgt - cur)))
        drift = float(np.max(np.abs(tgt - start_pos)))
        if drift > args.max_excursion:
            print(f"⛔ 漂移护栏：关节最大偏离 {drift*1000:.0f} mrad > "
                  f"{args.max_excursion*1000:.0f}，停止（机械臂留在原地）")
            _teardown()
            os._exit(5)
        if gap0 < GAP_EPS:
            n_skip += 1
            _fr_s = "· 免发 ·"
        else:
            if args.do_exec:
                _st = send_burst(bursts[kk], frames_all)
                if _st is not None:
                    print(f"⛔ 下发非 SUCCESS（{_st}），停止")
                    _teardown()
                    os._exit(4)
            _fr_s = (f"{len(bursts[kk])}帧@{args.fps*per:.0f}Hz"
                     if args.mode == "stream" else f"1令@{args.fps}Hz")
        last_tgt = tgt.copy()
        ach = read_joints()
        err = float(np.max(np.abs(ach - tgt))) * 1000
        print(f"  拍{kk}: |Δcmd| {d_mrad:5.1f} mrad"
              f" | 导程 {gap0*1000:5.1f} → {_fr_s}"
              f" | 残差 {err:4.1f} mrad | replay")
    print(f"── 块 {bi+1} 完成，耗时 {1000*(time.perf_counter()-t_blk):.0f} ms ──")
    if bi < n_blocks - 1 and args.swap_gap_ms > 0:
        time.sleep(args.swap_gap_ms / 1000.0)

print(f"\n== 回放汇总 == 块 {n_blocks} | 免发 {n_skip} | "
      f"包络钳位帧 {n_env_clip} | 总耗时 {time.perf_counter()-t0:.1f} s")
_teardown()
