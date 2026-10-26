#!/usr/bin/env python
"""set_joint_commands 流式回放台——把保存的推理 chunk 喂回同一条下发路径，隔离控制器变量

目的（2026-10-10 排查设计）：机械臂抖动/往复的三个嫌疑——①模型输出增量特征、
②插值倍频、③控制器（含 tfs 配速语义）——本台隔离 ②③：同一份数据、同一条
插值/限幅/突发代码路径（与 run_g1_loop_traj 逐处同构），无推理无模型，
逐变量 A/B 真机手感。遥测三线（|Δcmd|/导程/残差）格式与主脚本一致，
同一解析器可直接对比闭环日志。

数据源一：run_g1_loop_traj --dump-chunks 落盘的 chunk_*.npz（原始 (50,16) 模型输出
+BASE_ARM）。臂 14 维回放（爪维不喂）。语义与主脚本一致：绝对目标=BASE_ARM+chunk，
链式限幅从**当前实际关节**出发，块间 hold --swap-gap-ms 模拟推理落地窗。

数据源二：--dataset ~/holy/datasets/XXX --episode N —— 数据集 action 绝对位置流
（30Hz 16 维，臂 0-6=右臂 / 8-14=左臂，右臂在前；剥爪维 7/15 取臂 14 维）按 50 行
切伪块，首行=就位目标。demo 连续无推理缝，扫窗口建议 --swap-gap-ms 0；各 A/B 档
保持同一 --delta-max 才公平（demo 最快步可能超 0.05，装载时会提示）。
需 pyarrow 读取（flash_pyrt311 无 → 用 flashrt 环境跑，缺库时报错带命令）。

A/B 维度：
  --mode stream   frames-per-row 帧插值突发（=主脚本同路径；240Hz 口径）
  --mode direct   每行一次 set_joint_commands（30Hz 口径，天然配对 tfs≈33.3）
  --mode stepwait 不按节拍：每行走完（gap≤--step-done-mrad）再发下一行，逐计
                  收敛 ms。指数律预测 t=ln(g₀/ε)/k（k≈2.6/s，10-12 探针）内联
                  对照并实测 k̂；⚠ 量的是伺服阶跃响应（含指数尾巴），不是流式
                  下发的节拍下限——流式不等尾巴、步进重叠。
  --tfs-ms        0=fastest arrival（现行为）| 帧节奏≈4.2 | 行节奏≈33.3
  --interp cubic|linear|quintic / --ema-alpha / --delta-max / --frames-per-row

安全：包络钳位（joint_envelope.json；stream 钳帧 direct 钳行，同主脚本帧级钳位）、
漂移护栏（偏离起始位超限停）、q/Ctrl-C 随退（主脚本同款 QuitWatcher）、
起始慢速就位（set_joint_positions 0.2 rad/s 官方低频路径）、
--nav-suspend 导航栈搁置（同主脚本，默认关）。纯执行无推理。

用法：
  ~/holy/run.sh ~/holy/scripts/probes/replay_stream_bench.py \
      --chunks-dir logs/chunks_traj_XXXX --rounds 3 --mode stream
  # 数据集整集回放（需 pyarrow，flashrt 环境跑）：
  #   LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  #   ~/miniforge3/envs/flashrt/bin/python scripts/probes/replay_stream_bench.py \
  #   --dataset ~/holy/datasets/pick_place_balence --episode 0 --exec
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
K_PURSUIT = 2.6   # 指数追踪 v≈k·gap 的 k（10-12 单发探针定案；stepwait 自校对照）
SW_WINS_MS = (33.3, 66.7, 100.0)    # stepwait 逐窗位移实测窗（30Hz 一拍口径）
SW_THR_MRAD = (5.0, 2.0, 1.0, 0.5)  # “执行完”多档阈值（首个采样穿入时刻）

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--chunks-dir", default=None,
                help="--dump-chunks 落盘目录（与 --dataset 二选一）")
ap.add_argument("--dataset", default=None,
                help="lerobot 数据集目录：整集 action 绝对位置流按 50 行切伪块"
                     "（需 pyarrow；flash_pyrt311 无，报错带替代命令）")
ap.add_argument("--episode", type=int, default=0, help="--dataset 的 episode_index")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真机下发（不带=干跑：打印计划即退）")
ap.add_argument("--rounds", type=int, default=3, help="回放块数上限（0=全部）")
ap.add_argument("--fps", type=int, default=30, help="行节拍 Hz")
ap.add_argument("--mode", choices=["stream", "direct", "stepwait"], default="stream",
                help="stream=帧插值突发（默认，同主脚本）；"
                     "direct=每行一令（30Hz 口径）；"
                     "stepwait=走完再发（每步收敛计时）")
ap.add_argument("--frames-per-row", type=int, default=8)
ap.add_argument("--interp", choices=["linear", "cubic", "quintic"], default="cubic")
ap.add_argument("--merge", type=int, default=1,
                help="合步（等时粗化）：滤波后每 N 行取 1 个宏步目标，每宏步占 N 拍；"
                     "目标频率 fps/N、帧按拍均分 → 下发 fps×frames-per-row/N；"
                     "同一轨迹同速同墙钟（非时间压缩），EMA/链钳/连续参考保持行级语义")
ap.add_argument("--delta-max", type=float, default=0.05,
                help="链式限幅 rad/行（与主脚本同义）")
ap.add_argument("--ema-alpha", type=float, default=0.0, help="行目标 EMA（0=关）")
ap.add_argument("--tfs-ms", type=float, default=0.0,
                help="time_from_start_s 毫秒：0=fastest（默认）；"
                     "stream 帧节奏≈4.2 | direct 行节奏≈33.3")
ap.add_argument("--step-done-mrad", type=float, default=1.0,
                help="stepwait 收敛阈：max|目标-实际|≤该值算走完（默认 1 mrad；"
                     "ε 越小尾巴越长，t∝ln(g₀/ε)）")
ap.add_argument("--step-timeout-s", type=float, default=3.0,
                help="stepwait 单步超时秒（指数收敛无严格终点，防挂死）")
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
ap.add_argument("--continuous-ref", dest="continuous_ref", action="store_true",
                help="跨块连续参考：链锚/EMA 从上一块末行接续，不回臂位重锚"
                     "（数据集流本连续，块首跳变=重锚惯例产物；默认关=主脚本同构重锚）")
ap.add_argument("--nav-suspend", action="store_true",
                help="循环期 SIGSTOP 冻结导航栈七进程（同 run_g1_loop_traj；默认关）")
args = ap.parse_args()
if bool(args.chunks_dir) == bool(args.dataset):
    raise SystemExit("--chunks-dir 与 --dataset 恰好给一个")

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

# ── 数据装载 ──
# npz：chunk delta + BASE_ARM → 绝对目标（主脚本 build_rows 同一换算）；
# dataset：action 绝对位置流直切（剥爪维 7/15，右臂在前）。
def load_dataset_blocks(ds_dir, episode):
    """lerobot 数据集一集 → (N,14) 绝对臂位置流按 50 行切伪块。"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(
            "--dataset 需要 pyarrow（flash_pyrt311 没有）。两种走法：\n"
            "  ① flashrt 环境跑本台（自带 pyarrow+SDK）：\n"
            "    LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \\\n"
            "    ~/miniforge3/envs/flashrt/bin/python " + os.path.abspath(__file__) + " \\\n"
            "        --dataset " + ds_dir + " --episode " + str(episode) + " ...\n"
            "  ② 给 flash_pyrt311 装 pyarrow（⚠ 勿动 numpy==1.26.4，装时核对）")
    info = json.load(open(os.path.join(ds_dir, "meta", "info.json")))
    fps = int(info.get("fps", 30))
    pfiles = sorted(glob.glob(os.path.join(ds_dir,
                                           "data", "chunk-*", "file-*.parquet")))
    if not pfiles:
        raise SystemExit(f"数据集无 parquet：{ds_dir}")
    acts = None
    for pf in pfiles:            # episode 在文件内连续（lerobot 顺序写），命中即停
        t = pq.read_table(pf, columns=["action", "episode_index"])
        e = np.asarray(t.column("episode_index").to_pylist())
        idx = np.nonzero(e == episode)[0]
        if len(idx):
            sub = t.take(pa.array(idx))
            acts = np.asarray(sub.column("action").to_pylist(), dtype=np.float32)
            break
    if acts is None or len(acts) < 2:
        raise SystemExit(f"数据集无 episode_index={episode}（或行数 <2）")
    if acts.shape[1] != 16:
        raise SystemExit(f"action 维数 {acts.shape[1]} ≠ 16（维序契约变了？先查 info.json）")
    arm = np.concatenate([acts[:, :7], acts[:, 8:15]], axis=1)   # (N,14) 绝对
    steps = np.abs(np.diff(arm, axis=0))
    p50, p95, mx = (float(np.percentile(steps, 50)),
                    float(np.percentile(steps, 95)),
                    float(steps.max()))
    print(f"[数据集] {ds_dir} ep{episode}: {len(arm)} 行 ≈ {len(arm)/fps:.1f}s "
          f"@ {fps}Hz | 行间步长 p50 {p50*1000:.1f} / p95 {p95*1000:.1f} / "
          f"max {mx*1000:.1f} mrad")
    if mx > args.delta_max:
        print(f"[数据集] ⚠ demo 最快步 {mx*1000:.0f} > 链钳 ±{args.delta_max*1000:.0f}"
              f" mrad：最快段会被削平；忠实回放加 --delta-max {mx:.2f}"
              f"（各 A/B 档保持同值才公平）")
    if args.swap_gap_ms > 0:
        print("[数据集] 提示：demo 连续无推理缝，扫窗口建议 --swap-gap-ms 0")
    return [arm[i:i + 50].copy() for i in range(0, len(arm), 50)]


if args.dataset:
    files = None
    BLOCKS = load_dataset_blocks(args.dataset, args.episode)
else:
    files = sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.npz")))
    if not files:
        raise SystemExit(f"目录无 chunk_*.npz：{args.chunks_dir}")
    BLOCKS = []
    for f in files:
        try:
            d = np.load(f)
            ch = np.asarray(d["chunk"], dtype=np.float32)         # (50,16)
            base = np.asarray(d["base_arm"], dtype=np.float32)    # (14,)
        except Exception as e:
            print(f"[装载] ⚠ 跳过坏 npz {os.path.basename(f)}: {e}（q 掐断常留截尾件）")
            continue
        BLOCKS.append(np.concatenate([ch[:, :7], ch[:, 8:15]], axis=1)
                      + base[None, :])                        # (50,14) 绝对目标
n_total = len(BLOCKS)
n_blocks = n_total if args.rounds <= 0 else min(args.rounds, n_total)
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
    """链式限幅 + 可选 EMA（主脚本 build_rows 同构：每行 ≤±delta-max）。
    --continuous-ref：链锚/EMA 从上一块末行接续（跨块连续参考，不回臂位重锚）。
    --merge N：滤波后每 N 行取 1 个宏步目标（等时粗化，宏步占 N 拍）。"""
    global _chain_state
    rows = np.empty_like(tgt_blk)
    prev = np.asarray(cur, dtype=np.float32)
    if args.continuous_ref and _chain_state is not None:
        prev = _chain_state.copy()
    for kk in range(len(tgt_blk)):
        prev = prev + np.clip(tgt_blk[kk] - prev,
                              -args.delta_max, args.delta_max)
        rows[kk] = prev
    if args.ema_alpha > 0:
        a = float(args.ema_alpha)
        # 主脚本同构：携带状态时块首行也吃 EMA（否则连续模式块首=裸钳位步长）
        carry = args.continuous_ref and _chain_state is not None
        acc = _chain_state.copy() if carry else rows[0]
        for kk in range(0 if carry else 1, len(rows)):
            acc = a * rows[kk] + (1.0 - a) * acc
            rows[kk] = acc
    if args.merge > 1:
        # 合步：滤波后每 N 行取 1 个宏步目标（末行必保=块尾位置/链状态不动；
        # 等时粗化——宏步间墙钟在流式循环展开为 N 拍，宏步长=N 行钳位位移之和，
        # 行级 ±delta-max 语义不变）
        idx = list(range(0, len(rows), args.merge))
        if idx[-1] != len(rows) - 1:
            idx[-1] = len(rows) - 1   # 末行替换尾采样（宏步数=严格 ceil，墙钟精确）
        rows = rows[idx]
    if args.continuous_ref:
        _chain_state = rows[-1].copy()
    return rows


def read_joints():
    v = robot.get_joint_positions([], ARM_NAMES)
    return np.asarray([float(x) for x in v], dtype=np.float32)


def send_burst(frames_idx_list, frames_all):
    """帧突发：绝对截止配速（预算 tick−reserve 匀速铺满），主脚本同款。"""
    if not frames_idx_list:
        return None    # 空突发（per<merge 时帧按拍均分必有空拍）：本拍不发令只走节拍
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
      + (f" | {args.frames_per_row} 帧/行（≈{args.fps*per/args.merge:.0f}Hz {args.interp}）"
         f" | 突发预算 {TICK*1000 - args.burst_reserve_ms:.1f} ms/行"
         + (f" | 合步×{args.merge}（目标 {args.fps//args.merge}Hz 宏步、每宏步 "
            f"{args.merge} 拍、下发 ≈{args.fps*per/args.merge:.0f}Hz，块墙钟不变；"
            f"拍线|Δcmd|按下发帧流记）"
            if args.merge > 1 else "")
         if args.mode == "stream" else
         (f" | 走完再发：阈 {args.step_done_mrad:g} mrad / 超时 "
          f"{args.step_timeout_s:g}s / 阈下秒过"
          if args.mode == "stepwait" else " | 每行一令"))
      + f" | 链式限幅 ±{args.delta_max} | EMA α={args.ema_alpha:g}"
      + (" | 跨块连续参考" if args.continuous_ref else "")
      + f" | tfs {args.tfs_ms:g} ms | 块间 hold {args.swap_gap_ms:.0f} ms"
      + (f" | 源 {os.path.basename(args.dataset.rstrip('/'))} ep{args.episode}"
         if args.dataset else "")
      + f" | 回放 {n_blocks}/{n_total} 块")
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
        rows = rows[: (min(args.rows_per_block, len(rows))
                       if args.rows_per_block > 0 else len(rows))]
        d = np.abs(np.diff(np.vstack([cur[None], rows]), axis=0))
        lead0 = float(np.max(np.abs(rows[0] - cur))) * 1000
        print(f"[干跑] 块{bi+1}: 行{len(rows)} |Δcmd| mean {d[1:].mean()*1000:.1f} "
              f"p95 {np.percentile(d[1:],95)*1000:.1f} max {d[1:].max()*1000:.1f} mrad "
              f"| 首行导程 {lead0:.1f} mrad"
              + (f" | 帧 {INTERP.interpolate(np.vstack([cur[None], rows])).shape[0]}"
                 if INTERP is not None else ""))
        if args.mode == "stepwait":
            _eps = args.step_done_mrad / 1000.0
            _g = d[1:].max(axis=1)
            _act = _g[_g > _eps]
            _pred = float(np.log(_act / _eps).sum() / K_PURSUIT) if len(_act) else 0.0
            print(f"[干跑][stepwait] ≥阈步 {len(_act)}/{len(_g)}"
                  f" | 预测 Σ收敛 {_pred:.1f}s（k={K_PURSUIT}/s，"
                  f"ε={args.step_done_mrad:g} mrad；另加每步下发+读回 RTT）")
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
last_tgt = None    # 跨块延续（闭环同口径）：块首拍 |Δcmd| 才能暴露换块跳变
_prev_ach = None   # 上拍实际关节（|Δact| 臂动量遥测）
_chain_state = None  # --continuous-ref 跨块链状态（上一块末行目标）
# stepwait 计时账本
_sw_zero = _sw_to = 0
_sw_win = {w: [] for w in SW_WINS_MS}
_sw_thr = {t: [] for t in SW_THR_MRAD}
_sw_conv = []      # (步长mrad, 收敛ms, 律预测ms)
_sw_calls = []     # set_joint_commands 调用 RTT ms
_sw_k = []         # k̂ = ln(g₀ᵦ/ε)/t（绑定关节口径）
_sw_bind = {}      # 绑定关节 idx -> 次数
for bi, tgt_blk in enumerate(BLOCKS):
    cur = read_joints()
    rows = build_rows(cur, tgt_blk)
    n_rows = (len(rows) if args.rows_per_block <= 0
              else min(args.rows_per_block, len(rows)))
    rows = rows[:n_rows]
    if args.mode in ("direct", "stepwait") and ENV_LO14 is not None:
        rows = np.clip(rows, ENV_LO14, ENV_HI14)    # direct/stepwait 的"帧"=行本身
    if args.mode == "stream":
        # 换块时一次性：[cur]+行目标 样条 → 逐行分桶帧 + 帧级包络钳位（主脚本同）
        _macro_rows = rows          # 块头统计用宏步口径（merge 展开后 rows=拍级）
        frames_all = INTERP.interpolate(
            np.vstack([cur[None].astype(np.float32), rows]))
        if ENV_LO14 is not None:
            _pre = frames_all.copy()
            np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :],
                    out=frames_all)
            n_env_clip += int(np.sum(np.any(_pre != frames_all, axis=1)))
        n_macro = len(rows)
        if args.merge > 1:
            # 等时粗化：宏步占 N 拍、帧按拍均分（下发 = fps×per/N Hz，块墙钟不变）
            rows = np.repeat(rows, args.merge, axis=0)
            bursts = []
            for _m in range(n_macro):
                for _grp in np.array_split(np.arange(_m * per, (_m + 1) * per),
                                           args.merge):
                    bursts.append([int(i) for i in _grp])
        else:
            bursts = [list(range(kk * per, (kk + 1) * per)) for kk in range(n_rows)]
        n_rows = len(rows)
    else:
        frames_all = rows
        bursts = [[kk] for kk in range(n_rows)]
        _macro_rows = rows
    d = np.abs(np.diff(np.vstack([cur[None], _macro_rows]), axis=0))
    t_blk = time.perf_counter()
    _hd = (f"拍 {n_rows} = 宏步{n_macro}×{args.merge}"
           if args.merge > 1 and args.mode == "stream" else f"行 {n_rows}")
    print(f"\n── 回放块 {bi+1}/{n_blocks}（{_hd}）| "
          f"|Δcmd| mean {d[1:].mean()*1000:.1f} max {d[1:].max()*1000:.1f} mrad ──")

    if args.mode == "stepwait":
        _eps = args.step_done_mrad / 1000.0
        for kk in range(n_rows):
            cur = read_joints()
            tgt = rows[kk]
            gap0 = float(np.max(np.abs(tgt - cur)))
            drift = float(np.max(np.abs(tgt - start_pos)))
            if drift > args.max_excursion:
                print(f"⛔ 漂移护栏：关节最大偏离 {drift*1000:.0f} mrad > "
                      f"{args.max_excursion*1000:.0f}，停止（机械臂留在原地）")
                _teardown()
                os._exit(5)
            _step_m = gap0 * 1000
            last_tgt = tgt.copy()
            if _step_m <= args.step_done_mrad:
                _sw_zero += 1
                print(f"  拍{kk}: 步 {_step_m:5.2f} mrad ≤ 阈 · 秒过")
                continue
            _t0 = time.perf_counter()
            _cmds = []
            for _ji in range(14):
                _c = JointCommand()
                _c.position = float(tgt[_ji])   # 标准关节只消费 position
                _cmds.append(_c)
            _st = robot.set_joint_commands(_cmds, joint_names=ARM_NAMES,
                                           time_from_start_s=args.tfs_ms / 1000.0)
            _call_ms = 1000 * (time.perf_counter() - _t0)
            if not str(_st).startswith("ControlStatus.SUCCESS"):
                print(f"⛔ 下发非 SUCCESS（{_st}），停止")
                _teardown()
                os._exit(4)
            _pred = 1000 * float(np.log(gap0 / _eps)) / K_PURSUIT
            _gvec = np.abs(tgt - cur)
            _bind = int(np.argmax(_gvec))
            _g0b = float(_gvec[_bind])
            _tw = time.perf_counter()
            _to = False
            _tr = []                             # (t_ms, gap_max_mrad, 绑定关节已走mrad)
            while True:
                _cur2 = read_joints()
                _ts = 1000 * (time.perf_counter() - _tw)
                _gabs = np.abs(tgt - _cur2)
                _gm = float(_gabs.max())
                _tr.append((_ts, _gm * 1000, (_gvec[_bind] - _gabs[_bind]) * 1000))
                if _gm * 1000 <= args.step_done_mrad:
                    break
                if time.perf_counter() - _tw > args.step_timeout_s:
                    _to = True
                    break
                time.sleep(0.001)
            _conv = 1000 * (time.perf_counter() - _tw)
            if _to:
                _sw_to += 1
                print(f"  拍{kk}: 步 {_step_m:5.1f} mrad | 令 {_call_ms:4.1f} ms | "
                      f"⚠ 超时>{args.step_timeout_s:g}s（余 {_gm*1000:.1f} mrad）")
            else:
                _kk = float(np.log(max(_g0b, _eps) / _eps) / (_conv / 1000.0))
                _sw_k.append(_kk)
                _sw_conv.append((_step_m, _conv, _pred))
                _sw_calls.append(_call_ms)
                _sw_bind[_bind] = _sw_bind.get(_bind, 0) + 1
                _w = []
                for _win in SW_WINS_MS:          # 逐窗位移：首个采样≥窗
                    _dv = next((db for ts, _, db in _tr if ts >= _win), None)
                    if _dv is not None:
                        _sw_win[_win].append(_dv)
                    _w.append(_dv)
                for _thr in SW_THR_MRAD:         # 执行完多档：首个穿入时刻
                    if args.step_done_mrad <= _thr:
                        _tc = next((ts for ts, gm, _ in _tr if gm <= _thr), None)
                        if _tc is not None:
                            _sw_thr[_thr].append(_tc)
                _w1s = (f" | 33ms走 {_w[0]:.2f} mrad" if _w[0] is not None else "")
                print(f"  拍{kk}: 步 {_step_m:5.1f} mrad | 令 {_call_ms:4.1f} ms | "
                      f"收敛 {_conv:6.0f} ms（律预测 {_pred:5.0f}）{_w1s} | k̂ {_kk:.2f}/s")
        print(f"── 块 {bi+1} 完成（stepwait，耗时 "
              f"{time.perf_counter()-t_blk:.1f} s）──")
        continue

    _next_t = time.perf_counter()
    for kk in range(n_rows):
        _now = time.perf_counter()
        if _now < _next_t:
            time.sleep(_next_t - _now)
        _next_t += TICK
        cur = read_joints()                      # 拍首读实际（主脚本同序）
        tgt = rows[kk]
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
            _fr_s = (("· 无帧 ·" if not bursts[kk]
                      else f"{len(bursts[kk])}帧@{args.fps*per/args.merge:.0f}Hz")
                     if args.mode == "stream" else f"1令@{args.fps}Hz")
        # 拍线 |Δcmd| 口径：merge 模式记真实下发帧流——发送拍=本拍最后发送帧
        # vs 上次发送帧；静默/免发拍持有上次发送帧（Δ=0=真静默）。行目标经
        # np.repeat 重复对齐拍，按行记账会把整宏步步长全记在首拍、其余拍恒 0，
        # 静默与原地不可辨；帧流口径下 帧+0.0=真原地、无帧=静默。
        # merge=1/direct 维持行目标口径（历史可比：块首拍0=α×钳位一行 校准不破）。
        if args.mode == "stream" and args.merge > 1:
            _ref_new = (frames_all[bursts[kk][-1]]
                        if (gap0 >= GAP_EPS and bursts[kk]) else last_tgt)
        else:
            _ref_new = tgt
        cmd_delta = ((_ref_new - last_tgt) if last_tgt is not None
                     else np.zeros_like(tgt))
        d_mrad = float(np.max(np.abs(cmd_delta))) * 1000
        last_tgt = _ref_new.copy() if _ref_new is not None else None
        ach = read_joints()
        if _prev_ach is not None:
            act = float(np.max(np.abs(ach - _prev_ach))) * 1000   # 臂实际动量：0=伺服僵住（SUCCESS 但不动）
        else:
            act = 0.0
        _prev_ach = ach
        err = float(np.max(np.abs(ach - tgt))) * 1000
        print(f"  拍{kk}: |Δcmd| {d_mrad:5.1f} mrad"
              f" | 导程 {gap0*1000:5.1f} → {_fr_s}"
              f" | 动 {act:4.1f} 残差 {err:4.1f} mrad | replay")
    print(f"── 块 {bi+1} 完成，耗时 {1000*(time.perf_counter()-t_blk):.0f} ms ──")
    if bi < n_blocks - 1 and args.swap_gap_ms > 0:
        time.sleep(args.swap_gap_ms / 1000.0)

print(f"\n== 回放汇总 == 块 {n_blocks} | 免发 {n_skip} | "
      f"包络钳位帧 {n_env_clip} | 总耗时 {time.perf_counter()-t0:.1f} s")
if args.mode == "stepwait" and _sw_conv:
    _cs = [c for _, c, _ in _sw_conv]
    _top = sorted(_sw_bind.items(), key=lambda x: -x[1])[:3]
    print(f"== stepwait 汇总 == 计时步 {len(_sw_conv)} | 秒过 {_sw_zero} | "
          f"超时 {_sw_to}"
          f" | 步长 mrad p50/p95 {np.percentile([s for s, _, _ in _sw_conv],50):.1f}/"
          f"{np.percentile([s for s, _, _ in _sw_conv],95):.1f}"
          f" | 收敛 ms p50/p95/max {np.percentile(_cs,50):.0f}/"
          f"{np.percentile(_cs,95):.0f}/{max(_cs):.0f}"
          f" | 令 RTT p50 {np.percentile(_sw_calls,50):.1f} ms"
          f" | k̂ 中位 {float(np.median(_sw_k)):.2f}/s（律 2.6；"
          f"10-12 单发探针）"
          + (" | 绑定关节 " + ", ".join(f"{ARM_NAMES[i]}×{n}" for i, n in _top)
             if _top else ""))
if args.mode == "stepwait" and (_sw_win[SW_WINS_MS[0]] or _sw_thr[SW_THR_MRAD[0]]):
    _wp = " | ".join(
        (f"{w:g}ms {np.median(_sw_win[w]):.2f} mrad(n={len(_sw_win[w])})"
         if _sw_win[w] else f"{w:g}ms —") for w in SW_WINS_MS)
    print(f"== stepwait 逐窗实测 == 位移（绑定关节，首个采样≥窗）: {_wp}")
    _tp = " | ".join(
        (f"≤{t:g} {np.median(_sw_thr[t]):.0f} ms(n={len(_sw_thr[t])})"
         if _sw_thr[t] else f"≤{t:g} 未测(需 --step-done-mrad ≤{t:g})")
        for t in SW_THR_MRAD)
    print(f"== stepwait 执行完多档 == {_tp}")
elif args.mode == "stepwait" and not _sw_conv:
    print(f"== stepwait 汇总 == 计时步 0 | 秒过 {_sw_zero} | 超时 {_sw_to}"
          "（全部步长 ≤ 阈，无计时样本——可减小 --step-done-mrad）")
_teardown()
