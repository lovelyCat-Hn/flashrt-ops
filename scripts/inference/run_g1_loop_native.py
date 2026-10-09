#!/usr/bin/env python
"""G1 双臂闭环——lerobot 式固定节拍执行器（v3 配速律的 A/B 对照脚本）。

与 run_g1_loop.py（v3）同一条推理链路、同一套护栏，只换执行侧调度：
  v3：pace 窗口配速律（速度=导程÷(div×窗口)，1.59× 数据集原速，指令数≈轮数）
  本脚本：固定节拍消费（每 tick 发 1 步，30fps=训练控制率），速度由
     Δaction/tick 自然涌现（≈1.0× 数据集原速）；消费到 --n-action-steps
     水位即触发新推理，新块落地**整块换入、弃尾**（lerobot
     chunk_size_threshold=0.5 同语义——v3 的 n_action_steps 在这里变成
     水位阈值而非消费配额）
调度约定出处（机制参照，非逐行移植；Apache-2.0）：
  lerobot async_inference/robot_client.py L477-489 control_loop 固定节拍
  （environment_dt 截止，sleep(max(0, dt−elapsed))）；L410-413
  _ready_to_send_observation 水位换块（qsize/chunk_size ≤ 0.5）。
时序铁律：绝对截止（next_t += TICK），落后超 1 拍即重锚——绝不连发追赶
  （追发=抖动源）；星饿（块耗尽且推理未归）=不下发（刚性伺服保持最后
  目标原地等待），超过 --starve-limit 判推理挂死安全停机。
RTC：--rtc-horizon N > 0 时启用推理侧前缀引导（FlashRT 引擎 guided RTC，
  lerobot policies/rtc 移植）：旧块未消费尾段重锚进新块坐标系作为前缀，
  每个去噪步把新块开头拉向旧计划，消换块接缝跳变。0=关（默认）。
参数命名对 lerobot 术语的对照：docs/lerobot-alignment.md。

⚠ 加 --exec 会连续真实驱动双臂！安全设计（与 v3 同款）：
  1. 只动双臂 14 关节；腿/头维度永不下发；夹爪仅 --grip 显式开启时下发
     （0~100% → manifest 标定宽度，变化超 --grip-chg 才发，非阻塞不等反馈）
  2. 每 tick 目标 = 当前读数朝 chunk 第 k 步绝对构型限幅 --delta-max rad
     （逐拍限幅；速度 = clip(导程÷tick, 0.02, --speed)）
  3. 漂移护栏：任一关节偏离起始位超 --max-excursion → 立即停止循环
  4. q 键即时退出（后台监听线程，SDK 阻塞中也可退）；物理急停第一优先级
  5. 执行前回车确认

每块遥测：推理 ms / 取图 ms / tick 服务时长与抖动 / 星饿窗 / 免发拍 /
接缝尖峰（换块首拍 |Δcmd| vs 块内均值——RTC 收益主指标）/
汇总分位数（LatencyTracker，lerobot 同款 API）。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/run_g1_loop_native.py \
      [--ckpt ~/holy/models/pi05_g1_onlypick_deploy] [--exec] \
      [--rounds 60] [--fps 30] [--speed 2.0] [--rtc-horizon 10]
"""
import argparse
import atexit
from collections import deque
import functools
import g1_config  # noqa: E402  同目录共享配置（CLI > config/g1.toml > 内置默认）
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

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--prompt", default=None, help="任务指令（默认 config [run].prompt，须用训练原句）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实连续驱动双臂（默认干跑只打印首块 5 拍目标；不进配置，仅 CLI）")
ap.add_argument("--rounds", type=int, default=None,
                help="换块次数（每块消费到水位即换，弃尾；config [loop].rounds）")
ap.add_argument("--n-action-steps", type=int, default=None,
                help="预取水位阈值：本块消费到第 K 步即触发新推理（lerobot "
                     "chunk_size_threshold 语义；≤ --chunk-size；"
                     "config [loop].n_action_steps；v3 里它是消费配额，这里只定换块时机）")
ap.add_argument("--chunk-size", type=int, default=50,
                help="chunk 长度（lerobot chunk_size），默认 50=训练原生长度"
                     "（env FLASH_RT_PI05_ACTION_CHUNK_SIZE 自动设）")
ap.add_argument("--delta-max", type=float, default=None,
                help="每 tick 目标限幅 rad（config [loop].delta_max）")
ap.add_argument("--fps", type=int, default=30,
                help="固定节拍频率 Hz（默认 30=训练控制率；>40 拒绝：tick 低于"
                     "取图+SDK 硬开销 ~25ms 必丢拍）")
ap.add_argument("--speed", type=float, default=2.0,
                help="伺服速度安全天花板 rad/s（默认 2.0；不读 config [loop].speed"
                     "——那是 v3 配速律的参数；本脚本速度由 Δaction/tick 涌现，"
                     "该上限只在模型瞬发大步时兜底）")
ap.add_argument("--starve-limit", type=float, default=2.0,
                help="星饿安全停机阈值 s：块耗尽且推理未归超过该时长即停循环"
                     "（默认 2.0s=正常推理余量 ~6×；星饿期不下发=伺服保持）")
ap.add_argument("--rtc-horizon", type=int, default=0,
                help="推理侧 RTC 前缀引导窗口（行数）：>0 启用（lerobot 默认 10），"
                     "0=关（默认）。前缀=旧块未消费尾段重锚进新块坐标系")
ap.add_argument("--rtc-max-w", type=float, default=10.0,
                help="RTC 最大引导权重（lerobot max_guidance_weight 默认 10）")
ap.add_argument("--cache-frames", type=int, default=None,
                help="K/V 时序复用周期：1=每帧全量（默认，无损）；config [loop].cache_frames")
ap.add_argument("--max-excursion", type=float, default=None,
                help="偏离起始位护栏 rad（任一关节超限即停；config [loop].max_excursion）")
ap.add_argument("--grip", action="store_true", default=None,
                help="启用夹爪下发（dim7/dim15 0~100%% → manifest 标定宽度；config [gripper].enabled）")
ap.add_argument("--no-grip", action="store_true",
                help="显式关闭夹爪下发（覆盖 config 的 gripper.enabled=true）")
ap.add_argument("--grip-speed", type=float, default=None, help="夹爪速度 m/s（config [gripper].speed）")
ap.add_argument("--grip-effort", type=float, default=None, help="夹爪力矩 N（config [gripper].effort）")
ap.add_argument("--grip-chg", type=float, default=None,
                help="夹爪下发变化阈值 %%（config [gripper].chg）")
ap.add_argument("--config", default=g1_config.DEFAULT_PATH,
                help="配置文件路径（优先级 CLI > config > 内置默认）")
ap.add_argument("--log-file", default=None,
                help="日志文件路径（默认 <仓库>/logs/loop_native_<时间戳>.log；终端照常显示，"
                     "全文同步写文件，判读免复制）")
ap.add_argument("--grip-state-cmd", action="store_true",
                help="state 夹爪维(dim7/15)改喂【指令值】而非 SDK 回读：回读滞后 "
                     "~6.3-8s 而 pick 关键窗口 ~7s——喂回读=模型全程看到冻结在起始值"
                     "的夹爪状态（数据集 state 跟随指令仅 ~0.3-1s，10-05 判读）。"
                     "pick 闭环必带；需 --grip；默认关=喂回读")
ap.add_argument("--envelope-margin", type=float, default=0.15,
                help="关节包络护栏余量，rad（默认 0.15）。"
                     "包络=<ckpt>/joint_envelope.json"
                     "（任务数据集臂关节逐维 [min,max]）：指令越界=模型在幻想训练分布"
                     "里不存在的构型——往桌下伸/顶桌即此类（刚性模式压桌→fault）")
ap.add_argument("--envelope-abort-n", type=int, default=5,
                help="连续 N 块指令被包络截断即停循环（默认 5）："
                     "持续越界=感知漂移，继续跑只会反复撞桌")
ap.add_argument("--no-envelope-guard", action="store_true",
                help="关闭关节包络护栏（排障对照用）")
ap.add_argument("--nav-suspend", action="store_true",
                help="闭环期间 SIGSTOP 搁置导航栈 7 进程（localization/fusion/"
                     "navigation_plan/vtn/surround/swallows/lidar_capture）："
                     "2026-10-07 实测推理 375.3→320.5ms（−54.8ms）、GR3D 50→7。"
                     "退出自动 CONT（含 kill -9，守护网兜底）。"
                     "⚠ 只用于臂上任务+底盘不动窗口：冻结期 localization 位姿停更，"
                     "底盘先恢复导航。默认关")
args = ap.parse_args()

if args.fps > 40:
    raise SystemExit(f"--fps {args.fps} > 40 拒绝：tick {1000 / args.fps:.1f}ms 低于"
                     "取图+SDK 硬开销 ~25ms，必然连续丢拍")
if args.rtc_horizon < 0:
    raise SystemExit("--rtc-horizon 须 ≥ 0（0=关）")
TICK = 1.0 / args.fps
V_MIN = 0.02          # 伺服速度下限 rad/s（驻停缓爬）
GAP_EPS = 1e-4        # 导程低于该值免发（伺服保持）

# ── 全文日志落盘：q 退出走 os._exit 不刷缓冲（09-23 坑），故逐行强制 flush ──
_log_path = args.log_file or str(
    pathlib.Path(__file__).resolve().parents[2] / "logs" /
    time.strftime("loop_native_%Y%m%d_%H%M%S.log"))
pathlib.Path(_log_path).parent.mkdir(parents=True, exist_ok=True)


class _Tee:
    """stdout/stderr 双写：硬拷贝到日志文件（crash traceback 也收）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


_log_fh = open(_log_path, "a", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)
print(f"[log] 全文日志 → {_log_path}")
g1_config.apply(args, {
    "ckpt": ("run", "ckpt"),
    "prompt": ("run", "prompt"),
    "rounds": ("loop", "rounds"),
    "n_action_steps": ("loop", "n_action_steps"),
    "delta_max": ("loop", "delta_max"),
    "cache_frames": ("loop", "cache_frames"),
    "max_excursion": ("loop", "max_excursion"),
    "grip": ("gripper", "enabled"),
    "grip_speed": ("gripper", "speed"),
    "grip_effort": ("gripper", "effort"),
    "grip_chg": ("gripper", "chg"),
})
if args.no_grip:
    args.grip = False
print(f"[loop-native] tick {TICK * 1000:.1f} ms（{args.fps} Hz）| 水位 {args.n_action_steps}"
      f" 步换块弃尾 | 星饿上限 {args.starve_limit}s"
      + (f" | RTC 引导 horizon {args.rtc_horizon}（max_w {args.rtc_max_w}）"
         if args.rtc_horizon > 0 else " | RTC 关"))


# ── 导航栈 SIGSTOP 搁置（--nav-suspend 开启，默认关）──
# 机制：kill -STOP 进程原地冻结——无 exit 事件，launcher 的退出拉起逻辑不触发
# （2026-10-07 vtn 8s 探针实证：冻结期 launcher 日志零增量、兄弟全活），
# CONT 原地复活，不重初始化不重载图。约束：只用于「臂上任务+底盘不动」窗口。
# 恢复多重保险：① 各退出路径显式 resume（q/Ctrl-C/异常/正常完成）；
# ② 独立守护进程盯父进程，父进程任何死法（含 kill -9/段错误）即 CONT 全部。
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
    """导航栈搁置器：只停 NAV_SUSPEND_NAMES 内、且当前不在冻结态的进程。

    只认领自己 STOP 的 pid（resume 时只 CONT 这些），对别人冻结的进程不碰。
    """

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
        # 守护网先起（继承 pid 表），再下发 STOP——任何死法都有人收尾
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


NAV = None   # suspend 后指向 NavSuspend 实例；各退出路径 None 守卫调用 resume


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
                    if NAV is not None:
                        NAV.resume()
                    os._exit(130)
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出（已下发目标可能仍在限速执行）")
                    self.restore()
                    if NAV is not None:
                        NAV.resume()
                    os._exit(2)
            except Exception:
                self.restore()
                if NAV is not None:
                    NAV.resume()
                os._exit(3)   # 监听线程死了比静默更危险：宁可误退不可失控


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
    if NAV is not None:
        NAV.resume()
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

# ── 关节包络护栏（压桌防护，2026-10-06）：任务数据集臂关节合法构型盒。SDK 侧
# 臂端 fault/堵钻不回传（指令恒 SUCCESS，10-06 实录），刚性伺服撞桌只会硬压到
# 进 fault——唯一的软防线是在指令侧拦住"训练分布外的构型"。Motion.init 脱机
# 挂死（实测 >120s），FK 路线不可用，故用数据集包络做几何代理。
# 比对对象=BASE_ARM+delta 绝对构型（chunk_abs_block，2026-10-07 修范畴错误）。
ENV_LO = ENV_HI = None
_env_file = CKPT / "joint_envelope.json"
if args.no_envelope_guard:
    print("[护栏] 关节包络护栏已 --no-envelope-guard 关闭")
elif _env_file.exists():
    _env = json.loads(_env_file.read_text())
    _inf = float("inf")
    ENV_LO = [(-_inf if v is None else float(v)) for v in _env["action_lo"]]
    ENV_HI = [(_inf if v is None else float(v)) for v in _env["action_hi"]]
    print(f"[护栏] 关节包络开：{_env_file.name}（来源 {_env.get('source', '?')}；"
          f"余量 ±{args.envelope_margin} rad；越界=截断 chunk，连续 {args.envelope_abort_n} "
          f"块→停）")
else:
    print(f"[护栏] 未找到 {_env_file}——关节包络护栏关闭（当前 ckpt 无任务包络）")

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
    if args.grip_state_cmd:
        print("⚠ state 夹爪维(7/15)喂【指令值】（--grip-state-cmd）：SDK 回读滞后 "
              "~6.3-8s > pick 窗口 ~7s，回读=冻结起始值（10-05 四跑实证）")

# 2026-09-23 tf_matrix 实证：INT8 两档（全 INT8 / 仅编码器）均毁动作质量
# （块均 cos 0.15/0.27 vs bf16 0.98）——定档 bf16，显式锁定（详见 BENCHMARKS 附录）
os.environ.setdefault("FVK_PI05_RTX_FORCE_BF16", "1")
# 图开关：0=开图（默认，2026-09-29 WithFlags 热修已打、9/30 bench 图模式稳定，
# 收益 ~3%）；回退 eager：前缀 PI05_NO_GRAPH=1
os.environ.setdefault("PI05_NO_GRAPH", "0")
# state 以十进制文本拼进 prompt（format_pi05_prompt）：关节值一漂、bin 数位
# 变化 → token 数变；默认 exact 模式每种长度一条 pipeline，换长=整条重建+
# 重 autotune（~800ms，2026-09-23 合成实验实锤）。fixed=定长 200 一条
# pipeline 只换 embeds，实测含 state 切换恒定 240-253ms
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
# chunk 长度：pi05_rtx 前端模块导入时读此 env，必须在下面 import flash_rt 前定死
os.environ["FLASH_RT_PI05_ACTION_CHUNK_SIZE"] = str(args.chunk_size)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

# 包络护栏数组化+余量（解析在上方 import 之前完成；margin 语义同 10-06 原版）
if ENV_LO is not None:
    _m = args.envelope_margin
    ENV_LO = np.array(ENV_LO, dtype=np.float32)
    ENV_HI = np.array(ENV_HI, dtype=np.float32)
    ENV_LO[:7] -= _m
    ENV_LO[8:15] -= _m
    ENV_HI[:7] += _m
    ENV_HI[8:15] += _m
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

# 回退 eager 时才封图（PI05_NO_GRAPH=1）；默认开图（scripts/test/bench_pi05.py 同款）
if os.environ.get("PI05_NO_GRAPH", "0") == "1":
    _orig_init = _fe.Pi05TorchFrontendRtx.__init__

    @functools.wraps(_orig_init)
    def _no_graph_init(self, *a, **kw):
        kw["use_cuda_graph"] = False
        _orig_init(self, *a, **kw)

    _fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init
    print("[loop-native] CUDA graph 已禁用（PI05_NO_GRAPH=1），eager 模式")

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, SensorType, G1JointGroup  # noqa: E402

# ── 关节表（数据集维序：右臂在前）──
LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT            # 与 chunk[0:7]+chunk[8:15] 逐位配对
STATE_NAMES = (RIGHT + ["right_gripper_joint1"]
               + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])
GRIP_IDX = (7, 15)                  # state 里的夹爪维：SDK 米 → 数据集 0~100%
ARM_SLICE = list(range(0, 7)) + list(range(8, 15))   # state 里的臂维（0-6 右 / 8-14 左）
# 当前 chunk 的预测时刻臂位——chunk 臂维是 delta（见 plan_step），换块时更新
BASE_ARM = None
_gcal = mf.get("gripper", {})
GRIP_WMIN, GRIP_WMAX = _gcal.get("width_min"), _gcal.get("width_max")


def state_from_joints(vals) -> np.ndarray:
    """原始关节读数 → 模型 state：夹爪两维按 manifest 标定换算 0~100%。

    数据集 state dim7/15 是百分比；SDK 读数是米——直接喂会被归一化成
    恒 ≈闭合（0.12m 当 0.12% 算），模型看到的夹爪状态永远错。
    """
    st = np.array(vals, dtype=np.float32)
    if GRIP_WMIN is None or GRIP_WMAX is None:
        print("⚠ 夹爪未标定：SDK 宽度(米)原样进 state，与数据集 0~100% 单位不符!")
        return st
    for i in GRIP_IDX:
        st[i] = float(np.clip((st[i] - GRIP_WMIN)
                              / (GRIP_WMAX - GRIP_WMIN + 1e-9) * 100.0,
                              0.0, 100.0))
    return st


def grip_state_cmd_override(st_raw):
    """--grip-state-cmd：state 夹爪维(7/15)改喂最后指令值（默认=SDK 回读）。

    SDK 夹爪反馈滞后 ~6.3-8s（2026-09-23 实测），pick 的关键窗口只有 ~7s——
    喂回读=模型全程看到冻结在起始值的夹爪状态；数据集 state 跟随指令仅
    ~0.3-1s（10-05 判读 ep0：开爪 1s 内跟上、闭到物体稳读 33.5%）。指令值
    即数据集语义的快跟随，把 train/serve 的夹爪 state 拉回一致。
    未发过指令的爪（GRIP["sent"] 空）保留回读值——起步时两者本就一致。
    """
    if GRIP is None or not args.grip_state_cmd:
        return st_raw
    st = np.array(st_raw, dtype=np.float32)
    for name, dim in GRIP["names"]:
        sent = GRIP["sent"].get(name)
        if sent is not None:
            st[dim] = sent
    return st


def read_joints(robot, names) -> np.ndarray:
    vals = robot.get_joint_positions([], names)
    if not vals or len(vals) != len(names):
        raise SystemExit(f"关节读取失败（返回 {len(vals) if vals else 0} 维）")
    return np.array(vals, dtype=np.float32)


_LAST_FRAMES = {}     # 相机冻结守卫：逐轮像素级对比（2026-09-24 右臂下压排查引入）
_FREEZE_STREAK = {}


def grab_views(robot) -> dict:
    out = {}
    for key, cam in list(CAM_MAP.items())[:VIEWS]:
        d = robot.get_rgb_data(getattr(SensorType, cam))
        if not d or not d.get("data"):
            raise SystemExit(f"取图失败: {cam}")
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"解码失败: {cam}")
        last = _LAST_FRAMES.get(cam)
        _LAST_FRAMES[cam] = img
        if last is not None and np.array_equal(last, img):
            _FREEZE_STREAK[cam] = _FREEZE_STREAK.get(cam, 0) + 1
        else:
            _FREEZE_STREAK[cam] = 0
        if _FREEZE_STREAK[cam] >= 3:
            raise SystemExit(f"⛔ 相机流疑似冻结：{cam} 连续 3 轮逐字节相同"
                             f"（实时流有传感器噪声不可能全等）——观测已失真，"
                             f"停止循环。排查 RT 相机服务")
        out[key] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out


class LatencyTracker:
    """时延样本收集器。结构参照 lerobot policies/rtc/latency_tracker.py
    （Apache-2.0），两处有意偏离（docs/lerobot-alignment.md §七）：
    maxlen 无界（lerobot 默认 100 会截断 200 轮跑的统计窗口）、保持 float64
    （lerobot 的 float32 cast 可能让 .1f 打印差 0.1ms——本类与旧
    np.percentile 路径逐字节等价：np.quantile(x, 0.95)≡np.percentile(x, 95)）。"""

    def __init__(self):
        self._buf = deque()   # 无界；perf_counter 差恒正，负值丢弃只是护栏

    def add(self, value_ms):
        if value_ms >= 0.0:
            self._buf.append(value_ms)

    def __len__(self):
        return len(self._buf)

    def max(self):
        return max(self._buf) if self._buf else float("nan")

    def percentile(self, q):
        if not self._buf:
            return float("nan")
        return float(np.quantile(np.asarray(self._buf), q))

    def p95(self):
        return self.percentile(0.95)


def pstats(tr):
    """p50/p95/max 毫秒，一行（tr: LatencyTracker）。"""
    return (f"p50 {tr.percentile(0.5):.1f} | p95 {tr.percentile(0.95):.1f} | "
            f"max {tr.max():.1f} ms")


def chunk_abs_block(chunk):
    """chunk delta 块 → 绝对构型块（包络护栏的比对对象，2026-10-07 修范畴错误）。

    训练管线 relative_actions_processor：臂维=delta（相对本块预测时刻臂位
    BASE_ARM）、夹爪维=绝对 0-100。
    包络盒来自数据集 action 列逐维 [min,max]=绝对构型——必须用 BASE_ARM+delta
    比对；直接拿 delta 比对=必拦停（delta≈0 恒不在盒内，10-07 首跑实录）。
    """
    a = np.empty_like(chunk)
    a[:, :7] = chunk[:, :7] + BASE_ARM[None, :7]
    a[:, 7] = chunk[:, 7]            # 夹爪界=±inf，原值恒过
    a[:, 8:15] = chunk[:, 8:15] + BASE_ARM[None, 7:14]
    a[:, 15] = chunk[:, 15]
    return a


def env_first_violation(ch):
    """关节包络检查：返回首个越界行 (行号, 维, 值, 界)；全干净返回 None。

    ch: (n,16) 数据集维序动作块。臂维 0-6=右臂、8-15=左臂（ENV 已含余量），
    夹爪维界=±inf 恒过。逐行向量化，n≤chunk_size(50)，微秒级。
    """
    bad = (ch > ENV_HI[None, :]) | (ch < ENV_LO[None, :])
    rows = np.flatnonzero(bad.any(axis=1))
    if rows.size == 0:
        return None
    r = int(rows[0])
    j = int(np.flatnonzero(bad[r])[0])
    upper = bool(ch[r, j] > ENV_HI[j])
    return r, j, float(ch[r, j]), float(ENV_HI[j] if upper else ENV_LO[j])


def send_grip(robot, chunk_row):
    """chunk 行 dim7/dim15（0~100%）→ 标定宽度下发（超阈值才发，非阻塞）。

    ⚠ SDK 夹爪反馈滞后 ~6.3s 且期间 is_moving 恒 False——本函数只发不查
    不等待（否则拖死控制环）；终态核对放循环结束后。
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

if args.nav_suspend:
    NAV = NavSuspend()
    NAV.suspend()

HOME = read_joints(robot, ARM_NAMES)   # 起始位 = 漂移护栏基准（应在预热工作位）
print(f"起始位（漂移护栏基准）: {np.round(HOME, 3).tolist()}")
print("⚠ 若当前不是预热工作位，先跑 g1_pose_warmup.py 再来")

t0 = time.time()
model = flash_rt.load_model(
    str(CKPT), config="pi05", num_views=VIEWS,
    cache_frames=args.cache_frames, action_dim=ACTION_DIM,
    rtc=args.rtc_horizon > 0,
    rtc_horizon=args.rtc_horizon or 10,
    rtc_max_weight=args.rtc_max_w)
ns = model._pipe.norm_stats
print(f"tier=int8_full views={VIEWS} | load {time.time() - t0:.1f}s")

obs = grab_views(robot)
state0 = state_from_joints(read_joints(robot, STATE_NAMES))
state_n0 = normalize_state(state0, ns)
BASE_ARM = state0[ARM_SLICE]   # 首块 delta 基准 = 预测时刻臂位
chunk = np.asarray(model.predict(obs, prompt=args.prompt, state=state_n0))  # 建管线
# 首用引擎构建吸收：真实控制首拍曾撞 ~800ms 惰性构建（autotune 中途重现，
# 2026-09-22 tegrastats 已排除热/内存）。连做 3 次新鲜取图推理，把构建成本
# 烧在计时区外，避免首拍尖峰污染遥测
for _ in range(3):
    model.predict(grab_views(robot), prompt=args.prompt,
                  state=normalize_state(read_joints(robot, STATE_NAMES), ns))
print("预热推理 ×3 完成（吸收惰性引擎构建）")
WMARK = max(1, min(args.n_action_steps, args.chunk_size))   # 预取水位阈值
env_hits = 0                         # 包络护栏：连续越界块数（干净块清零）
env_trunc = 0                        # 包络护栏：累计截断块数（汇总用）
if args.rtc_horizon > 0:
    print(f"[RTC] 前缀引导开：换块时旧块尾段重锚进新块坐标系喂引擎"
          f"（horizon {args.rtc_horizon}；前缀不可用时该块自动免引导）")


def plan_step(k, cur, budget=None):
    """chunk 第 k 步 → 限幅后目标（14 维，右臂在前）；budget=位移限幅。

    ⚠ 2026-09-23 语义修正：训练管线 relative_actions_processor 把臂维动作
    转成 delta（相对预测时刻 state，夹爪维除外）——chunk 臂维不是绝对目标，
    须加回本块预测时刻臂位 BASE_ARM（换块时随 job 更新）。
    """
    arm_tgt = BASE_ARM + np.concatenate([chunk[k][:7], chunk[k][8:15]])
    b = args.delta_max if budget is None else budget
    return cur + np.clip(arm_tgt - cur, -b, b)


def _rtc_prefix(new_base):
    """旧块未消费尾段 → 新块 delta 坐标系（RTC 前缀，推理侧引导输入）。

    臂维重锚：绝对构型 (OLD_BASE+delta) − NEW_BASE（新块自己的 delta 基准）
    夹爪维（7/15）数据集语义即绝对 0-100，原值携带。
    尾段不足 horizon 时按 lerobot rollout 语义 hold-pad（重复末行——零填充
    会解码到数据集均值）；块耗尽/RTC 关时返回 None（引擎侧 enable=0 短路）。
    须在 job.start 前调用（chunk/BASE_ARM 还属旧块，new_base=本次取关节读数，
    与 job 即将存下的 state_arm 同源）。
    """
    if not args.rtc_horizon or chunk is None or k >= len(chunk):
        return None
    left = chunk[k:]
    pf = np.empty_like(left)
    pf[:, :7] = left[:, :7] + BASE_ARM[None, :7] - new_base[None, :7]
    pf[:, 7] = left[:, 7]
    pf[:, 8:15] = left[:, 8:15] + BASE_ARM[None, 7:14] - new_base[None, 7:14]
    pf[:, 15] = left[:, 15]
    h = args.rtc_horizon
    if len(pf) < h:
        pf = np.concatenate([pf, np.repeat(pf[-1:], h - len(pf), axis=0)], axis=0)
    return pf


class _PredictJob:
    """后台推理作业：GPU 推理与固定节拍执行重叠。

    只有 predict 放后台；SDK 取图/读关节/下发全部留在主线程，不碰 SDK
    线程安全。同一时刻只有一个 predict 在飞（到水位才启动下一个）。
    """

    def __init__(self, model, prompt):
        self._model, self._prompt = model, prompt
        self._done = threading.Event()
        self._res = None
        self._err = None
        self.dur_ms = 0.0

    def start(self, obs, state_n, state_arm, rtc_prefix=None):
        self._done.clear()
        self._err = None
        self.state_arm = state_arm   # 本块 delta 基准（预测时刻臂位）

        def _run():
            t = time.perf_counter()
            try:
                self._res = np.asarray(self._model.predict(
                    obs, prompt=self._prompt, state=state_n,
                    rtc_prefix=rtc_prefix))
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
    """取图+读关节+夹爪换算+归一化（主线程，~22ms），耗时计入 grab_ms。

    返回 (obs, 归一化 state, 臂位原始值)——臂位原始值作该块 delta 基准随块走。
    """
    t_g = time.perf_counter()
    obs = grab_views(robot)
    st_raw = grip_state_cmd_override(
        state_from_joints(read_joints(robot, STATE_NAMES)))
    st_n = normalize_state(st_raw, ns)
    grab_ms.add((time.perf_counter() - t_g) * 1000)
    return obs, st_n, st_raw[ARM_SLICE]


# ── ② 干跑：只打印首块前 5 拍计划，绝不下发 ──
infer_ms = LatencyTracker()
grab_ms = LatencyTracker()
job = _PredictJob(model, args.prompt)
job.start(*fresh_obs())              # 首块
chunk = job.result()
infer_ms.add(job.dur_ms)
BASE_ARM = job.state_arm
job = None


def swap_envelope_check():
    """换块包络检查（v3 轮头同款）：越界截断，行 0 越界/连续越界→停。返回是否中止。"""
    global chunk, env_hits, env_trunc, aborted
    if ENV_LO is None:
        return False
    _v = env_first_violation(chunk_abs_block(chunk))
    if _v is None:
        env_hits = 0
        return False
    _r, _j, _val, _bnd = _v
    env_hits += 1
    env_trunc += 1
    _side = "上" if _val > _bnd else "下"
    print(f"⛔ 包络护栏：指令行{_r} 维{_j}={_val:.3f} 越{_side}界 "
          f"{_bnd:.3f}——截到行{_r}前缀（连续 {env_hits}/"
          f"{args.envelope_abort_n} 块）")
    if _r == 0:
        print("⛔ 行 0 即越界，无处可截——停止循环（机械臂留在原地）")
        aborted = True
        return True
    if env_hits >= args.envelope_abort_n:
        print("⛔ 连续越界=感知漂移（场景认错了），继续跑只会反复撞桌"
              "——停止循环（机械臂留在原地）")
        aborted = True
        return True
    chunk = chunk[:_r]
    return False


if not args.do_exec:
    print(f"\n== [干跑] 首块前 5 拍（tick {TICK * 1000:.1f} ms，逐拍限幅 "
          f"±{args.delta_max} rad）未下发任何命令 ==")
    for tk in range(5):
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(tk, cur)
        gap0 = float(np.max(np.abs(tgt - cur)))
        v = max(V_MIN, min(gap0 / TICK, args.speed))
        drift = float(np.max(np.abs(tgt - HOME)))
        print(f"  拍{tk}: 限幅后 {np.round(tgt, 3).tolist()}"
              f"\n        导程 {gap0 * 1000:.1f} mrad → 速度 {v:.3f} rad/s"
              f" | 离起始位峰值 {drift * 1000:.0f} mrad"
              f"（护栏 {args.max_excursion * 1000:.0f}）")
    if ENV_LO is not None:
        _v = env_first_violation(chunk_abs_block(chunk))
        _m = ("全块通过" if _v is None else
              f"⛔ 行{_v[0]} 维{_v[1]}={_v[2]:.3f} 越界 {_v[3]:.3f}")
        print(f"  包络预检: {_m}")
    if GRIP:
        desc = []
        for name, dim in GRIP["names"]:
            p = float(np.clip(chunk[0][dim], 0.0, 100.0))
            w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
            desc.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm")
        print(f"  夹爪目标: {'  '.join(desc)}")
    print("\n[干跑] 未下发任何命令。加 --exec 真实执行。")
    WATCH.restore()
    if NAV is not None:
        NAV.resume()
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)

# ── ③ 固定节拍循环（绝对截止时序；水位预取；换块弃尾）──
WATCH.pause()
input(f"\n⚠ 将连续驱动双臂 {args.rounds} 块 × 至多 {len(chunk)} 拍"
      f"（tick {TICK * 1000:.1f} ms，逐拍限幅 ±{args.delta_max} rad，"
      f"速度天花板 {args.speed} rad/s，漂移护栏 ±{args.max_excursion} rad"
      + ("，夹爪下发开启）。\n" if GRIP else ")。\n")
      + "急停就绪后回车开始，循环期间随时按 q 退出...")
WATCH.resume()

round_ms = LatencyTracker()          # 块耗时（换块到换块，对齐 v3 轮耗时口径）
tick_ms = LatencyTracker()           # tick 服务时长（不含节拍睡眠）
track_err = []                       # 每拍回读跟踪误差 mrad
aborted = False
k = 0                                # 本块已消费步数
n_swapped = 1                        # 已换入块数（首块同步换入计 1）
last_tgt = None                      # 上一拍下发目标（|Δcmd|/接缝基准）
t_loop0 = time.perf_counter()        # 任务秒表
t_block0 = t_loop0                   # 块秒表（换块节拍）
seam_first = None                    # 换块后首拍 |Δcmd|（接缝尖峰，主指标）
seam_rest = []                       # 换块后其余拍 |Δcmd|（块内均值基准）
seam_first_all = []                  # 跨块累计：每块首拍 |Δcmd|
seam_intra_all = []                  # 跨块累计：每块块内均值 |Δcmd|


def seam_flush():
    """把当前块的接缝样本并入跨块累计并复位（换块/收尾两处调用）。"""
    global seam_first, seam_rest
    if seam_first is not None:
        seam_first_all.append(seam_first)
    if seam_rest:
        seam_intra_all.append(float(np.mean(seam_rest)))
    seam_first = None
    seam_rest = []
starve_windows = 0                   # 星饿段数（块耗尽等推理的连续段）
starve_ticks = 0                     # 星饿拍总数
starve_total = 0.0                   # 星饿总时长 s
starve_worst = 0.0                   # 最长星饿段 s
_starve_t0 = None                    # 当前星饿段起点（None=不在星饿）
skip_sends = 0                       # 导程 <GAP_EPS 免发拍数
dropped_ticks = 0                    # 落后超 1 拍的重锚次数（丢拍）
next_t = time.perf_counter()
swap_envelope_check()
print(f"\n── 块 1 | t+0.0s | 推理 {infer_ms.percentile(1.0) if len(infer_ms) else float('nan'):.0f} ms ──")

while not aborted:
    # ── 节拍头：睡到绝对截止；落后超 1 拍即重锚（绝不连发追赶）──
    now = time.perf_counter()
    if now < next_t:
        time.sleep(next_t - now)
    elif now - next_t > TICK:
        dropped_ticks += 1
        next_t = now
    next_t += TICK
    t_tick = time.perf_counter()

    # ── 换块：推理落地 → 整块换入弃尾（lerobot 水位语义）──
    if job is not None and job.done():
        chunk = job.result()
        infer_ms.add(job.dur_ms)
        BASE_ARM = job.state_arm
        job = None
        n_swapped += 1
        prev_block_ticks = k          # 上块消费拍数（重置前捕获，表头用）
        k = 0
        seam_flush()
        if _starve_t0 is not None:
            _d = t_tick - _starve_t0
            starve_windows += 1
            starve_total += _d
            starve_worst = max(starve_worst, _d)
            _starve_t0 = None
        round_ms.add((t_tick - t_block0) * 1000)
        t_block0 = t_tick
        print(f"\n── 块 {n_swapped} | t+{t_tick - t_loop0:6.1f}s | "
              f"推理 {infer_ms.percentile(1.0):.0f} ms（上块 {round_ms.percentile(1.0) / 1000:.1f}s，"
              f"{prev_block_ticks} 拍）──")
        if swap_envelope_check():
            break

    exhausted = k >= len(chunk)
    if exhausted:
        # ── 星饿：不下发（刚性伺服保持最后目标原地等待），只等推理 ──
        if _starve_t0 is None:
            _starve_t0 = t_tick
        starve_ticks += 1
        if t_tick - _starve_t0 > args.starve_limit:
            print(f"⛔ 星饿超 {args.starve_limit}s（推理未归）——判挂死，停止循环"
                  "（机械臂留在原地）")
            aborted = True
    else:
        # ── 常规拍：读关节 → 限幅目标 → 夹爪 → 速度涌现下发 ──
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(k, cur, args.delta_max)
        drift = float(np.max(np.abs(tgt - HOME)))
        if drift > args.max_excursion:
            print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                  f"{args.max_excursion * 1000:.0f}，停止循环（机械臂留在原地）")
            aborted = True
            break
        cmd_delta = (tgt - last_tgt) if last_tgt is not None else np.zeros_like(tgt)
        d_mrad = float(np.max(np.abs(cmd_delta))) * 1000
        if seam_first is None:
            seam_first = d_mrad          # 接缝尖峰：换块首拍 |Δcmd|
        else:
            seam_rest.append(d_mrad)
        gp = send_grip(robot, chunk[k])
        gap0 = float(np.max(np.abs(tgt - cur)))
        if gap0 >= GAP_EPS:
            # 速度由 Δaction/tick 涌现：导程÷tick（1 步/拍 ≈ 数据集原速），
            # 上限 --speed 兜底模型瞬发大步，下限 V_MIN 驻停缓爬
            v = max(V_MIN, min(gap0 / TICK, args.speed))
            st = robot.set_joint_positions(tgt.tolist(), joint_names=ARM_NAMES,
                                           is_blocking=False, speed_rad_s=v)
            if not str(st).startswith("ControlStatus.SUCCESS"):
                print(f"⛔ 下发非 SUCCESS（{st}），停止循环")
                aborted = True
                break
        else:
            skip_sends += 1
            st = "held（导程≈0 免发）"
            v = 0.0
        last_tgt = tgt.copy()
        k += 1
        ach = read_joints(robot, ARM_NAMES)
        err = float(np.max(np.abs(ach - tgt))) * 1000
        track_err.append(err)
        print(f"  拍{k - 1}: |Δcmd| {d_mrad:5.1f} mrad | 导程 {gap0 * 1000:5.1f}"
              f" → {v:.2f} rad/s | 残差 {err:4.1f} mrad | {gp if gp else st}")

    # ── 出口：最后一块消费到水位（或包络截短后耗尽）──
    _w = min(WMARK, len(chunk))
    if n_swapped >= args.rounds and (k >= _w or k >= len(chunk)):
        break

    # ── 预取：到水位且无在飞推理 → 带前缀启动下块 ──
    if job is None and n_swapped < args.rounds and k >= _w:
        _obs, _st_n, _st_arm = fresh_obs()
        job.start(_obs, _st_n, _st_arm, rtc_prefix=_rtc_prefix(_st_arm))

    tick_ms.add((time.perf_counter() - t_tick) * 1000)

# ── ④ 汇总 ──
seam_flush()
if _starve_t0 is not None:
    _d = time.perf_counter() - _starve_t0
    starve_windows += 1
    starve_total += _d
    starve_worst = max(starve_worst, _d)
if GRIP:
    time.sleep(8.0)   # 夹爪反馈滞后 ~6.3s（2026-09-23 实测），留足再读终态
    for name, _ in GRIP["names"]:
        gs = robot.get_gripper_state(getattr(G1JointGroup, name))
        if gs is not None:
            print(f"夹爪终态 {name}: {gs.width * 1000:.1f} mm (moving={gs.is_moving})")
fin = read_joints(robot, ARM_NAMES)
exc = float(np.max(np.abs(fin - HOME))) * 1000
print(f"\n== 汇总（{n_swapped} 块 / {len(tick_ms)} 拍 / tick {TICK * 1000:.1f} ms）==")
print(f"总用时: {time.perf_counter() - t_loop0:.1f} s"
      f"（数据集单条任务 13s / 390 步参照）")
if len(infer_ms):
    print(f"推理: {pstats(infer_ms)}")
if len(grab_ms):
    print(f"取图+读关节: {pstats(grab_ms)}")
if len(tick_ms):
    print(f"tick 服务时长: {pstats(tick_ms)} | 丢拍重锚 {dropped_ticks} | 免发 {skip_sends}")
if len(round_ms):
    print(f"块耗时: {pstats(round_ms)}")
if track_err:
    print(f"导程残差: mean {np.mean(track_err):.1f} | max {max(track_err):.1f} mrad"
          + (f"；包络护栏截断 {env_trunc} 块" if ENV_LO is not None else ""))
if seam_first_all:
    _intra = float(np.mean(seam_intra_all)) if seam_intra_all else float("nan")
    print(f"接缝尖峰（换块首拍 |Δcmd|）: 均值 {np.mean(seam_first_all):.1f} | 最大 "
          f"{max(seam_first_all):.1f} mrad | 块内均值 {_intra:.1f} mrad"
          f"（RTC A/B 主指标：--rtc-horizon on 应显著压低首拍/块内比）")
if starve_windows or starve_ticks:
    print(f"星饿: {starve_windows} 段 / {starve_ticks} 拍 / 共 {starve_total:.1f} s"
          f"（最长 {starve_worst:.1f}s，上限 {args.starve_limit}s 未触发即健康）")
if dropped_ticks:
    print(f"⚠ 丢拍重锚 {dropped_ticks} 次（tick 服务超 {TICK * 1000:.1f} ms——"
          "取图拍撞推理落地拍等；持续丢拍说明 --fps 偏高）")
print(f"最终偏离起始位: {exc:.0f} mrad（护栏 {args.max_excursion * 1000:.0f} mrad）"
      + (" ⛔ 护栏触发过" if aborted else ""))

WATCH.restore()
if NAV is not None:
    NAV.resume()
print("SDK 关闭中...")
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
os._exit(0)   # SDK 残留线程，干净退出
