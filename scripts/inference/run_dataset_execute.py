#!/usr/bin/env python
"""数据集图像 → 真机执行：图像取自数据集 mp4，state/执行都在真机（混合回放）。

与 run_g1_loop（实时相机闭环）和 run_dataset_replay（纯开环对比）的区别：
  图像 = 数据集时间线游标处的 mp4 帧（逐轮前移），state = 真机实时读数，
  输出 = 模型增量 Δ，按【增量语义】下发给真机双臂。

⚠ 与 run_g1_loop 的关键语义差异（2026-09-24）：
  loop 的 plan_step 把 chunk 当绝对目标（arm_tgt − cur 向目标滑），而模型输出
  是增量——增量当绝对值会变成"每条指令恒向同方向滑 delta_max"，疑似闭环旧案
  "漂移持续同号 ~95 mrad/指令"的真正来源。本脚本目标 = 当前 + clip(Δ合步和,
  ±步数×delta_max)。
  夹爪维度=【绝对指令 %】（训练处理器 exclude_joints=["gripper"]，臂维才做
  delta 化，2026-09-28 回放实证：模型爪输出对绝对 action MAE ~1%）——下发
  无需换算，但本脚本 --grip 未真机验证前仍拒绝（闭环 grip 走 run_g1_loop）。

安全设计（同 run_g1_loop）：
  只动双臂 14 关节；每条指令位移限幅；偏离起始位护栏；回车确认 + q/Ctrl-C
  即退（信号直通管道，主线程卡死也能退）+ 物理急停第一优先级；
  默认干跑只打印计划，--exec 才真实执行。
  流水线：下一块推理与当前块执行重叠（_PredictJob，run_g1_loop 同款），
  轮间零等待；条件 state 比消费时刻早一个执行窗，由 plan_cmd 的
  "当前+Δ"自校正消化。

用法:
  ~/holy/run.sh ~/holy/scripts/inference/run_dataset_execute.py \
      [--episode 0] [--start-frame 0] [--rounds 10] \
      [--steps-per-round 3] [--steps-per-cmd 3] [--delta-max 0.05] \
      [--speed 0.15] [--max-excursion 3.0] [--switch-dist 0.06] \
      [--tier bf16] [--horizon 10] [--exec]
horizon：--horizon 50 用训练原生长度整块推理（env 自动设，延迟几乎不变），
配合 --steps-per-round 15~20 消费甜点区（2026-09-28 回放实验：pos20+ 衰减，
pos40-49 不可用）。
"""
import argparse
import atexit
import functools
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

import g1_config  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--npz", default="~/holy/datasets/replay_input.npz")
ap.add_argument("--episode", type=int, default=0, help="回放的 episode 号（默认 0）")
ap.add_argument("--start-frame", type=int, default=0, help="数据集帧游标起点")
ap.add_argument("--prompt", default=None, help="任务指令（默认用数据集 task 原句）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实驱动双臂（默认干跑只打印计划）")
ap.add_argument("--rounds", type=int, default=10, help="推理→执行轮数")
ap.add_argument("--steps-per-round", type=int, default=3,
                help="每轮消费 chunk 前 K 步（≤ --horizon；50 块甜点区 15-20）")
ap.add_argument("--steps-per-cmd", type=int, default=3,
                help="合步：一条 SDK 指令跨 K 个 chunk 步（track 平滑轮廓）")
ap.add_argument("--delta-max", type=float, default=0.05,
                help="每步增量限幅 rad（夹爪维不执行，见 --grip）")
ap.add_argument("--speed", type=float, default=0.15, help="关节速度上限 rad/s")
ap.add_argument("--max-excursion", type=float, default=3.0,
                help="偏离起始位护栏 rad（2026-09-28 放宽=loop 同款数据集标定："
                     "199 轨合法抓取全程包络 max 2.785；首测值 0.5 会把正常 "
                     "reach 半途掐断——ep0 实测 7 轮即触 635 mrad）")
ap.add_argument("--switch-dist", type=float, default=0.06,
                help="提前换目标阈值 rad（滑行中转向，消指令边界停顿）")
ap.add_argument("--align", action="store_true",
                help="执行前先把双臂限幅慢速挪到该 episode 帧 0 的真实录制位姿"
                     "（warmup 目标是 state.mean，与帧 0 可差 ~0.8 rad；不对齐则"
                     "模型会先花数轮把臂拉向帧 0 位姿，护栏要放够）")
ap.add_argument("--seed", type=int, default=0,
                help="flow-matching 噪声种子（按 帧号+seed 定种，同帧必同输出；"
                     "ab_real_camera 教训：predict 的随机噪声=每轮抽签，"
                     "同图换噪声两两 cos≈0.25，会抽出抬臂等野策略）")
ap.add_argument("--tier", default="bf16", choices=("bf16", "int8_enc", "int8_full"),
                help="量化档（默认 bf16；int8 数值已 A/B 验证等价，快 ~100ms）")
ap.add_argument("--horizon", type=int, default=10,
                help="chunk 长度（训练=50，2026-09-28 实锤；10=原部署切片。"
                     "50 块延迟几乎不变，质量甜点区≈前 15-20 步，配合 "
                     "--steps-per-round 消费）")
ap.add_argument("--grip", action="store_true",
                help="（未实现）夹爪增量语义换算完成前一律拒绝")
args = ap.parse_args()
sys.stdout.reconfigure(line_buffering=True)   # os._exit 不刷缓冲，管道跑必须行缓冲
# chunk 长度：pi05_rtx 前端模块导入时读此 env，必须在 import flash_rt 前定死
os.environ["FLASH_RT_PI05_ACTION_CHUNK_SIZE"] = str(args.horizon)
g1_config.apply(args, {"ckpt": ("run", "ckpt")})
if args.grip:
    raise SystemExit("夹爪维=绝对指令%（exclude_joints 实锤），下发本身无需换算，"
                     "但本脚本 --grip 未真机验证。暂只动双臂关节；闭环 grip 用 "
                     "run_g1_loop --grip（已干跑验证）。")

if args.tier == "int8_full":
    os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
elif args.tier == "int8_enc":
    os.environ.setdefault("FVK_PI05_RTX_INT8_ENCODER_ONLY", "1")
os.environ.setdefault("PI05_NO_GRAPH", "1")  # 本机 FlashRT 未打 hotfix_flashrt WithFlags 补丁（cuda_graph.py 仍旧式 Instantiate，r35.6 必段错误），必须 eager；打上热修后可翻回 0
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")


class QuitWatcher:
    """后台线程监听键盘 q：任何阶段即时退出（SDK 阻塞 C++ 调用吞 Ctrl-C）。

    Ctrl-C/SIGTERM 直通本线程：signal.set_wakeup_fd 让 C 层收到信号即往管道
    写字节（不等主线程跑 Python 处理函数——主线程卡死在 SDK C++ 调用时那才
    是致命的，2026-09-29 真机实录：q 之外所有中断全被吞，只能 kill -9）。
    管道有字节 → 恢复终端 + os._exit，绕过一切被吞的可能。"""

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
            signal.signal(signal.SIGINT, lambda *_: None)     # 退出走管道，别靠
            signal.signal(signal.SIGTERM, lambda *_: None)    # 会被推迟的异常
            self._wake_r = r
            self.active.set()
            threading.Thread(target=self._loop, daemon=True).start()
            print("（随时按 q 或 Ctrl-C 退出；急停第一优先级）")

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


WATCH = QuitWatcher()
robot = None


def _fatal_hook(t, v, tb):
    print(f"\n⛔ 异常退出: {t.__name__}: {v}")
    if robot is not None:
        try:
            robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
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

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import GalbotRobot  # noqa: E402

# ── npz（与 run_dataset_replay 同一份产物）──
npz_p = pathlib.Path(args.npz).expanduser()
z = np.load(npz_p, allow_pickle=False)
states, ep_id, frame_idx, task_idx = (z["states"], z["ep_id"], z["frame_idx"],
                                      z["task_idx"])
tasks = z["tasks"].tolist()
cams = z["cams"].tolist()
video_template = str(z["video_template"])
video_base = (pathlib.Path(str(z["dataset_root"])) if "dataset_root" in z
              else npz_p.parent)

EP = args.episode
ep_rows = np.where(ep_id == EP)[0]
if not len(ep_rows):
    raise SystemExit(f"npz 里没有 episode {EP}（有 {sorted(set(ep_id.tolist()))}）；"
                     "先跑 extract_dataset_frames.py --episodes 补提取")
base = int(ep_rows[0])
ep_last = int(frame_idx[ep_rows].max())
SDK2DS = {"HEAD": "observation.images.head_right",
          "LEFT_ARM": "observation.images.left_arm",
          "RIGHT_ARM": "observation.images.right_arm"}
ds_of = {}
for mkey, sdk in CAM_MAP.items():
    hits = [v for k, v in SDK2DS.items() if k in sdk.upper()]
    ds_of[mkey] = hits[0] if hits else None

_caps = {}


def grab_frame(fidx) -> dict:
    """数据集第 fidx 帧 → 模型 obs 三图（与回放脚本同一寻址）。"""
    r = base + int(np.where(frame_idx[ep_rows] == fidx)[0][0])   # 该帧在 npz 的行号
    out = {}
    for mkey, dskey in ds_of.items():
        ci = cams.index(dskey)
        path = str(video_base / video_template.format(
            video_key=dskey, chunk_index=int(z["vid_chunk"][r, ci]),
            file_index=int(z["vid_file"][r, ci])))
        cap = _caps.get(path)
        if cap is None:
            cap = _caps[path] = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(z["vid_frame"][r, ci])
                + (fidx - int(frame_idx[r])))    # 表内偏移 + 帧内推进
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"解码失败 ep{EP} frame{fidx} ({path})")
        out[mkey] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return {k: out[k] for k in list(ds_of)[:VIEWS]}


# ── SDK（无相机；图像来自数据集）──
print("== ① SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(5)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")
if not str(st).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止")

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT
STATE_NAMES = (RIGHT + ["right_gripper_joint1"] + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])
GRIP_IDX = (7, 15)
_gcal = mf.get("gripper", {})
GRIP_WMIN, GRIP_WMAX = _gcal.get("width_min"), _gcal.get("width_max")


def state_from_joints(vals) -> np.ndarray:
    st = np.array(vals, dtype=np.float32)
    if GRIP_WMIN is None or GRIP_WMAX is None:
        print("⚠ 夹爪未标定：SDK 宽度(米)原样进 state，与数据集 0~100% 单位不符!")
        return st
    for i in GRIP_IDX:
        st[i] = float(np.clip((st[i] - GRIP_WMIN)
                              / (GRIP_WMAX - GRIP_WMIN + 1e-9) * 100.0,
                              0.0, 100.0))
    return st


def read_joints(names):
    vals = robot.get_joint_positions([], names)
    if not vals or len(vals) != len(names):
        raise SystemExit(f"关节读取失败（返回 {len(vals) if vals else 0} 维）")
    return np.array(vals, dtype=np.float32)


HOME = read_joints(ARM_NAMES)      # 漂移护栏基准（应在预热工作位）
print(f"起始位: {np.round(HOME, 3).tolist()}")
print("⚠ 若当前不是预热工作位，先跑 g1_pose_warmup.py")

# ── 模型 ──
t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=VIEWS,
                            cache_frames=1, action_dim=ACTION_DIM)
ns = model._pipe.norm_stats
PROMPT = args.prompt or str(tasks[int(task_idx[base])])
print(f"tier={args.tier} views={VIEWS} | prompt={PROMPT!r} | "
      f"load {time.time() - t0:.1f}s")

obs0 = grab_frame(args.start_frame)
state_n = lambda: normalize_state(
    state_from_joints(read_joints(STATE_NAMES)), ns)


def _acts(x):
    """infer 返回 dict → (H,16) 动作数组（H=args.horizon，与回放脚本同提取）。"""
    if isinstance(x, dict):
        x = x.get("actions", x.get("raw_actions",
                    next(v for v in x.values() if hasattr(v, "shape"))))
    return np.asarray(x, dtype=np.float32)


class _PredictJob:
    """后台推理作业：GPU 推理与 SDK 阻塞执行重叠，消轮间 ~330ms 空窗
    （移植自 run_g1_loop 的流水线。FlashRT 的 RTC=prefix 对齐只有 Thor
    路径有，Orin 无此实现——本类是纯重叠版）。

    只有 set_prompt+infer 放后台；取图/读关节/下发全留主线程，不碰 SDK
    线程安全。同一时刻只有一个 predict 在飞（消费完才启动下一个）。
    噪声仍按 帧号+seed 固定（同帧必同输出），语义与串行版一致。
    """

    def __init__(self):
        self._done = threading.Event()
        self._res = None
        self._err = None
        self.dur_ms = 0.0

    def start(self, obs, fidx, st_n):
        """st_n 由主线程在启动时刻读（SDK 不进后台线程）。

        ⚠ 与串行版的语义差：下一块的 state 取自本轮执行【前】，比消费
        时刻早一个执行窗（15 步 ≈ 0.5s 数据集时）。执行侧 plan_cmd 是
        "当前+Δ"自校正，staleness 由闭环消化（loop 同款真机 5/5 验证）。
        """
        self._done.clear()
        self._err = None

        def _run():
            t = time.perf_counter()
            try:
                model.set_prompt(PROMPT, state=st_n)
                gen = torch.Generator().manual_seed(args.seed + fidx)
                self._res = _acts(model.infer(
                    obs, noise=torch.randn(args.horizon, 32,
                                           generator=gen)))
            except Exception as e:          # 主线程 result() 时再抛
                self._err = e
            self.dur_ms = (time.perf_counter() - t) * 1000
            self._done.set()

        threading.Thread(target=_run, daemon=True).start()

    def result(self):
        self._done.wait()
        if self._err is not None:
            raise self._err
        return self._res


model.predict(obs0, prompt=PROMPT, state=state_n())   # 首次必须走 predict 建管线
_warm = _PredictJob()
_warm.start(grab_frame(args.start_frame), args.start_frame, state_n())
_warm.result()                                        # 再吸收一次（固定噪声路径）
print("预热推理 ×2 完成")

n_steps = max(1, min(args.steps_per_round, args.horizon))
spc = max(1, min(args.steps_per_cmd, n_steps))
BUDGET = spc * args.delta_max      # 每条指令位移限幅 = 步数×单步限幅
cursor = args.start_frame


def plan_cmd(chunk, k, cur):
    """chunk 步 [k, k+spc) 的增量合步和 → 限幅后绝对目标（14 维，右臂在前）。"""
    d = np.zeros(14, np.float32)
    for j in range(k, min(k + spc, len(chunk))):
        row = chunk[j]
        d[:7] += row[:7]
        d[7:] += row[8:15]
    return cur + np.clip(d, -BUDGET, BUDGET)


# ── 干跑 ──
if not args.do_exec:
    print(f"\n== [干跑] 计划：ep{EP} 从帧 {cursor} 起 {args.rounds} 轮 × "
          f"{n_steps} 步（合步 {spc}/指令，每条位移 ≤{BUDGET:.2f} rad，"
          f"护栏 ±{args.max_excursion} rad）——未下发任何命令 ==")
    cur = (np.concatenate([states[base][:7], states[base][8:15]]).astype(np.float32)
           if args.align else HOME.copy())   # 对齐后模型从帧 0 位姿出发
    base_ref = cur if args.align else HOME   # 对齐后护栏从帧 0 位姿起算
    for r in range(args.rounds):
        f = min(cursor + r * n_steps, ep_last)
        _j = _PredictJob()
        _j.start(grab_frame(f), f, state_n())
        chunk = _j.result()
        tgt = plan_cmd(chunk, n_steps - 1, cur)
        drift = float(np.max(np.abs(tgt - base_ref)))
        step_d = [float(np.max(np.abs(
            np.concatenate([chunk[j][:7], chunk[j][8:15]]))))
            for j in range(n_steps)]
        print(f"  轮{r} 帧{f:4d}: 模型Δ max/步 {max(step_d):.3f} rad | "
              f"合步目标离{'帧0位' if args.align else '起始位'} {drift * 1000:.0f} mrad"
              f"（护栏 {args.max_excursion * 1000:.0f}）"
              + (" ⛔越护栏" if drift > args.max_excursion else ""))
        cur = tgt
        if drift > args.max_excursion:
            break
    print("\n[干跑] 未下发任何命令。加 --exec 真实执行。")
    WATCH.restore()
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)

# ── 执行 ──
def guarded_move_to(target14, label, guard_rad=None):
    """限幅慢速挪到目标位姿（对齐段用）；返回是否达成。

    guard_rad 默认 args.max_excursion；对齐目标本身可在护栏外（录制安全位姿），
    调用方按 |target−HOME| 放宽——路径仍是限幅+限速的小步逼近。
    """
    g = guard_rad if guard_rad is not None else args.max_excursion
    for it in range(60):
        cur = read_joints(ARM_NAMES)
        d = target14 - cur
        if float(np.max(np.abs(d))) <= args.switch_dist:
            return True
        if float(np.max(np.abs((cur + np.clip(d, -BUDGET, BUDGET)) - HOME))) > g:
            print(f"⛔ {label}: 第{it + 1}步判读越护栏（限 {g * 1000:.0f} mrad），"
                  f"停在半程")
            return False
        robot.set_joint_positions((cur + np.clip(d, -BUDGET, BUDGET)).tolist(),
                                  joint_names=ARM_NAMES, is_blocking=False,
                                  speed_rad_s=args.speed)
        deadline = time.perf_counter() + max(2.0, 1.5 * BUDGET / max(args.speed, 0.01))
        while time.perf_counter() < deadline:
            if float(np.max(np.abs(read_joints(ARM_NAMES)
                                   - (cur + np.clip(d, -BUDGET, BUDGET))))
                     <= args.switch_dist):
                break
            time.sleep(0.02)
    return float(np.max(np.abs(read_joints(ARM_NAMES) - target14))) <= args.switch_dist


WATCH.pause()
input(f"\n⚠ 将 {args.rounds} 轮真实驱动双臂（图像=数据集 ep{EP} 帧时间线，"
      f"每条指令位移 ≤{BUDGET:.2f} rad，漂移护栏 ±{args.max_excursion} rad）。\n"
      "急停就绪后回车开始，随时按 q 退出...")
WATCH.resume()

if args.align:
    tgt0 = np.concatenate([states[base][:7], states[base][8:15]]).astype(np.float32)
    off = float(np.max(np.abs(tgt0 - HOME)))
    g_align = max(args.max_excursion, off * 1.1)
    print(f"\n== 对齐段：挪到 ep{EP} 帧 0 录制位姿（限幅 {BUDGET:.2f} rad/指令，"
          f"距当前 {off * 1000:.0f} mrad，本段护栏放宽至 {g_align * 1000:.0f}）==")
    if not guarded_move_to(tgt0, "对齐", guard_rad=g_align):
        print("⛔ 对齐未达成，终止（臂留在原地）")
        WATCH.restore()
        robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
        os._exit(1)
    HOME = read_joints(ARM_NAMES)   # 护栏重基准：对齐后漂移从帧 0 位姿起算
    print(f"对齐完成，护栏基准更新: {np.round(HOME, 3).tolist()}")

cmd_hist = None
aborted = False
lat = []
job = _PredictJob()
job.start(grab_frame(min(cursor, ep_last)), min(cursor, ep_last), state_n())   # 轮 0
for r in range(args.rounds):
    if aborted:
        break
    f = min(cursor, ep_last)
    chunk = job.result()             # 上轮执行期间启动的推理——此刻早已就绪
    lat.append(job.dur_ms)
    print(f"\n── 轮 {r} | 数据集帧 {f} | 推理 {job.dur_ms:.0f} ms"
          "（与上轮执行重叠，零等待）──")
    cursor_next = cursor + n_steps
    if r + 1 < args.rounds and cursor_next + 1 <= ep_last:
        job.start(grab_frame(cursor_next), cursor_next, state_n())   # 藏进本轮执行期
    for k in range(0, n_steps, spc):
        cur = read_joints(ARM_NAMES)
        tgt = plan_cmd(chunk, k, cur)
        drift = float(np.max(np.abs(tgt - HOME)))
        if drift > args.max_excursion:
            print(f"⛔ 漂移护栏：{drift * 1000:.0f} mrad > "
                  f"{args.max_excursion * 1000:.0f}，停止（臂留在原地）")
            aborted = True
            break
        cmd_delta = (tgt - cmd_hist) if cmd_hist is not None else np.zeros_like(tgt)
        cmd_hist = tgt.copy()
        t_s = time.perf_counter()
        robot.set_joint_positions(tgt.tolist(), joint_names=ARM_NAMES,
                                  is_blocking=False, speed_rad_s=args.speed)
        min_dwell, gate = 0.15, args.switch_dist
        deadline = t_s + max(2.0, 1.5 * BUDGET / max(args.speed, 0.01))
        while True:
            ach = read_joints(ARM_NAMES)
            if (time.perf_counter() - t_s >= min_dwell
                    and float(np.max(np.abs(ach - tgt))) <= gate):
                break
            if time.perf_counter() > deadline:
                break
            time.sleep(0.02)
        err = float(np.max(np.abs(read_joints(ARM_NAMES) - tgt))) * 1000
        print(f"  指令[步{k}]: |Δcmd| "
              f"{float(np.max(np.abs(cmd_delta))) * 1000:5.1f} mrad | "
              f"执行 {time.perf_counter() - t_s:.2f} s | 回读偏差 {err:4.1f} mrad")
    if aborted:
        break
    cursor += n_steps
    if cursor + 1 > ep_last:
        print(f"\n数据集帧游标到头（ep{EP} 长 {ep_last + 1} 帧），结束")
        break

fin = read_joints(ARM_NAMES)
print(f"\n== 汇总 ==\n最终偏离起始位: "
      f"{float(np.max(np.abs(fin - HOME))) * 1000:.0f} mrad"
      f"（护栏 {args.max_excursion * 1000:.0f}）" + (" ⛔护栏触发过" if aborted else ""))
if lat:
    print(f"[推理延迟 ×{len(lat)}] mean {np.mean(lat):.0f} | p50 "
          f"{np.percentile(lat, 50):.0f} | max {np.max(lat):.0f} ms"
          "（首轮含管线构建，偏大属正常）")
WATCH.restore()
print("SDK 关闭中...")
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
for c in _caps.values():
    c.release()
os._exit(0)
