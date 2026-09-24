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
  夹爪维度也是增量百分比（数据 Δ 可达 −33.5/帧），当绝对 0~100% 下发会把
  负增量 clip 成 0% = 砸紧——增量→绝对目标的换算未设计完成前 --grip 拒绝。

安全设计（同 run_g1_loop）：
  只动双臂 14 关节；每条指令位移限幅；偏离起始位护栏（默认 0.5 rad，比
  loop 的 3.0 紧——首测小行程）；回车确认 + q 即退 + 物理急停第一优先级；
  默认干跑只打印计划，--exec 才真实执行。

用法:
  ~/holy/run.sh ~/holy/scripts/inference/run_dataset_execute.py \
      [--episode 0] [--start-frame 0] [--rounds 10] \
      [--steps-per-round 3] [--steps-per-cmd 3] [--delta-max 0.05] \
      [--speed 0.15] [--max-excursion 0.5] [--switch-dist 0.06] \
      [--tier bf16] [--exec]
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
                help="每轮消费 chunk 前 K 步（模型 10 步/次）")
ap.add_argument("--steps-per-cmd", type=int, default=3,
                help="合步：一条 SDK 指令跨 K 个 chunk 步（track 平滑轮廓）")
ap.add_argument("--delta-max", type=float, default=0.05,
                help="每步增量限幅 rad（夹爪维不执行，见 --grip）")
ap.add_argument("--speed", type=float, default=0.15, help="关节速度上限 rad/s")
ap.add_argument("--max-excursion", type=float, default=0.5,
                help="偏离起始位护栏 rad（首测收紧；ep0 全程左臂包络 2.64）")
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
ap.add_argument("--grip", action="store_true",
                help="（未实现）夹爪增量语义换算完成前一律拒绝")
args = ap.parse_args()
sys.stdout.reconfigure(line_buffering=True)   # os._exit 不刷缓冲，管道跑必须行缓冲
g1_config.apply(args, {"ckpt": ("run", "ckpt")})
if args.grip:
    raise SystemExit("夹爪维度是增量百分比，直接当绝对目标下发=砸紧。增量→绝对"
                     "换算未设计完成，本脚本暂只动双臂关节。")

if args.tier == "int8_full":
    os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
elif args.tier == "int8_enc":
    os.environ.setdefault("FVK_PI05_RTX_INT8_ENCODER_ONLY", "1")
os.environ.setdefault("PI05_NO_GRAPH", "0")   # graph 已修复（WithFlags）
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")


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
            print("（随时按 q 退出；确认提示符处用回车/Ctrl-C；急停第一优先级）")

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
    """infer 返回 dict → (10,16) 动作数组（与回放脚本同提取）。"""
    if isinstance(x, dict):
        x = x.get("actions", x.get("raw_actions",
                    next(v for v in x.values() if hasattr(v, "shape"))))
    return np.asarray(x, dtype=np.float32)


def predict_chunk(obs, fidx):
    """set_prompt + 按帧号固定噪声（与回放验证同路径）。

    不用 predict：它每轮掷随机噪声——同图换噪声两两 cos≈0.25（混沌底，
    ab_real_camera 实测），等于每轮从动作分布重新抽签，会抽出抬臂等野
    策略。固定噪声下回放帧 0-21 的输出就是"保持小步接近"，与此处预期
    一致。state 每次仍取真机实时值。
    """
    model.set_prompt(PROMPT, state=state_n())
    gen = torch.Generator().manual_seed(args.seed + fidx)
    return _acts(model.infer(obs, noise=torch.randn(10, 32, generator=gen)))


model.predict(obs0, prompt=PROMPT, state=state_n())   # 首次必须走 predict 建管线
predict_chunk(obs0, args.start_frame)                 # 再吸收一次（固定噪声路径）
print("预热推理 ×2 完成")

n_steps = max(1, min(args.steps_per_round, 10))
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
        chunk = predict_chunk(grab_frame(f), f)
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
for r in range(args.rounds):
    f = min(cursor, ep_last)
    chunk = predict_chunk(grab_frame(f), f)
    print(f"\n── 轮 {r} | 数据集帧 {f} | 推理完成 ──")
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
WATCH.restore()
print("SDK 关闭中...")
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
for c in _caps.values():
    c.release()
os._exit(0)
