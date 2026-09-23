#!/usr/bin/env python
"""G1 姿态预热：从任意当前姿态 → 数据集工作姿态（推理前置步骤）。

背景：pi0.5 的 state 走 256-bin 离散化进 prompt，姿态偏离采集分布时
归一化越界 → bin 打满 → 模型看到物理极限假象，动作无语义（2026-09-22
真机首跑 [-1,1] 外 15/23 维即此问题）。

两段式（安全优先）：
  ① move_whole_body_joint_zero —— SDK 预定义零位，带碰撞检查（reset_pos.py
     已验证用法），先脱离收拢/折叠等大偏移姿态。--skip-zero 可跳过
  ② 双臂 7×2 + 头 2 + 腿 5（躯干站高/前倾）set_joint_positions → 数据集
     state.mean（采集平均工作姿态，与零位接近，小步幅）。臂/头限速默认
     0.15 rad/s，腿 0.2 rad/s。--skip-leg 可跳过腿部（会改变站高/前倾，
     确认底盘平衡在位；SDK 若拒收腿关节会打印非 SUCCESS）
腿部默认随段② 对齐 state.mean（数据集各轨迹同一起点，躯干姿态也是分布
一部分）；--skip-leg 显式退出。
夹爪：读数按 manifest 标定换算 0~100% 展示；--grip 显式开启时闭合到数据集
起点 0%（width_min；199 轨全部从 0% 起步——2026-09-23 实证）。闭合命令在
段① 前一次性发出（非阻塞，与整身/臂动作并行——夹爪与臂位无物理耦合，
反馈滞后 ~6.3s 的等待被动作重叠掉），达标判定处统一核对，默认不动。
腿部 set_joint_positions 真机已验证（2026-09-23，SUCCESS 且逐关节到位；
无需换 GalbotMotion 腿接口）。state.mean 距自然站高仅 13-16°（相对 SDK
零位 ~82° 的行程机器人不会真走——平衡控制器托着站高）。
完成后回读 23 维 state 按部署 stats 归一化，报告 [-1,1] 外维数——arms
dim0-11 入域即达标；leg/head 窄维（宽 <1e-3）可能仍亮，属传感器噪声量级。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/g1_pose_warmup.py \
      [--ckpt ~/holy/models/pi05_g1_deploy] [--skip-zero] [--speed 0.15] [--grip]

中断：随时按 q 即退（含 SDK 阻塞运动中——Ctrl-C 会被阻塞 C++ 调用推迟，
q 走独立监听线程不受限）；需立即断运动拍物理急停。
"""
import argparse
import atexit
import json
import os
import pathlib
import select
import sys
import termios
import threading
import time
import tty

import g1_config   # noqa: E402  同目录共享配置（CLI > config/g1.toml > 内置默认）

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None,
                help="部署目录（默认 config [run].ckpt；读 norm_stats.json 的 state.mean 作目标）")
ap.add_argument("--skip-zero", action="store_true", default=None,
                help="跳过零位段（已在工作位附近时；config [warmup].skip_zero）")
ap.add_argument("--speed", type=float, default=None,
                help="臂/头关节速度上限 rad/s（默认 0.15；config [warmup].speed）")
ap.add_argument("--grip", action="store_true", default=None,
                help="闭合夹爪到数据集起点 0%%（width_min；需 manifest 标定；config [gripper].enabled）")
ap.add_argument("--no-grip", action="store_true",
                help="显式不闭合夹爪（覆盖 config 的 gripper.enabled=true）")
ap.add_argument("--skip-leg", action="store_true", default=None,
                help="跳过躯干/腿对齐（腿部动作改变站高/前倾，默认对齐；config [warmup].skip_leg）")
ap.add_argument("--config", default=g1_config.DEFAULT_PATH,
                help="配置文件路径（优先级 CLI > config > 内置默认）")
args = ap.parse_args()
g1_config.apply(args, {
    "ckpt": ("run", "ckpt"),
    "speed": ("warmup", "speed"),
    "skip_zero": ("warmup", "skip_zero"),
    "skip_leg": ("warmup", "skip_leg"),
    "grip": ("gripper", "enabled"),
})
if args.no_grip:
    args.grip = False


class QuitWatcher:
    """后台线程监听键盘 q：任何阶段即时退出脚本。

    Ctrl-C 的 KeyboardInterrupt 只在 Python 字节码间隙触发，SDK 阻塞的
    C++ 运动调用（set_joint_positions/move_whole_body_joint_zero）期间
    按了也没反应——q 键线程绕开这一限制。物理急停仍是第一优先级。
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._old = None
        self.active = threading.Event()   # input() 提示符处 pause，防抢键
        if os.isatty(self.fd):
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)          # 单键即达，无需回车
            atexit.register(self.restore)
            self.active.set()
            threading.Thread(target=self._loop, daemon=True).start()
            print("（运动等待期可按 q 退出；提示符处用回车/Ctrl-C；紧急停止拍物理急停）")

    def pause(self):
        self.active.clear()

    def resume(self):
        self.active.set()

    def restore(self):
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def _loop(self):
        while True:
            if not self.active.is_set() or not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出脚本（已下发目标可能仍在"
                          "限速执行中；需立即断运动请拍急停）")
                    self.restore()
                    os._exit(2)   # os._exit 跳过 atexit，终端状态先手动还原


WATCH = QuitWatcher()


def _kb_hook(t, v, tb):
    """Ctrl-C 落在 SDK 调用间隙时的收尾：关 SDK + 还原终端 + 硬退。"""
    if t is KeyboardInterrupt:
        print("\n⛔ Ctrl-C —— 收尾退出")
        rob = globals().get("robot")
        if rob is not None:
            try:
                rob.request_shutdown(); rob.wait_for_shutdown(); rob.destroy()
            except Exception:
                pass
        WATCH.restore()
        os._exit(130)
    sys.__excepthook__(t, v, tb)


sys.excepthook = _kb_hook

# ── 目标姿态：部署 stats 的 state.mean（单一事实来源）──
CKPT = pathlib.Path(args.ckpt)
ns_p = CKPT / "norm_stats.json"
if ns_p.exists():                       # prep 产物：{norm_mode, actions{...}, state{...}}
    ns = json.loads(ns_p.read_text())
    TARGET = ns["state"]["mean"]
    MODE = ns.get("norm_mode", "q01_q99")
else:                                   # 兜底：lerobot 数据集 schema
    ms_p = CKPT.parent / "meta" / "stats.json"
    if not ms_p.exists():
        raise SystemExit(f"找不到 stats：{ns_p} 或 {ms_p}")
    lg = json.loads(ms_p.read_text())
    TARGET = lg["observation.state"]["mean"]
    MODE = "q01_q99"
if len(TARGET) != 23:
    raise SystemExit(f"state.mean 应 23 维，实得 {len(TARGET)}")

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
HEAD = ["head_joint1", "head_joint2"]
LEG = [f"leg_joint{i}" for i in range(1, 6)]
GRIP = ["right_gripper_joint1", "left_gripper_joint1"]   # 数据集序: dim7 右 / dim15 左
# ⚠ 数据集 23 维序是【右臂在前】（meta/info.json 权威定义，2026-09-22 实证）:
#   0-6 右臂 / 7 右夹爪 / 8-14 左臂 / 15 左夹爪 / 16-20 腿 / 21-22 头
# 读取顺序一律跟数据集序，杜绝左←→右整块互换（曾把右臂目标下给左臂，
# 机械臂镜像甩到背后——用户实测 "到背面来抓东西了"）
STATE_NAMES = RIGHT + GRIP[:1] + LEFT + GRIP[1:] + LEG + HEAD
IDX = {"right_arm": list(range(0, 7)), "left_arm": list(range(8, 15)),
       "head": list(range(21, 23)), "leg": list(range(16, 21))}

sys.path.insert(0, "/home/galbot/holy/FlashRT")
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, GalbotMotion, G1JointGroup  # noqa: E402

# 夹爪标定（manifest；有则读数换算 0~100%，与数据集 state 单位对齐）
mf_p = CKPT / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
_gcal = mf.get("gripper", {})
GRIP_WMIN, GRIP_WMAX = _gcal.get("width_min"), _gcal.get("width_max")
if args.grip and (GRIP_WMIN is None or GRIP_WMAX is None):
    raise SystemExit("--grip 需要 manifest 标定；先跑 g1_ckpt_prep "
                     "--grip-wmin/--grip-wmax（2026-09-23 实测 0.0005/0.1200 m）")


def read23(robot):
    vals = robot.get_joint_positions([], STATE_NAMES)
    if not vals or len(vals) != 23:
        raise SystemExit(f"关节读取失败（返回 {len(vals) if vals else 0} 维）")
    st = [float(v) for v in vals]
    if GRIP_WMIN is not None and GRIP_WMAX is not None:
        for i in (7, 15):   # SDK 米 → 数据集 0~100%
            st[i] = (st[i] - GRIP_WMIN) / (GRIP_WMAX - GRIP_WMIN + 1e-9) * 100.0
    return st


def report(cur):
    print(f"{'维':>4} {'关节':<22} {'当前':>8} {'目标':>8} {'差':>8}")
    for i, n in enumerate(STATE_NAMES):
        d = TARGET[i] - cur[i]
        mark = " ←" if abs(d) > 0.05 and n not in GRIP else ""
        print(f"{i:>4} {n:<22} {cur[i]:>8.3f} {TARGET[i]:>8.3f} {d:>+8.3f}{mark}")


print(f"目标 = {CKPT.name} 的 state.mean（norm_mode={MODE}）")
cur = None

print("\n== ① SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
motion = GalbotMotion()
if not robot.init():
    raise SystemExit("robot.init 失败")
motion.init()
time.sleep(5)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")

cur = read23(robot)
report(cur)

def fire_grip_close():
    """夹爪闭合命令一次性发出（非阻塞），与臂动作并行；末端统一核对。"""
    s1 = robot.set_gripper_command(G1JointGroup.right_gripper,
                                   GRIP_WMIN, 0.05, 30, False)
    s2 = robot.set_gripper_command(G1JointGroup.left_gripper,
                                   GRIP_WMIN, 0.05, 30, False)
    print(f"夹爪闭合命令已发（右 {s1} / 左 {s2}）——与臂动作并行，"
          f"反馈滞后 ~6.3s 由动作期重叠，达标判定处核对")

if not args.skip_zero:
    WATCH.pause()
    input("\n⚠ 段①：move_whole_body_joint_zero（SDK 碰撞检查零位）"
          + ("+ 夹爪闭合（并行）" if args.grip else "") + "。急停就绪回车，"
          "Ctrl-C 中止...")
    WATCH.resume()
    if args.grip:
        fire_grip_close()
    st = motion.move_whole_body_joint_zero(is_blocking=True,
                                           leg_head_speed_rad_s=0.2,
                                           leg_head_timeout_s=30.0)
    print(f"move_whole_body_joint_zero → {st}")
    cur = read23(robot)

WATCH.pause()
input(f"\n⚠ 段②：双臂+头" + ("+躯干腿" if not args.skip_leg else "")
      + f" → 数据集均值位（臂/头 {args.speed} rad/s，腿 0.2 rad/s）。"
      "腿动会改变站高/前倾，确认底盘平衡在位。急停就绪回车...")
WATCH.resume()
if args.grip and args.skip_zero:
    fire_grip_close()   # 跳过段① 时在段② 前发，闭合仍与臂动作并行
arm_names = RIGHT + LEFT + HEAD   # 名字与目标值逐位配对，顺序跟数据集维序
arm_tgt = [TARGET[i] for i in IDX["right_arm"] + IDX["left_arm"] + IDX["head"]]
st = robot.set_joint_positions(arm_tgt, joint_names=arm_names,
                               is_blocking=True, speed_rad_s=args.speed,
                               timeout_s=45.0)
print(f"set_joint_positions(臂+头) → {st}")
time.sleep(1.0)

# ── 段②b：躯干/腿 → state.mean（站高/前倾对齐采集起点）──
if not args.skip_leg:
    leg_tgt = [TARGET[i] for i in IDX["leg"]]
    st_l = robot.set_joint_positions(leg_tgt, joint_names=LEG,
                                     is_blocking=True, speed_rad_s=0.2,
                                     timeout_s=45.0)
    print(f"set_joint_positions(躯干/腿) → {st_l}")
    time.sleep(1.0)

# ── 达标判定：归一化 [-1,1] 外维数 ──
cur = read23(robot)
report(cur)
import numpy as np  # noqa: E402
st_n = normalize_state(np.array(cur, dtype=np.float32), ns)
out_dims = [f"dim{i}({STATE_NAMES[i]})" for i in range(23) if abs(st_n[i]) > 1.0]
arms_out = [d for d in out_dims if int(d[3:d.index("(")]) < 16]   # 臂+夹爪段 = dim0-15
print(f"\n== 达标判定 ==\n[-1,1] 外 {len(out_dims)} 维: {out_dims or '无'}")
print(f"双臂维(0-7,8-15) 外 {len(arms_out)} 维 → "
      + ("✅ 达标，可跑推理" if not arms_out else "⚠ 仍偏，检查臂段是否到位"))
print("（leg/head 窄维亮灯属传感器噪声量级， Part B 已知）")
leg_dev = max(abs(cur[IDX['leg'][k]] - TARGET[IDX['leg'][k]]) for k in range(5))
print(f"躯干/腿最大偏差 {leg_dev:.3f} rad"
      + ("（已下发对齐；偏大=未到位，查上方状态）" if not args.skip_leg
         else "（--skip-leg 未下发，仅对照）"))
if args.grip:
    ok_g = cur[7] <= 10 and cur[15] <= 10
    print(f"夹爪闭合核对: 右 {cur[7]:.1f}% / 左 {cur[15]:.1f}%（目标 0%）→ "
          + ("✅" if ok_g else "⚠ 未收到位，查夹爪或重跑（--grip 已含在段① 并行发出）"))

robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
print("SDK 已关闭")
WATCH.restore()
os._exit(0)   # SDK 残留线程，干净退出（os._exit 跳过 atexit，先还原终端）
