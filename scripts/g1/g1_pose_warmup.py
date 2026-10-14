#!/usr/bin/env python
"""G1 姿态预热：从任意当前姿态 → 数据集工作姿态（推理前置步骤）。

背景：pi0.5 的 state 走 256-bin 离散化进 prompt，姿态偏离采集分布时
归一化越界 → bin 打满 → 模型看到物理极限假象，动作无语义（2026-09-22
真机首跑 [-1,1] 外 15/23 维即此问题）。

两段式（安全优先）：
  ① move_whole_body_joint_zero —— SDK 预定义零位，带碰撞检查（reset_pos.py
     已验证用法），先脱离收拢/折叠等大偏移姿态。2026-10-12 起**默认跳过**
     （规划器大偏移常 TIMEOUT；--no-skip-zero 显式执行）
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
      ~/holy/scripts/g1/g1_pose_warmup.py \
      [--ckpt ~/holy/models/pi05_g1_deploy] [--no-skip-zero] [--speed 0.15] [--grip]

中断：随时按 q 即退（含 SDK 阻塞运动中——Ctrl-C 会被阻塞 C++ 调用推迟，
q 走独立监听线程不受限）；需立即断运动拍物理急停。
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "inference"))  # noqa: E402  g1_config 在 inference/（同仓共享配置）
import g1_config   # noqa: E402  同目录共享配置（CLI > config/g1.toml > 内置默认）

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None,
                help="部署目录（默认 config [run].ckpt；读 norm_stats.json 的 state.mean 作目标）")
ap.add_argument("--skip-zero", action="store_true", default=None,
                help="跳过零位段（2026-10-12 起默认即跳过，config [warmup].skip_zero）")
ap.add_argument("--no-skip-zero", action="store_true",
                help="显式执行段①零位（覆盖默认跳过；折叠/收拢等大偏移姿态先"
                     "脱离时用——零位带碰撞检查，段② 直控没有）")
ap.add_argument("--speed", type=float, default=None,
                help="臂/头关节速度上限 rad/s（默认 0.15；config [warmup].speed）")
ap.add_argument("--grip", action="store_true", default=None,
                help="闭合夹爪到数据集起点 0%%（width_min；需 manifest 标定；config [gripper].enabled）")
ap.add_argument("--no-grip", action="store_true",
                help="显式不闭合夹爪（覆盖 config 的 gripper.enabled=true）")
ap.add_argument("--init-settle", type=float, default=None,
                help="robot.init 后 settle 秒（默认 5.0；独立运行勿动）。"
                     "编排器快速复位路径传小值——前一段推理子进程刚释放控制器，"
                     "SDK 栈是热的；settle 不足的表现=start_controller 非 SUCCESS"
                     " 直接报错退出（响亮失败，无静默风险）")
ap.add_argument("--then-move", nargs=3, type=float, default=None,
                metavar=("X", "Y", "YAW_DEG"),
                help="达标后【同一 SDK 会话】底盘横移（编排器 reset+move 合并段；"
                     "独立运行勿用）：move_straight_to 相对位移 x/y 米"
                     "（base_link 系 x前/y左，odom 系免定位）+ 原地转角 deg；"
                     "与 --then-goto 互斥")
ap.add_argument("--then-goto", default=None, choices=["place", "pick"],
                help="同上但走 map 系绝对坐标直达（g1_warm_core."
                     "nav_goto_station：navigate_to_goal(collision_check="
                     "False)+0.5s 轮询判到达，需定位锁在）；与 --then-move "
                     "互斥。编排器默认发 --then-goto place")
ap.add_argument("--move-timeout", type=float, default=30.0,
                help="--then-move/--then-goto 的导航内部超时 s")
ap.add_argument("--skip-leg", action="store_true", default=None,
                help="跳过躯干/腿对齐（腿部动作改变站高/前倾，默认对齐；config [warmup].skip_leg）")
ap.add_argument("--pose-file", default=None,
                help="外部位姿 json 覆盖（23 维 start_pose，g1_pose_extract.py 产物；"
                     "config [warmup].pose_file）")
ap.add_argument("--config", default=g1_config.DEFAULT_PATH,
                help="配置文件路径（优先级 CLI > config > 内置默认）")
args = ap.parse_args()
if args.then_move is not None and args.then_goto is not None:
    ap.error("--then-move 与 --then-goto 互斥，二选一")
g1_config.apply(args, {
    "ckpt": ("run", "ckpt"),
    "speed": ("warmup", "speed"),
    "skip_zero": ("warmup", "skip_zero"),
    "skip_leg": ("warmup", "skip_leg"),
    "grip": ("gripper", "enabled"),
})
if args.no_grip:
    args.grip = False
if args.no_skip_zero:
    args.skip_zero = False   # 覆盖 config/BUILTIN 的默认跳过，显式执行段①
# pose_file 是可选键（BUILTIN 无兜底，不进 apply mapping，避免双双缺省即退出）
POSE_FILE = args.pose_file
if POSE_FILE is None:
    _cfg, _ = g1_config.load(args.config)
    POSE_FILE = _cfg.get("warmup", {}).get("pose_file")


class QuitWatcher:
    """后台线程监听键盘 q：任何阶段即时退出脚本。

    Ctrl-C 的 KeyboardInterrupt 只在 Python 字节码间隙触发，SDK 阻塞的
    C++ 运动调用（set_joint_positions/move_whole_body_joint_zero）期间
    按了也没反应——q 键线程绕开这一限制。物理急停仍是第一优先级。
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._old = None
        self._wake_r = None
        self.active = threading.Event()   # input() 提示符处 pause，防抢键
        if os.isatty(self.fd):
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)          # 单键即达，无需回车
            atexit.register(self.restore)
            r, w = os.pipe()                # 信号直通管道：C 层收信号即写字节，
            os.set_blocking(w, False)       # 主线程卡死在 SDK C++ 调用也能退出
            signal.set_wakeup_fd(w)
            signal.signal(signal.SIGINT, lambda *_: None)     # 退出走管道，别靠
            signal.signal(signal.SIGTERM, lambda *_: None)    # 会被推迟的异常
            self._wake_r = r
            self.active.set()
            threading.Thread(target=self._loop, daemon=True).start()
            print("（运动等待期可按 q 或 Ctrl-C 退出；提示符处用回车；紧急停止拍物理急停）")

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
            try:
                fds = [self._wake_r] if self._wake_r is not None else []
                if self.active.is_set():
                    fds.append(sys.stdin)
                if not fds or not select.select(fds, [], [], 0.2)[0]:
                    continue
                if (self._wake_r is not None
                        and select.select([self._wake_r], [], [], 0)[0]):
                    os.read(self._wake_r, 1)
                    print("\n⛔ 收到中断信号 —— 立即退出脚本"
                          "（已下发目标可能仍在限速执行；需立即断运动请拍急停）")
                    self.restore()
                    os._exit(130)   # os._exit 跳过 atexit，终端状态先手动还原
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出脚本（已下发目标可能仍在"
                          "限速执行中；需立即断运动请拍急停）")
                    self.restore()
                    os._exit(2)   # os._exit 跳过 atexit，终端状态先手动还原
            except Exception:
                self.restore()
                os._exit(3)   # 监听线程死了比静默更危险：宁可误退不可失控


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

# ── 目标姿态：优先 task 起始位姿（episode_start_task0.json），兜底 state.mean ──
# ⚠ 2026-09-23 实证：state.mean 是 task0（夹爪 0%/j7=-1.54，98 条）与 task1
# （夹爪 33%/高位，101 条）两组起始位的平均——不对应任何任务起点，臂维差达
# 0.83 rad；模型训练轨迹全部从各 task 起始位姿出发。本文件由
# `分析脚本`从数据集 task_index=0 的 98 条起始帧取中位数生成。
CKPT = pathlib.Path(args.ckpt)
ns_p = CKPT / "norm_stats.json"
# 位姿来源三级 + 常量 + 读关节/摆臂/达标/横移序列：全部在共享核心
# g1_warm_core（2026-10-13 折叠改造：warmup 与 run_g1_loop.py 折叠段共用，
# 改一处两处生效，勿在此重写序列）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g1_warm_core import (LEFT, RIGHT, HEAD, LEG, GRIP, STATE_NAMES, IDX,  # noqa: E402
                          load_target, send_warm, judge as _judge_core,
                          read23 as _read23_core, report as _report_core,
                          nav_straight_move, nav_goto_station)
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import (GalbotRobot, GalbotMotion, G1JointGroup,  # noqa: E402
                           GalbotNavigation, G1ControllerName)

TARGET, ns, MODE = load_target(CKPT, POSE_FILE)

# 夹爪标定（manifest；有则读数换算 0~100%，与数据集 state 单位对齐）
mf_p = CKPT / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
_gcal = mf.get("gripper", {})
GRIP_WMIN, GRIP_WMAX = _gcal.get("width_min"), _gcal.get("width_max")
if args.grip and (GRIP_WMIN is None or GRIP_WMAX is None):
    raise SystemExit("--grip 需要 manifest 标定；先跑 g1_ckpt_prep "
                     "--grip-wmin/--grip-wmax（2026-09-23 实测 0.0005/0.1200 m）")


def _die(msg, code=1):
    """失败出口：SDK 活着时裸 raise=解释器退场撞残留线程 → SIGSEGV
    （rc=-11 掩盖真实原因，2026-10-13 八跑实录）——先关 SDK 再 os._exit。"""
    print(msg, flush=True)
    r = globals().get("robot")
    try:
        if r is not None:
            r.request_shutdown(); r.wait_for_shutdown(); r.destroy()
            print("SDK 已关闭", flush=True)
    except Exception:
        pass
    try:
        WATCH.restore()
    except Exception:
        pass
    os._exit(code)


def read23(robot):
    return _read23_core(robot, STATE_NAMES, GRIP_WMIN, GRIP_WMAX, _die)


def report(cur):
    _report_core(cur, TARGET, STATE_NAMES)


print(f"目标姿态已就绪（norm_mode={MODE}）——来源见上方 [warm] 行")
cur = None

print("\n== ① SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
# GalbotMotion 惰性创建：仅段①零位需要（规划器 move_whole_body_joint_zero）。
# 默认 skip_zero 复位路径只走 set_joint_positions 直控，不碰 motion.init
# ——省初始化时间，也绕开 FK 脱机挂死面（2026-09 实录）。5s settle 保留：
# 与 run_g1_loop 同款，双脚本日常实证，勿删。
motion = None
if not args.skip_zero:
    motion = GalbotMotion()
if not robot.init():
    _die("robot.init 失败")
if motion is not None:
    motion.init()
time.sleep(5.0 if args.init_settle is None else args.init_settle)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")

cur = read23(robot)
report(cur)

_grip_sent = False


def _grip_width(pct):
    """数据集 0~100% 开度 → SDK 宽度（米），按 manifest 标定线性换算。"""
    return GRIP_WMIN + float(pct) / 100.0 * (GRIP_WMAX - GRIP_WMIN)


def fire_grip_close():
    """夹爪对齐目标位姿开度（非阻塞一次性发出），与臂动作并行；末端统一核对。

    目标取位姿文件的夹爪维（dim7/15，0~100%）——2026-09-29 only_place 实测
    101 轨全部从 ≈33% 开度起步（places 任务持物起点），闭合到 0% 会错开
    训练分布入口；旧 pick_place 数据集 0% 起步，位姿文件里即 0，行为不变。
    """
    global _grip_sent
    if _grip_sent:          # 零位重试路径不再重复发（已到位状态下重发无害，省之）
        return
    _grip_sent = True
    wr, wl = _grip_width(TARGET[7]), _grip_width(TARGET[15])
    s1 = robot.set_gripper_command(G1JointGroup.right_gripper,
                                   wr, 0.05, 30, False)
    s2 = robot.set_gripper_command(G1JointGroup.left_gripper,
                                   wl, 0.05, 30, False)
    print(f"夹爪对齐命令已发（右 → {TARGET[7]:.1f}% = {wr:.4f} m / "
          f"左 → {TARGET[15]:.1f}% = {wl:.4f} m）——与臂动作并行，"
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
    # 臂走规划器+碰撞检查（leg/head 才是直控），伸直/大偏移姿态下规划
    # 可能内部超时（TIMEOUT，臂停中途）——重试常能过，或跳零位走段②直控
    if not str(st).startswith("MotionStatus.SUCCESS"):
        WATCH.pause()
        ans = input("⚠ 零位未完全到位。回车=重试零位 / 输入 s=跳过零位"
                    "（段② 关节直控到工作位，不经规划器、无碰撞检查，"
                    "先清空臂前空间）/ Ctrl-C=退出：").strip().lower()
        WATCH.resume()
        if ans == "s":
            args.skip_zero = True
        else:
            if args.grip:
                fire_grip_close()
            st = motion.move_whole_body_joint_zero(is_blocking=True,
                                                   leg_head_speed_rad_s=0.2,
                                                   leg_head_timeout_s=30.0)
            print(f"move_whole_body_joint_zero(重试) → {st}")
    if not args.skip_zero:
        cur = read23(robot)

WATCH.pause()
input(f"\n⚠ 段②：双臂+头" + ("+躯干腿" if not args.skip_leg else "")
      + f" → 目标位姿（臂/头 {args.speed} rad/s，腿 0.2 rad/s）。"
      "腿动会改变站高/前倾，确认底盘平衡在位。急停就绪回车...")
WATCH.resume()
if args.grip and args.skip_zero:
    fire_grip_close()   # 跳过段① 时在段② 前发，闭合仍与臂动作并行
send_warm(robot, TARGET, args.speed, skip_leg=bool(args.skip_leg))

# ── 达标判定：归一化 [-1,1] 外维数（共享核心版）──
cur = read23(robot)
report(cur)
arms_out = _judge_core(cur, TARGET, ns, skip_leg=bool(args.skip_leg))
if args.grip:
    ok_g = abs(cur[7] - TARGET[7]) <= 10 and abs(cur[15] - TARGET[15]) <= 10
    print(f"夹爪核对: 右 {cur[7]:.1f}%（目标 {TARGET[7]:.1f}%）/ "
          f"左 {cur[15]:.1f}%（目标 {TARGET[15]:.1f}%，±10% 容差）→ "
          + ("✅" if ok_g else "⚠ 未收到位，查夹爪或重跑（--grip 已含在段① 并行发出）"))

if args.then_move is not None:
    # ③ 同会话底盘横移（官方 GalbotNavigation.move_straight_to 四步，
    # 契约细节见 g1_warm_core.nav_straight_move 文档）
    sx, sy, syaw = args.then_move
    print(f"\n== ③ 同会话横移 x={sx:+.3f} y={sy:+.3f} m yaw={syaw:+.2f}°"
          f"（base_link 系 y+=左）timeout {args.move_timeout:.0f}s ==", flush=True)
    try:
        ok_m, st_m = nav_straight_move(robot, sx, sy, syaw, args.move_timeout)
    except Exception as exc:       # SDK 异常走 _die 干净退出，防 teardown 段错误掩盖 rc
        _die(f"横移 API 异常: {exc!r}")
    if not ok_m:
        _die(f"横移失败 status={st_m}")   # 复位已成、横移失败：rc=1 编排器中止

if args.then_goto is not None:
    # ③′ 同会话底盘绝对坐标直达（10-14 编排器 goto 模式，契约见
    # g1_warm_core.nav_goto_station：is_localized 闸+WAIT_INITIALIZED
    # 限时重试+0.5s 轮询 check_goal_arrival）
    print(f"\n== ③′ 同会话 goto 工位 {args.then_goto} "
          f"timeout {args.move_timeout:.0f}s ==", flush=True)
    try:
        ok_m, st_m = nav_goto_station(robot, args.then_goto, args.move_timeout)
    except Exception as exc:
        _die(f"goto API 异常: {exc!r}")
    if not ok_m:
        _die(f"goto 失败 status={st_m}")   # 复位已成、横移失败：rc=1 编排器中止

robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
print("SDK 已关闭")
WATCH.restore()
os._exit(0)   # SDK 残留线程，干净退出（os._exit 跳过 atexit，先还原终端）
