"""G1 机体专属件：关节表/state 拼装/取图/爪下发/导航搁置（common 收编版 10-31 步2c）。

原文=run_g1_loop_traj.py 逐行，搬运不改逻辑。S1 对应件住 common/bodies/s1.py；
letterbox224/head_training_crop 双机同件已收编 common/bodies/vision.py（与 G1
原文逐字节相等已证），本模块 import 复用。

跨模块状态走模块属性回填：`args`/爪标定/相机表由入口 init 后一次回填；爪状态
dict/计数器（_GRIP_EMA/_GRIPJ_* 等）以本模块为唯一持有地（send_grip* 经
`global` 改写的 int 计数器住这里，入口零直读）——

    import common.bodies.g1 as B
    B.args, B.GRIP, B.GRIP_SS = args, GRIP, GRIP_SS
    B.GRIP_WMIN, B.GRIP_WMAX = GRIP_WMIN, GRIP_WMAX   # manifest 爪行程标定
    B.CAM_MAP, B.VIEWS = CAM_MAP, VIEWS               # manifest 相机表
"""
import os
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

try:    # 离线纯函数复用零 SDK；取图/爪下发路径须真 SDK（入口同款 import）
    from galbot_sdk.g1 import G1JointGroup, JointCommand, SensorType
except ImportError:
    G1JointGroup = JointCommand = SensorType = None

from common.bodies.vision import head_training_crop, letterbox224  # noqa: E402

# ── 回填占位（入口 init 后赋值；先于此处任何函数调用）──
args = None        # argparse Namespace（do_exec/grip_*/head_crop 全族）
GRIP = None        # 爪标定/状态 dict（None=无爪）；send_grip* 原地改写
GRIP_SS = None     # --grip-ss (开阈, 合阈)；None=关（旧 chg 门行为）
GRIP_WMIN = None   # 爪行程标定下限（米；manifest gripper 段）
GRIP_WMAX = None   # 爪行程标定上限（米；manifest gripper 段）
CAM_MAP = None     # manifest 相机表 {模型键: SensorType 名}
VIEWS = 3          # 使用的相机数（CAM_MAP 前 VIEWS 个）

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


LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT            # 与 chunk[0:7]+chunk[8:15] 逐位配对
STATE_NAMES = (RIGHT + ["right_gripper_joint1"]
               + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])
GRIP_IDX = (7, 15)                  # state 里的夹爪维：SDK 米 → 数据集 0~100%
ARM_SLICE = list(range(0, 7)) + list(range(8, 15))   # state 里的臂维（0-6 右 / 8-14 左）


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
    """--grip-state-cmd：state 夹爪维(7/15)改喂指令值，持物段例外（10-12 修正）。

    数据集 state 爪维语义=跟随段记指令值——喂物理回读=分布漂移（模型全程
    看到滞后/爬坡中的爪宽）。⚠「SDK 回读滞后 6.3-8s」旧说已被 10-13 标定
    翻案（grip_speed_bench 四程：回读时戳新鲜 p50 10ms，伺服守 0.15 m/s）；
    本旗标的行为必要性（10-05/10-08 四跑实证）不受影响。

    ⚠ 持物签名修正（2026-10-12 帧级实锤）：数据集「抓到物体」语义=指令 0%
    + state 被物体撑在 ~33.5%（ep0 t5.1s 起；40 集持物段 state 中位 33.5、
    p10-p90=33.0-33.5 恒定）——模型靠该组合识别「已持物→后撤→交接」。
    闭合后若照喂指令 0% = 把签名抹成「空爪闭紧」（预抓取/释放后分布），
    交接序列永不触发（loop_20261012_165104 实录：L t+5.3s 抓住、模型不后撤、
    R 爪全程 0%·hold）。故分三段：
      开爪/跟随段（指令 >10%）：喂指令值（含回读不可信的整个滞后窗）；
      持物段（指令 <10% 且该爪历史指令峰值 >50% = 开→合全程走完）：
        喂持物签名 33.5（数据集恒定值；SDK 回读在交接关键窗内仍处滞后，弃用）；
      未开过爪的闭合（峰值 ≤50%，如起步双爪 0%）：喂指令值（=空爪闭合，正确）。
    已知限制：释放后（指令仍 0）滞留持物签名——pick 线释放后任务即转 R 臂，
    state 不再参与关键决策，首版接受；空抓（夹空）时也会误报持物，行为=
    提前交接 R，属可接受恢复路径。
    """
    if GRIP is None or not args.grip_state_cmd:
        return st_raw
    st = np.array(st_raw, dtype=np.float32)
    for name, dim in GRIP["names"]:
        sent = GRIP["sent"].get(name)
        if sent is None:
            continue
        peak = GRIP["sent_peak"].get(name, 0.0)
        GRIP["sent_peak"][name] = max(peak, sent)
        if sent < 10.0 and GRIP["sent_peak"][name] > 50.0:
            st[dim] = 33.5
        else:
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
        _lb = letterbox224(img)
        if args.head_crop and key == "image":
            _lb = head_training_crop(_lb)   # 头图训练裁剪放大（仅头相机）
        out[key] = np.ascontiguousarray(cv2.cvtColor(_lb, cv2.COLOR_BGR2RGB))
    return out


def send_grip(robot, chunk_row):
    """chunk 行 dim7/dim15（0~100%）→ 标定宽度下发（超阈值才发，非阻塞）。

    ⚠ 10-21 起默认路径是 send_grip_joint（--grip-path joint=lerobot 采集
    语义）；本函数=set_gripper_command 行程控制器路【逃生门/复现对照】，
    下列各门只在此路适用（该路 onset=0.12s+行程/速度+吞令=训练外注入行为，
    P6 三格破案在案）。

    只发不查不等待（拖死控制环）；终态核对放循环结束后。回读新鲜（10-13
    标定 p50 10ms），实际下发拍顺带采样一次物理实宽进日志（[实 X.Xmm]），
    爪的物理时序证据逐 send 可见。已知物理动态（同标定）：起步死区 ~0.9s +
    0.15 m/s 帽（满程 ~0.8s）。
    ⚠ 111901 终审：每条新指令把行程从头重启（死区重吃）——逐拍补发的指令流
    p50 间隙 2-3 拍，死区永不耗尽 → 110 send 全 SUCCESS 实宽全程 0.5mm 冻结；
    q 停流静→最后一条 1.5s 走满（「q 后突然张开」全案闭合）。故
    --grip-ss 状态机=少发（每相 1-2 条、每条 ≥2s 静默窗）；无 ss 时退回旧
    chg 门（2% 变化即发，111901 证明该模式爪会冻结——保留只为对照）。
    --grip-send-every N=定时静默制（2026-10-20 定稿）：chg 死区门整体旁路，
    每 N 个机会拍无条件发当前 EMA 值一条、其余拍纯静默 → 爪在静默窗内跑完
    行程；N=4→≈7.5Hz/静默 75%。纯去掉死区（≡chg=0）=143831 实证完全冻结，
    勿用。⚠ 10-20 晚定律终版（174428）补判决：拍隙 <0.9s 死区的周期拍从静止
    永起步不了（合步2 N=4→0.27s=永冻），此路要真静默须 N≥18。
    --grip-min-interval N=事件门（2026-10-20 定律终版，三选一门之一）：距上次
    实发 ≥N 秒才发（固件 onset=0.12s+行程/速度，0.15 全行程 0.91s < N=1.2 →
    下一条来时上一条必已在途，吞令结构性不可能）；间隔未满仅「在途(|v|>0.02)
    且意图差≥3%」豁免——在途同向/反向均平滑零罚（P2 174428 满速零凹陷滑到底；
    P4 183232 反向掉头 ~0.2s 不碰零；10-16「反转重启律」随之改写：杀手=死区
    密发，方向只是伴随特征）。
    --grip-preclose V=预合爪爬行【185317 极限梯判死勿启用】：固件速度下限
    0.03 m/s（0.01 指令实测执行 0.03）且 onset=0.12s+行程/速度——爬速 0.03
    全行程起步要 ~4.1s，慢速反把起步拖晚，「低速吸收死区」前提不成立。
    """
    if GRIP is None:
        return None
    if not args.do_exec:
        return "dry（不下发）"
    parts = []
    now = time.monotonic()
    # --grip-send-every：节流判定每调用一次（=一个机会拍），R/L 共用同一判定
    # 保持同拍配对；合步模式只组末拍调用，节奏自然跟随。chg 门在节流模式下
    # 整体旁路（死区逻辑去掉，见函数 docstring 的剂量-响应证据）。
    global _GRIP_BEAT
    _pace = int(args.grip_send_every) > 0
    _send_beat = False
    if _pace:
        _send_beat = (_GRIP_BEAT % int(args.grip_send_every)) == 0
        _GRIP_BEAT += 1
    _minint = float(args.grip_min_interval or 0.0)
    _pre = float(args.grip_preclose or 0.0)
    for name, dim in GRIP["names"]:
        p = float(np.clip(chunk_row[dim], 0.0, 100.0))
        if args.grip_ema_alpha > 0:
            prev = _GRIP_EMA.get(name)
            p = p if prev is None else \
                args.grip_ema_alpha * p + (1.0 - args.grip_ema_alpha) * prev
            _GRIP_EMA[name] = p
        last = GRIP["sent"].get(name)
        why = ""
        if GRIP_SS is not None:
            st0 = _GRIP_SS_STATE.get(name)
            t0, v0 = _GRIP_SS_SENT.get(name, (None, None))
            if st0 is None:                      # 首拍定态并发一次
                st0 = "hi" if p >= GRIP_SS[0] else "lo"
                _GRIP_SS_STATE[name] = st0
                why = "首拍"
            elif p >= GRIP_SS[0] and st0 != "hi":
                _GRIP_SS_STATE[name] = "hi"
                why = "开阈"
            elif p <= GRIP_SS[1] and st0 != "lo":
                _GRIP_SS_STATE[name] = "lo"
                why = "合阈"
            elif (t0 is not None and abs(p - v0) >= 15.0
                  and now - t0 >= 2.0):
                why = "漂移补发"                 # 中间宽度意图，限频 0.5Hz
            if not why:                          # 无翻转未到补发条件=一律不发
                parts.append(f"{name[0].upper()} {p:.1f}%·hold")
                continue
        elif _minint > 0:
            # 事件门（定律终版）：间隔窗=静默保证起步；在途同向豁免=免费微调。
            # 意图没动（<max(chg,3%)）连间隔窗也不发——零travel重发只会白白
            # 重置静默钟、推迟下一条真令。
            lt = _GRIP_SENT_AT.get(name)
            moved = last is not None and \
                abs(p - last) >= max(float(GRIP["chg"] or 2.0), 3.0)
            if lt is None:
                why = "首拍"
            elif not moved:
                parts.append(f"{name[0].upper()} {p:.1f}%·hold")
                continue
            elif now - lt >= _minint:
                why = "间隔窗"
            else:
                _gs = robot.get_gripper_state(getattr(G1JointGroup, name))
                if _gs is not None and abs(_gs.velocity) > 0.02:
                    why = "在途"                # P2/P4 双实锤：在途同向/反向均
                                                # 平滑零罚（183232 掉头仅 ~0.2s
                                                # 减速-再加速，不碰零）
            if not why:
                parts.append(f"{name[0].upper()} {p:.1f}%·hold")
                continue
        elif not _pace and last is not None and abs(p - last) < GRIP["chg"]:
            parts.append(f"{name[0].upper()} {p:.1f}%·hold")
            continue
        if _pace and not _send_beat:         # 节流窗内的拍一律不发（ss 放行也压住）
            parts.append(f"{name[0].upper()} {p:.1f}%·hold")
            continue
        # --grip-preclose：意图下行（合爪）且未平稳 → 爬速下发；平稳回正常速收敛
        spd = GRIP["speed"]
        _tag = ""
        if _pre > 0:
            h = _GRIP_PRE_HIST.setdefault(name, [])
            h.append((now, p))
            while h and now - h[0][0] > 0.75:
                h.pop(0)
            if len(h) >= 2 and p < h[0][1] - 3.0:   # 净降 >3 点=真下行（±2 平台
                spd = _pre                           # 摆动不重燃爬速，EMA 流常见）
                _tag = " 爬"
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        st = robot.set_gripper_command(getattr(G1JointGroup, name),
                                       w, spd, GRIP["effort"], False)
        GRIP["sent"][name] = p
        _GRIP_SENT_AT[name] = now
        if GRIP_SS is not None:
            _GRIP_SS_SENT[name] = (now, p)
        _gs = robot.get_gripper_state(getattr(G1JointGroup, name))
        # eff 回读单位未知（144325 实证：饱和轨 100、全开 114mm 也 100，
        # 头文件 "Newtons" 被推翻）——只当负载/堵转指示，勿按牛顿判读
        _phys = (f" 实{_gs.width * 1000:.1f}mm eff{_gs.effort:.1f}"
                 if _gs is not None else " 实?")
        parts.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm@{spd:.2f}{_tag}"
                     f"{_phys} {str(st).replace('ControlStatus.', '')}"
                     + (f"({why})" if why else ""))
    return "  ".join(parts)


# --grip-path joint：爪维走关节固件路径的独立 EMA 状态（与 cmd 路径 _GRIP_EMA
# 分开，A/B 切路径不串状态）
_GRIPJ_EMA = {}
# --grip-hold-on-stall 憋死保持状态（每爪独立）：on=保持中；hist=实宽平台窗；
# enters=累计进入次数（汇总打印）；tgt=冻结目标；floor=进入时实宽（外松上限）；
# rail=eff 贴轨连续拍数（自动外松节拍器）；relax_n=累计外松步数（汇总打印）；
# quiet=距上次武装的已发拍数（静默制节拍器）；silent_n=累计静默拍数（汇总打印）
_GRIPJ_HOLD = {}
# --gripj-every 爪通道节拍器（全局连续拍，不随换块复位）
_GRIPJ_TX = 0


def send_grip_joint(robot, chunk_row, tfs_s, frame=None):
    """--grip-path joint：夹爪 dim7/15 并回 set_joint_commands 流（10-21 P6 破案）。

    同事 infer_policy / lerobot 采集语义原样：position=width_m、
    velocity=速度帽（P6j 实测 0.15→0.1500 被尊重，非 pursuit 追踪）、
    effort=10（同事默认；非 set_gripper_command 路的 30）、tfs 与臂同参。
    「onset=0.12s+行程/速度＋吞令」整条定律属 set_gripper_command 端点
    （P6 三格实证：关节路径 onset≈0.14s 纯 L 与行程无关、密流照跟不冻），
    故 chg/min-interval/send-every/ss/preclose 各门在此全部旁路——每发送拍
    无条件发当前 EMA 值（密流在该路径免费，P6j_dense 实证）。EMA 照带
    （独立状态 _GRIPJ_EMA）。双爪一包：groups=[right_gripper,left_gripper]
    顺序=命令序（P6 单组已验组写安全；group 模式读的顺序坑不适用于写）。

    GRIP["sent"] 照常更新 → --grip-state-cmd / 遥测 / sent_peak 持物签名
    无感知路径切换。

    --grip-hold-on-stall（2026-10-21 力矩碰撞案）：模型按训练语义持物段持续
    发全闭指令，位置伺服每拍重发=斜坡重启反复撞进物体。检测「指令≤实宽−gap
    且实宽平台」→ 改发 进入时实宽−margin 冻结目标@effort_hold（钳位保持；
    ⚠勿逐拍跟随实宽=棘轮碾进，144325 首跑实证）；指令上穿实宽+exit
    （想开爪）→ 恢复流式并清平台史。只改物理下发，不改指令记账。
    自动外松（--grip-hold-relax-mm，095207 定案）：钳位后稳态误差=margin 仍
    使伺服满驱（eff 恒贴饱和轨 100，「有差就追」=位置伺服天性，收 margin 无效
    ——1mm 时 156/159 拍仍贴轨可闻声）。SDK 无阻抗模式，执行侧导纳仿真：eff
    贴轨持续 6 拍→目标外松一步，直到 eff 掉头或目标=进入时实宽（零误差停追）。
    单向只松不紧=无棘轮。
    真静默制（--grip-hold-quiet-n，101418 定案）：外松到零误差后 eff 仍恒贴轨
    100+响（112 拍无衰减）——每拍重发冻结目标虽逐位相同，但 10-20 爪定律=每条
    新令打断在途从头重来，密发把爪永远钉在起步段 settle 不了。进 hold/外松步后
    只发 N 拍即停发该爪（固件执行最后一条收工），读回/退出/外松评估照常。
    --gripj-every 爪通道节流（10-22 合步 A/B 判读）：仅物理下发被节流，
    状态机照常每拍跑（窗口墙钟不变）。
    """
    if GRIP is None:
        return None
    if not args.do_exec:
        return "dry（不下发）"
    global _GRIPJ_TX
    _tx = True
    if args.gripj_every > 1:
        _tx = _GRIPJ_TX % args.gripj_every == 0
        _GRIPJ_TX += 1
    parts = []
    cmds, groups = [], []
    for _ni, (name, dim) in enumerate(GRIP["names"]):
        if frame is not None:
            # 爪随臂宏步管线（2026-10-26）：帧值已在上游行准备中 EMA+插帧，
            # 此处只钳位——二次 EMA 会双滤波
            p = float(np.clip(frame[_ni], 0.0, 100.0))
        else:
            p = float(np.clip(chunk_row[dim], 0.0, 100.0))
            if args.grip_ema_alpha > 0:
                prev = _GRIPJ_EMA.get(name)
                p = p if prev is None else \
                    args.grip_ema_alpha * p + (1.0 - args.grip_ema_alpha) * prev
                _GRIPJ_EMA[name] = p
        GRIP["sent"][name] = p   # 记模型意图（state-cmd/sent_peak 语义不变；
                                 # hold 只改物理下发，不改指令记账）
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        _gs = robot.get_gripper_state(getattr(G1JointGroup, name))
        w_sent, eff_sent, vel_sent, _tag = w, 10.0, GRIP["speed"], ""
        _skip = False
        if args.grip_hold_on_stall and _gs is not None:
            phys = _gs.width
            h = _GRIPJ_HOLD.setdefault(name, {"on": False, "hist": [],
                                              "enters": 0, "tgt": None,
                                              "floor": None,
                                              "rail": 0, "relax_n": 0,
                                              "quiet": 0, "silent_n": 0})
            # 双史窗口：(实宽, 本拍指令是否够深)。进入须「实宽平台」∧「深指令
            # 持续满窗」——150603 死锁教训：爪静停开位时模型刚下合令，只看
            # 当拍深指令会误进（onset 未到爪还没动）→ 冻结在开位=意图永不上
            # 穿 exit 门、爪全程不闭合。满窗深指令=「给了 onset+行程时间仍到
            # 不了」才算真憋死；自由爪满窗前早动了，平台破即不进。
            h["hist"].append((phys,
                              w <= phys - args.grip_hold_gap_mm / 1000.0))
            _win = max(2, int(args.grip_hold_window))
            while len(h["hist"]) > _win:
                del h["hist"][:-_win]
            _plateau = (len(h["hist"]) >= _win and
                        (max(p for p, _ in h["hist"])
                         - min(p for p, _ in h["hist"])) * 1000.0
                        <= args.grip_hold_plateau_mm)
            _deep_all = all(d for _, d in h["hist"])
            if h["on"]:
                if w >= phys + args.grip_hold_exit_mm / 1000.0:
                    h["on"] = False
                    h["hist"].clear()   # 退出清史：再进须新平台（防近阈闪断）
                    h["tgt"] = None
                    h["floor"] = None
                    h["rail"] = 0
                    h["quiet"] = 0
                    _tag = " HOLD退出"
            elif _plateau and _deep_all:
                h["on"] = True
                h["enters"] += 1
                # 进入时冻结目标（144325 教训：逐拍跟随 实宽−margin=棘轮
                # 碾进；冻结后刚性物同样钉死、软物最多压进 margin 一次）
                h["tgt"] = max(0.0,
                               phys - args.grip_hold_margin_mm / 1000.0)
                h["floor"] = phys   # 外松上限=进入时实宽（零误差=停追静音）
                h["quiet"] = 0      # 重新武装：进 hold 先发满 quiet-n 拍
                _tag = " HOLD进入"
            if h["on"]:
                if args.grip_hold_relax_mm > 0 and h["floor"] is not None:
                    # eff 闭环自动外松（执行侧导纳仿真，095207 实证：1mm 误差
                    # 仍伺服满驱 eff 恒贴轨=「有差就追」天性，收 margin 治不了
                    # 声）。贴轨满 6 拍→目标外移一步；eff 掉头或到进入时实宽
                    # （零误差）即停。单向只松不紧=无棘轮回旋余地。先于
                    # w_sent 赋值：步进当拍即按新目标下发。
                    if _gs.effort >= args.grip_hold_relax_eff:
                        h["rail"] += 1
                    else:
                        h["rail"] = 0
                    if h["rail"] >= 6 and h["tgt"] < h["floor"] - 1e-6:
                        h["tgt"] = min(h["floor"],
                                       h["tgt"] + args.grip_hold_relax_mm
                                       / 1000.0)
                        h["rail"] = 0
                        h["relax_n"] += 1
                        h["quiet"] = 0   # 外松步重新武装：发满 quiet-n 拍
                        _tag = f" HOLD松{h['tgt'] * 1000:.1f}"
                # 目标恒定=稳态误差恒为 margin（观测 093335：误差即伺服持续
                # 内顶驱动力，故 margin 1mm+保持段速度帽放软双管压声）
                w_sent = h["tgt"]
                if args.grip_effort_hold > 0:
                    eff_sent = args.grip_effort_hold
                if args.grip_hold_vel > 0:
                    vel_sent = min(args.grip_hold_vel, GRIP["speed"])
                _tag = _tag or " HOLD"
                if args.grip_hold_quiet_n > 0:
                    # 静默制（101418 定案）：外松到零误差 eff 仍恒贴轨+响——
                    # 每拍重发虽逐位相同，但 10-20 爪定律=每条新令打断在途
                    # 从头重来，密发把爪永远钉在起步段 settle 不了。发满
                    # quiet-n 拍后本爪退出指令包=真静默，让最后一条走完收工；
                    # 读回/退出/外松评估照常（不依赖发送）。
                    if h["quiet"] >= args.grip_hold_quiet_n:
                        _skip = True
                        h["silent_n"] += 1
                        _tag = " HOLD静"
                    else:
                        h["quiet"] += 1
        c = JointCommand()
        c.position = float(w_sent)
        c.velocity = float(vel_sent)        # 速度帽语义：恒速段被钳在此值
        c.effort = float(eff_sent)
        if not _skip and _tx:   # 静默制/节流：该爪本拍退出指令包（读回照常）
            cmds.append(c)
            groups.append(getattr(G1JointGroup, name))
        # eff 回读单位未知（144325 实证：饱和轨 100、全开 114mm 也 100，
        # 头文件 "Newtons" 被推翻）——只当负载/堵转指示，勿按牛顿判读
        _phys = (f" 实{_gs.width * 1000:.1f}mm eff{_gs.effort:.1f}"
                 if _gs is not None else " 实?")
        if not _tx and not _skip:
            _tag = f"{_tag} 跳" if _tag else " 跳"
        parts.append(f"{name[0].upper()} {p:.1f}%→{w_sent * 1000:.0f}mm@j"
                     f"{_phys}{_tag}")
    if not cmds:    # 双爪皆静默：本拍无指令包
        return "  ".join(parts) + " j静"
    st = robot.set_joint_commands(cmds, joint_groups=groups,
                                  time_from_start_s=tfs_s)
    ok = str(st).startswith("ControlStatus.SUCCESS")
    return "  ".join(parts) + (" jOK" if ok else
                               f" j{str(st).replace('ControlStatus.', '')}")


_GRIP_EMA = {}      # 爪指令 EMA 状态（--grip-ema-alpha>0 时启用；跨块连续，爪流
                    # 换块不弃尾——爪相位是低频意图，逐拍直通=物理包络被抖动打断）
                    # ⚠ GRIP_SS 的初值 None 放 argparse 段（≈300 行）——若在此处再
                    # 置 None 会把上面解析好的 --grip-ss 盖掉（125953 实录：banner
                    # 打了 50/35 但循环里 GRIP_SS 已被本行旧赋值抹成 None，静默退
                    # 回 chg 门=状态机整跑未生效）
_GRIP_SS_STATE = {}  # 状态机每爪当前态 'hi'/'lo'
_GRIP_SS_SENT = {}   # 状态机每爪 (上次实发时刻, 上次实发值)
_GRIP_BEAT = 0       # --grip-send-every 机会拍计数（send_grip 每调用 +1）
_GRIP_SENT_AT = {}   # 事件门每爪上次实发时刻（--grip-min-interval；send_grip 记账）
_GRIP_PRE_HIST = {}  # 预合爪每爪意图史 (t, p)，0.75s 窗（判「下行未平稳」用）
