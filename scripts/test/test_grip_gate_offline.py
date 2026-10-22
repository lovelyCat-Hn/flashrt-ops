#!/usr/bin/env python
"""send_grip 事件门（--grip-min-interval）/预合爪爬行（--grip-preclose）/
关节路径（--grip-path joint）离线单测。

零 SDK、零运动：ast 抽取 run_g1_loop_traj.py 的 send_grip / send_grip_joint
源码，在伪造命名空间（假 robot/假时钟）里喂全行流断言门判定时序。
背景=d835c6e 教训（首版 last_cmd 无守护→门永不触发，143851 实锤；门类逻辑
必须全行流单测，不能只看单点）。

判据（定律终版 174428；T4=10-21 P6 破案）：
  T1 事件门  首拍必发；意图没动（<max(chg,3%)）永不发（零travel重发只会重置
             静默钟）；间隔未满+静止=hold；间隔未满+在途=发（P2/P4：同向反向
             均平滑零罚，183232）；间隔≥N=发。
  T2 爬行    意图下行未平稳=爬速；窗内平稳=正常速；首拍（史短）=正常速。
  T3 回归    min_interval=0 时行为与旧 chg 门逐拍一致。
  T4 关节路径 双爪一包（groups 序=命令序）；position=宽度映射、velocity=
             GRIP 速度帽、effort=10（同事同款，非 cmd 路 30）；密流每拍必发
             （无 chg 门）；EMA 独立状态；GRIP["sent"] 照更（喂 state-cmd）；
             tfs 透传；dry 不发。
  T5 憋死保持 --grip-hold-on-stall（2026-10-21 力矩碰撞案）：平台窗未满=
             正常流式；平台满+深指令=进入（钳实宽−margin@effort_hold）；
             指令记账=模型意图不变；实宽让步→目标冻结不跟随（棘轮修复
             144325）；指令上穿实宽+exit=
  T6 深指令满窗 --grip-hold-window（150603 死锁教训）：静停开位+刚下合
             令=深指令史不满窗→不进（防冻结在开位死锁）；深指令持续满窗
             且平台=进；行程中平台破=永不进。
             退出当拍发模型目标；退出清史→再进须新平台；effort_hold=0=沿用10。
"""
import ast
import types
import time as _time

import numpy as np

SRC = "/home/galbot/holy/scripts/inference/run_g1_loop_traj.py"
tree = ast.parse(open(SRC).read())
FNS = [n for n in tree.body if isinstance(n, ast.FunctionDef)
       and n.name in ("send_grip", "send_grip_joint")]


class _Grp:
    def __init__(self, nm):
        self.nm = nm


class G1JointGroup:
    right_gripper = _Grp("right_gripper")
    left_gripper = _Grp("left_gripper")


class GS:
    def __init__(self, width, velocity, effort=0.0):
        self.width = width
        self.velocity = velocity
        self.effort = effort
        self.is_moving = abs(velocity) > 0.005
        self.timestamp_ns = 0


class JointCommand:
    def __init__(self):
        self.position = None
        self.velocity = None
        self.effort = None


class FakeRobot:
    def __init__(self):
        self.state = {G1JointGroup.right_gripper: GS(0.108, 0.0),
                      G1JointGroup.left_gripper: GS(0.108, 0.0)}
        self.calls = []
        self.jcalls = []    # (group 名序, [(pos, vel, eff)...], tfs_s)

    def set_gripper_command(self, grp, w, spd, eff, blk):
        self.calls.append((grp.nm, round(w, 4), round(spd, 4)))
        return "ControlStatus.SUCCESS"

    def set_joint_commands(self, cmds, joint_groups=(), joint_names=(),
                           time_from_start_s=0.0):
        self.jcalls.append(([g.nm for g in joint_groups],
                            [(c.position, c.velocity, c.effort) for c in cmds],
                            time_from_start_s))
        return "ControlStatus.SUCCESS"

    def get_gripper_state(self, grp):
        return self.state[grp]


def make_ns(clock):
    """伪造命名空间：假 time.monotonic 可控，其余为当前门配置。"""
    ns = {"np": np, "time": types.SimpleNamespace(monotonic=lambda: clock[0]),
          "G1JointGroup": G1JointGroup, "JointCommand": JointCommand,
          "GRIP_SS": None,
          "_GRIP_EMA": {}, "_GRIP_SENT_AT": {}, "_GRIP_PRE_HIST": {},
          "_GRIP_BEAT": 0, "_GRIPJ_EMA": {}, "_GRIPJ_HOLD": {}}
    mod = ast.Module(body=FNS, type_ignores=[])
    exec(compile(mod, "<send_grip+send_grip_joint>", "exec"), ns)
    return ns


def args_of(min_interval=0.0, preclose=0.0, ema=0.0,
            hold=False, gap=5.0, exit_mm=3.0, margin=3.0,
            plateau=0.8, window=5, effort_hold=4.0, hold_vel=0.05):
    return types.SimpleNamespace(do_exec=True, grip_send_every=0,
                                 grip_min_interval=min_interval,
                                 grip_preclose=preclose, grip_ema_alpha=ema,
                                 grip_hold_on_stall=hold,
                                 grip_hold_gap_mm=gap,
                                 grip_hold_exit_mm=exit_mm,
                                 grip_hold_margin_mm=margin,
                                 grip_hold_plateau_mm=plateau,
                                 grip_hold_window=window,
                                 grip_effort_hold=effort_hold,
                                 grip_hold_vel=hold_vel)


def grip_of(chg=2.0):
    return {"names": (("right_gripper", 7), ("left_gripper", 15)),
            "wmin": 0.0, "wmax": 0.12, "speed": 0.15, "effort": 30.0,
            "chg": chg, "sent": {}, "sent_peak": {}}


def rg(robot):
    return G1JointGroup.right_gripper


FAIL = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name} {detail}")
    if not cond:
        FAIL.append(name)


# ── T1 事件门 ──
print("T1 事件门 --grip-min-interval 1.2")
clock = [0.0]
ns = make_ns(clock)
ns.update(GRIP=grip_of(), args=args_of(min_interval=1.2), robot=FakeRobot())
robot = ns["robot"]
row = {7: 90.0, 15: 90.0}

send_grip = ns["send_grip"]
out = send_grip(robot, row)
r = next(c for c in robot.calls if c[0] == "right_gripper")
check("首拍必发", len([c for c in robot.calls if c[0] == "right_gripper"]) == 1,
      f"{r}")

robot.calls.clear()
for _ in range(3):
    send_grip(robot, row)                       # 意图没动
check("意图没动=hold（零travel重发不重置静默钟）",
      not robot.calls, f"{robot.calls}")

row[7] = 50.0
robot.state[rg(robot)] = GS(0.108, 0.0)         # 静止
clock[0] = 0.5                                  # 间隔未满
send_grip(robot, row)
check("间隔未满+静止=hold", not robot.calls)

clock[0] = 1.3                                  # 间隔已满
send_grip(robot, row)
check("间隔≥N=发", len(robot.calls) == 1, f"{robot.calls}")

robot.calls.clear()
row[7] = 20.0
robot.state[rg(robot)] = GS(0.07, -0.15)        # 在途合爪
clock[0] = 1.5
send_grip(robot, row)
check("间隔未满+在途同向=豁免发", len(robot.calls) == 1, f"{robot.calls}")

robot.calls.clear()
row[7] = 95.0
robot.state[rg(robot)] = GS(0.05, -0.10)        # 在途合爪但意图=开（反向）
clock[0] = 1.6
send_grip(robot, row)
check("在途反向=豁免发（P4 183232：反向也平滑掉头）",
      len(robot.calls) == 1, f"{robot.calls}")

# ── T2 预合爪爬行 ──
print("T2 预合爪爬行 --grip-preclose 0.03")
clock = [10.0]
ns = make_ns(clock)
ns.update(GRIP=grip_of(), args=args_of(preclose=0.03), robot=FakeRobot())
robot, send_grip = ns["robot"], ns["send_grip"]

speeds = []
# 前 5 拍下行；后 2 拍平稳（时钟先越过 0.75s 窗让下行值滚出历史，且意图
# 摆动 ≥chg 逼门放行，否则平稳拍被 chg 门 hold 观察不到速度）
seq = [(10.00, 90.0), (10.15, 85.0), (10.30, 80.0), (10.45, 74.0),
       (10.60, 70.0), (11.50, 73.0), (11.65, 71.0)]
for t, p in seq:
    clock[0] = t
    robot.state[rg(robot)] = GS(0.05, 0.0)
    robot.calls.clear()
    send_grip(robot, {7: p, 15: 90.0})
    speeds.append(next((c[2] for c in robot.calls
                        if c[0] == "right_gripper"), None))
check("首拍史短=正常速 0.15", speeds[0] == 0.15, f"{speeds}")
check("下行段=爬速 0.03", speeds[1:5] == [0.03] * 4, f"{speeds}")
check("平稳后=正常速 0.15", speeds[5:] == [0.15, 0.15], f"{speeds}")

# ── T3 chg 门回归 ──
print("T3 回归 min_interval=0/preclose=0（旧 chg 门行为不变）")
clock = [20.0]
ns = make_ns(clock)
ns.update(GRIP=grip_of(chg=2.0), args=args_of(), robot=FakeRobot())
robot, send_grip = ns["robot"], ns["send_grip"]
send_grip(robot, {7: 90.0, 15: 90.0})           # 首拍（chg 门首拍 last=None 也发）
n0 = len(robot.calls)
send_grip(robot, {7: 90.5, 15: 90.0})           # Δ0.5 < 2 → hold
check("Δ<chg=hold", len(robot.calls) == n0)
send_grip(robot, {7: 93.0, 15: 90.0})           # Δ3 ≥ 2 → 发
check("Δ≥chg=发", len(robot.calls) == n0 + 1)

# ── T4 关节路径 ──
print("T4 关节路径 send_grip_joint（10-21 P6 破案：定律属 set_gripper_command 端点）")
clock = [30.0]
ns = make_ns(clock)
ns.update(GRIP=grip_of(), args=args_of(), robot=FakeRobot())
robot, send_grip_joint = ns["robot"], ns["send_grip_joint"]

send_grip_joint(robot, {7: 90.0, 15: 10.0}, 0.33)
check("双爪一包：groups 序=命令序、宽度映射对",
      len(robot.jcalls) == 1
      and robot.jcalls[0][0] == ["right_gripper", "left_gripper"]
      and [p for p, _, _ in robot.jcalls[0][1]] == [0.108, 0.012],
      f"{robot.jcalls}")
jc = robot.jcalls[0][1]
check("velocity=速度帽 0.15 / effort=10（同事同款非 cmd 路 30）",
      all(v == 0.15 and e == 10.0 for _, v, e in jc), f"{jc}")
check("tfs 透传（与臂同参）", robot.jcalls[0][2] == 0.33)
check("GRIP['sent'] 照更（喂 --grip-state-cmd/sent_peak）",
      ns["GRIP"]["sent"] == {"right_gripper": 90.0, "left_gripper": 10.0})

robot.jcalls.clear()
for _ in range(5):
    send_grip_joint(robot, {7: 90.0, 15: 10.0}, 0.0)
check("密流每拍必发（无 chg/min-interval 门，P6j_dense）",
      len(robot.jcalls) == 5)

ns2 = make_ns(clock)
ns2.update(GRIP=grip_of(), args=args_of(ema=0.3), robot=FakeRobot())
r2, sgj2 = ns2["robot"], ns2["send_grip_joint"]
sgj2(r2, {7: 100.0, 15: 0.0}, 0.0)
r2.jcalls.clear()
sgj2(r2, {7: 0.0, 15: 100.0}, 0.0)
ps = [p for p, _, _ in r2.jcalls[0][1]]
check("EMA 独立状态（α=0.3：100→0 首拍=70%→0.084m）",
      all(abs(a - b) < 1e-9 for a, b in zip(ps, [0.084, 0.036])), f"{ps}")

ns3 = make_ns(clock)
ns3.update(GRIP=grip_of(), args=args_of(), robot=FakeRobot())
ns3["args"].do_exec = False
_dry = ns3["send_grip_joint"](ns3["robot"], {7: 50.0, 15: 50.0}, 0.0)
check("dry 不发", _dry == "dry（不下发）" and not ns3["robot"].jcalls)

# ── T5 憋死保持 ──
print("T5 憋死保持 --grip-hold-on-stall（2026-10-21 力矩碰撞案）")
clock = [40.0]
ns = make_ns(clock)
ns.update(GRIP=grip_of(), args=args_of(hold=True), robot=FakeRobot())
robot, sgj = ns["robot"], ns["send_grip_joint"]

# 前 4 拍：实宽被物撑在 42mm，指令 5%（=6mm，深指令 gap 36mm）但平台窗未满
for _ in range(4):
    robot.state[G1JointGroup.right_gripper] = GS(0.042, 0.0)
    robot.jcalls.clear()
    sgj(robot, {7: 5.0, 15: 90.0}, 0.0)
    jc = robot.jcalls[0][1][0]
check("平台窗未满=正常流式（pos 6mm/eff 10）",
      jc[0] == 0.006 and jc[2] == 10.0, f"{jc}")

# 第 5 拍：平台满 → 进入保持：钳 39mm@eff4
robot.jcalls.clear()
sgj(robot, {7: 5.0, 15: 90.0}, 0.0)
jc = robot.jcalls[0][1][0]
check("平台满+深指令=进入保持（钳实宽−3mm）", abs(jc[0] - 0.039) < 1e-9, f"{jc}")
check("保持段 effort 降 4", jc[2] == 4.0, f"{jc}")
check("保持段速度帽放软 0.05（093335：稳态误差=每拍重武装内顶）",
      jc[1] == 0.05, f"{jc}")
check("指令记账=模型意图（state-cmd/sent_peak 语义不变）",
      ns["GRIP"]["sent"]["right_gripper"] == 5.0)
check("左爪不受牵连（90%=108mm 正常发）",
      abs(robot.jcalls[0][1][1][0] - 0.108) < 1e-9)

# 持物中实宽让步/被顶开 → 目标冻结不动（棘轮修复 144325：勿跟随实宽，
# 否则接触面一让步目标再进 margin=慢速碾进）
robot.state[G1JointGroup.right_gripper] = GS(0.045, 0.0)
robot.jcalls.clear()
sgj(robot, {7: 5.0, 15: 90.0}, 0.0)
jc = robot.jcalls[0][1][0]
check("保持中目标冻结（实宽让到 45mm 仍钳39mm）",
      abs(jc[0] - 0.039) < 1e-9, f"{jc}")

# 模型想开爪：指令 90%（108mm ≥ 45+3）→ 退出，当拍直接发模型目标
robot.jcalls.clear()
sgj(robot, {7: 90.0, 15: 90.0}, 0.0)
jc = robot.jcalls[0][1][0]
check("指令上穿实宽+exit=退出当拍发模型目标（速度恢复 0.15）",
      abs(jc[0] - 0.108) < 1e-9 and jc[2] == 10.0 and jc[1] == 0.15, f"{jc}")

# 退出清史：再合须新平台（1-4 拍不进，第 5 拍才进）
reentered = None
for k in range(1, 8):
    robot.state[G1JointGroup.right_gripper] = GS(0.045, 0.0)
    robot.jcalls.clear()
    sgj(robot, {7: 5.0, 15: 90.0}, 0.0)
    if abs(robot.jcalls[0][1][0][0] - 0.042) < 1e-9:
        reentered = k
        break
check("退出清史→再进须新平台窗（第5拍）", reentered == 5, f"第{reentered}拍")

# effort_hold=0：保持段沿用 10（只钳位不降压）
ns = make_ns(clock)
ns.update(GRIP=grip_of(), args=args_of(hold=True, effort_hold=0.0),
          robot=FakeRobot())
robot, sgj = ns["robot"], ns["send_grip_joint"]
for _ in range(5):
    robot.state[G1JointGroup.right_gripper] = GS(0.042, 0.0)
    sgj(robot, {7: 5.0, 15: 90.0}, 0.0)
jc = robot.jcalls[-1][1][0]
check("effort_hold=0=保持段沿用 10", jc[2] == 10.0, f"{jc}")
check("钳位仍生效", abs(jc[0] - 0.039) < 1e-9, f"{jc}")

# ── T6 深指令满窗（150603 死锁教训） ──
print("T6 深指令满窗 --grip-hold-window 5（静停开位+新合令防误进）")
clock = [50.0]
ns = make_ns(clock)
ns.update(GRIP=grip_of(), args=args_of(hold=True, window=5), robot=FakeRobot())
robot, sgj = ns["robot"], ns["send_grip_joint"]

sent_seq = []
# 拍1-3：爪停在开位 115.5mm、指令还开着（98%=117.6mm 不深）；拍4起指令转合
# （80%=96mm 深）。拍4-7 深指令史不满窗 → 不得进（否则冻结在开位死锁）；
# 拍8（深满窗+平台）才进。
for k in range(1, 9):
    p = 98.0 if k <= 3 else 80.0
    robot.state[G1JointGroup.right_gripper] = GS(0.1155, 0.0)
    robot.jcalls.clear()
    sgj(robot, {7: p, 15: 90.0}, 0.0)
    sent_seq.append(robot.jcalls[0][1][0][0])
check("拍1-3 开指令不进（发意图 117.6mm）",
      all(abs(x - 0.1176) < 1e-9 for x in sent_seq[:3]), f"{sent_seq}")
check("拍4-7 深指令史不满窗=不进（发意图 96mm，非钳位）",
      all(abs(x - 0.096) < 1e-9 for x in sent_seq[3:7]), f"{sent_seq}")
check("拍8 深满窗+平台=进（钳 112.5mm）",
      abs(sent_seq[7] - 0.1125) < 1e-9, f"{sent_seq}")

print("T6b 行程中平台破=永不进（自由合爪不误判憋死）")
clock = [60.0]
ns = make_ns(clock)
ns.update(GRIP=grip_of(), args=args_of(hold=True, window=5), robot=FakeRobot())
robot, sgj = ns["robot"], ns["send_grip_joint"]
sents = []
for k in range(8):
    robot.state[G1JointGroup.right_gripper] = GS(0.100 - k * 0.005, 0.0)
    robot.jcalls.clear()
    sgj(robot, {7: 10.0, 15: 90.0}, 0.0)
    sents.append(robot.jcalls[0][1][0][0])
check("行程中永不进（实宽每拍走 5mm，每拍发意图 12mm 非钳位）",
      all(abs(x - 0.012) < 1e-9 for x in sents), f"{sents}")

print(f"\n{'全绿' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}")
raise SystemExit(1 if FAIL else 0)
