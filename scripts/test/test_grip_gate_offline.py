#!/usr/bin/env python
"""send_grip 事件门（--grip-min-interval）与预合爪爬行（--grip-preclose）离线单测。

零 SDK、零运动：ast 抽取 run_g1_loop_traj.py 的 send_grip 源码，在伪造命名空间
（假 robot/假时钟）里喂全行流断言门判定时序。背景=d835c6e 教训（首版 last_cmd
无守护→门永不触发，143851 实锤；门类逻辑必须全行流单测，不能只看单点）。

判据（定律终版 174428）：
  T1 事件门  首拍必发；意图没动（<max(chg,3%)）永不发（零travel重发只会重置
             静默钟）；间隔未满+静止=hold；间隔未满+在途=发（P2/P4：同向反向
             均平滑零罚，183232）；间隔≥N=发。
  T2 爬行    意图下行未平稳=爬速；窗内平稳=正常速；首拍（史短）=正常速。
  T3 回归    min_interval=0 时行为与旧 chg 门逐拍一致。
"""
import ast
import types
import time as _time

import numpy as np

SRC = "/home/galbot/holy/scripts/inference/run_g1_loop_traj.py"
tree = ast.parse(open(SRC).read())
fn = next(n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "send_grip")


class _Grp:
    def __init__(self, nm):
        self.nm = nm


class G1JointGroup:
    right_gripper = _Grp("right_gripper")
    left_gripper = _Grp("left_gripper")


class GS:
    def __init__(self, width, velocity):
        self.width = width
        self.velocity = velocity
        self.is_moving = abs(velocity) > 0.005
        self.timestamp_ns = 0


class FakeRobot:
    def __init__(self):
        self.state = {G1JointGroup.right_gripper: GS(0.108, 0.0),
                      G1JointGroup.left_gripper: GS(0.108, 0.0)}
        self.calls = []

    def set_gripper_command(self, grp, w, spd, eff, blk):
        self.calls.append((grp.nm, round(w, 4), round(spd, 4)))
        return "ControlStatus.SUCCESS"

    def get_gripper_state(self, grp):
        return self.state[grp]


def make_ns(clock):
    """伪造命名空间：假 time.monotonic 可控，其余为当前门配置。"""
    ns = {"np": np, "time": types.SimpleNamespace(monotonic=lambda: clock[0]),
          "G1JointGroup": G1JointGroup, "GRIP_SS": None,
          "_GRIP_EMA": {}, "_GRIP_SENT_AT": {}, "_GRIP_PRE_HIST": {},
          "_GRIP_BEAT": 0}
    mod = ast.Module(body=[fn], type_ignores=[])
    exec(compile(mod, "<send_grip>", "exec"), ns)
    return ns


def args_of(min_interval=0.0, preclose=0.0, ema=0.0):
    return types.SimpleNamespace(do_exec=True, grip_send_every=0,
                                 grip_min_interval=min_interval,
                                 grip_preclose=preclose, grip_ema_alpha=ema)


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

print(f"\n{'全绿' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}")
raise SystemExit(1 if FAIL else 0)
