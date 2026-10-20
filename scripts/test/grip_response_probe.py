#!/usr/bin/env python
"""夹爪固件响应三阶段探针（零臂运动，只动单爪）——10-13 标定台的续篇。

10-13 台（grip_speed_bench）只测了左爪单发，测得动启动 0.91-0.96s（速度列
实锤真零）。用户记忆「固件立马响应」与之冲突，本台三阶段分辨：

  P1 onset 复测（左右爪对比）——静止态单发全行程，30Hz 轮询，速度列判启动。
     左右爪差异 / 与 10-13 的 0.91s 复现性，一次看清。
  P2 重启规则（行程中同向重发）——合爪已动 0.3s 时再发一条全合指令
     （正是闭环干的事）：若出现中途 ≥0.25s 零速段再起步 = 「每条新指令
     重启行程重吃死区」实锤；若一路滑到底 = 重启律证伪。
     〔174428 已判：同向在途重发=满速零凹陷平滑，重启律（在途版）证伪〕
  P4 在途反向——合爪在途 0.3s 时反向发全开：事件门反向豁免的判决格
     （183232 已判=平滑掉头 ~0.2s 不碰零；10-16 反转律改写：杀手=死区密发
     非方向；保留供复现/其它爪验证）。
  P5 最小速度极限梯——降序速度梯（--creep-ladder）从静止全开起步全合，每档
     --creep-wait 秒窗（默认 6s），判据 |v|>0.005 或宽度位移≥0.5mm（慢速档
     速度列可能贴噪声，位移兜底）：得 onset(v) 谱 + 固件最小可执行速度。
     0.01/0.03 已知 4s 窗未启动（183232/184439），故拉窗；未启动≠不可执行
     ——两点拟合 onset=L+D/v 外推失败档，分辨「窗太短」vs「真低于最小速」。
     末尾在最小可执行档跑接管格（在途同目标改发正常速：接管 or 继承）=
     预合爪爬行（--grip-preclose）的前提格。
  P3 密发复现（chg 门模拟）——0.1s 间隔连发 3s 微摆目标（模拟裸 chg 门），
     随后静默 2.5s：密发期宽度冻结 + 静默期一次补走 = 111901 定律现场复现。

产物落 evidence/grip_response_probe/。判读全在控制台摘要，CSV 供存档。

用法（真机；确认爪行程内无障碍、急停可及）:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \\
  ~/miniforge3/envs/flash_pyrt311/bin/python \\
      ~/holy/scripts/test/grip_response_probe.py --side both
可选: --side left|right|both | --speed 0.15 | --effort 30 | --poll-hz 30
安全: 只发夹爪指令，臂/腿/头维度永不触碰；Ctrl-C 随时中止。
"""
import argparse
import csv
import pathlib
import time

from galbot_sdk.g1 import GalbotRobot, G1JointGroup

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/galbot/holy/models/pi05_g1_onlypick_deploy")
ap.add_argument("--side", choices=["left", "right", "both"], default="both")
ap.add_argument("--speed", type=float, default=0.15, help="m/s，生产值 0.15")
ap.add_argument("--creep-ladder", default="0.12,0.10,0.08,0.06,0.05,0.04,0.03,0.02,0.01",
                help="P5 速度梯 m/s，降序逗号分隔（找固件最小可执行速度）")
ap.add_argument("--creep-wait", type=float, default=6.0,
                help="P5 每档首动观察窗 s（0.03 在 4s 窗未启动过，默认 6s）")
ap.add_argument("--effort", type=float, default=30)
ap.add_argument("--poll-hz", type=float, default=30.0)
ap.add_argument("--only", default="",
                help="只跑指定相位，逗号分隔，如 --only p5（默认全部 p1,p2,p4,p5,p3）")
args = ap.parse_args()

mf_p = pathlib.Path(args.ckpt) / "flashrt_deploy.json"
mf = mf_p and (mf_p.read_text() if mf_p.exists() else "{}")
import json
_g = json.loads(mf).get("gripper", {})
WMIN, WMAX = _g.get("width_min"), _g.get("width_max")
if WMIN is None or WMAX is None:
    raise SystemExit("manifest 无夹爪标定（width_min/width_max）——先跑 g1_ckpt_prep.py")
WMID = (WMIN + WMAX) / 2.0

print("== 夹爪固件响应三阶段探针 ==")
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(3)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")
if not str(st).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止")

out_d = pathlib.Path("/home/galbot/holy/evidence/grip_response_probe")
out_d.mkdir(parents=True, exist_ok=True)
tag = time.strftime("%Y%m%d-%H%M%S")
rows = []          # [side, phase, t_rel_s, width_mm, velocity_mps, is_moving, ts_age_s, status]


def poll_span(robot, group, t0, seconds, side, phase, target=None, speed=None):
    """轮询 seconds 秒；target 非 None 时先发一条指令。返回首个 |v|>0.005 的 t。"""
    if target is not None:
        stc = robot.set_gripper_command(group, target, speed or args.speed,
                                        args.effort, False)
    else:
        stc = "-"
    first_move = None
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        tel = time.perf_counter() - t0
        gs = robot.get_gripper_state(group)
        if gs is not None:
            age = (time.time_ns() - gs.timestamp_ns) / 1e9 if getattr(gs, "timestamp_ns", 0) else float("nan")
            v = gs.velocity
            w = gs.width * 1000.0
            rows.append([side, phase, round(tel, 3), round(w, 2), round(v, 4),
                         int(gs.is_moving), round(age, 3),
                         str(stc).replace("ControlStatus.", "")])
            if first_move is None and abs(v) > 0.005:
                first_move = tel
        time.sleep(1.0 / args.poll_hz)
    return first_move


def settle(robot, group, need=1.2, timeout=8.0):
    """等爪静止（|v|<0.005 连续 need 秒）。need 必须 > 死区上限 ~0.96s：
    待发行程在死区里速度恒零，need 太短会把「还没动」误判成「已停稳」，
    下一条指令会把前一条吞掉（173520 P2 实录：0.8s < 0.9s 死区 → 开爪
    指令被吞、120mm 行程从未执行）。"""
    t0 = time.perf_counter()
    last_move = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        gs = robot.get_gripper_state(group)
        if gs is None or abs(gs.velocity) < 0.005:
            if time.perf_counter() - last_move > need:
                return
        else:
            last_move = time.perf_counter()
        time.sleep(0.05)


def wait_first_motion(robot, group, t0, timeout=3.0):
    """等在途首动（|v|>0.005）。返回首动相对 t0 的时刻，超时 None。"""
    while time.perf_counter() - t0 < timeout:
        gs = robot.get_gripper_state(group)
        if gs and abs(gs.velocity) > 0.005:
            return time.perf_counter() - t0
        time.sleep(1.0 / args.poll_hz)
    return None


def trace_seg(robot, group, side, phase, stc, trace_s):
    """trace_s 秒 30Hz 采样并写 rows；返回 [(t_rel, |v|)]。"""
    seg = []
    t2 = time.perf_counter()
    t_end = t2 + trace_s
    while time.perf_counter() < t_end:
        tel = time.perf_counter() - t2
        gs = robot.get_gripper_state(group)
        if gs is not None:
            age = (time.time_ns() - gs.timestamp_ns) / 1e9 if getattr(gs, "timestamp_ns", 0) else float("nan")
            seg.append((tel, abs(gs.velocity)))
            rows.append([side, phase, round(tel, 3), round(gs.width * 1000, 2),
                         round(gs.velocity, 4), int(gs.is_moving), round(age, 3),
                         str(stc).replace("ControlStatus.", "")])
        time.sleep(1.0 / args.poll_hz)
    return seg


def zero_runs(seg):
    """|v| 零速段 [(起, 止, 是否中途)]；中途=段后仍有运动。"""
    runs, i = [], 0
    while i < len(seg):
        if seg[i][1] < 0.005:
            j = i
            while j < len(seg) and seg[j][1] < 0.005:
                j += 1
            runs.append((seg[i][0], seg[j - 1][0], j < len(seg)))
            i = j
        else:
            i += 1
    return runs


input("确认爪行程内无障碍、急停可及，回车开始（Ctrl-C 中止）> ")

_ONLY = ({s.strip().lower() for s in args.only.split(",") if s.strip()}
         if args.only.strip() else None)


def _run(ph):
    """--only 相位过滤（默认全跑）。"""
    return _ONLY is None or ph in _ONLY


def phase_p1(robot, group, side):
    """P1 onset：静止态单发全行程（开→各回），速度列判启动。"""
    print("P1 onset（单发全行程）:")
    settle(robot, group)
    t0 = time.perf_counter()
    fm = poll_span(robot, group, t0, 4.0, side, "P1_open", target=WMAX)
    print(f"  开: 启动 {fm if fm is not None else -1:.2f}s" if fm is not None else "  开: 4s 窗内未动 ⚠")
    settle(robot, group)
    t0 = time.perf_counter()
    fm = poll_span(robot, group, t0, 4.0, side, "P1_close", target=WMIN)
    print(f"  合: 启动 {fm if fm is not None else -1:.2f}s" if fm is not None else "  合: 4s 窗内未动 ⚠")
    settle(robot, group)


def _close_stroke(robot, group):
    """从当前态回到全开静止，再起步一次全合行程；返回首动时刻（None=未启动）。"""
    robot.set_gripper_command(group, WMAX, args.speed, args.effort, False)
    settle(robot, group)                      # 回到全开静止
    t0 = time.perf_counter()
    robot.set_gripper_command(group, WMIN, args.speed, args.effort, False)
    return wait_first_motion(robot, group, t0)


def phase_p2(robot, group, side):
    """P2 重启规则：合爪在途 0.3s 时【同向重发】（174428 已判=平滑零凹陷；
    保留供复现/其它爪验证）。"""
    print("P2 重启规则（合爪中途同向重发）:")
    mv = _close_stroke(robot, group)
    if mv is None:
        print("  ⚠ 3s 内未启动，P2 跳过")
    else:
        time.sleep(0.3)                       # 已在途 0.3s
        stc = robot.set_gripper_command(group, WMIN, args.speed, args.effort, False)
        seg = trace_seg(robot, group, side, "P2_retarget", stc, 3.5)
        mid = [r for r in zero_runs(seg) if r[2] and (r[1] - r[0]) >= 0.25]
        if mid:
            p = max(mid, key=lambda r: r[1] - r[0])
            print(f"  首动 {mv:.2f}s → 同向重发后 {p[0]:.2f}s 停车、{p[1]:.2f}s 再动"
                  f"（停顿 {p[1] - p[0]:.2f}s）→ 重启律实锤")
        else:
            term = [r for r in zero_runs(seg) if not r[2]]
            print("  同向重发后无中途零速段（一路滑到底）→ 重启律证伪"
                  + (f"（仅末段到位停车 @{term[0][0]:.2f}s）" if term else ""))
    settle(robot, group)


def phase_p4(robot, group, side):
    """P4 在途反向：合爪在途 0.3s 时反向发全开（183232 已判=平滑掉头 ~0.2s
    不碰零、事件门反向豁免成立；保留供复现/其它爪验证）。"""
    print("P4 在途反向（合爪中途发全开）:")
    mv = _close_stroke(robot, group)
    if mv is None:
        print("  ⚠ 3s 内未启动，P4 跳过")
    else:
        time.sleep(0.3)
        stc = robot.set_gripper_command(group, WMAX, args.speed, args.effort, False)
        seg = trace_seg(robot, group, side, "P4_reverse", stc, 4.5)
        mid = [r for r in zero_runs(seg) if r[2] and (r[1] - r[0]) >= 0.25]
        if mid:
            p = max(mid, key=lambda r: r[1] - r[0])
            print(f"  反向重发后 {p[0]:.2f}s 停车、{p[1]:.2f}s 再动"
                  f"（停顿 {p[1] - p[0]:.2f}s）→ 在途反向=重启实锤")
        else:
            term = [r for r in zero_runs(seg) if not r[2]]
            print("  在途反向无中途零速段 → 反向也平滑"
                  + (f"（仅末段停车 @{term[0][0]:.2f}s）" if term else ""))
    settle(robot, group)


def phase_p5(robot, group, side):
    """P5 固件最小速度极限梯：降序速度梯逐档从静止全开起步全合，判据
    |v|>0.005 或宽度位移≥0.5mm；两点拟合 onset=L+D/v 外推失败档；最小可执行
    档上跑接管格（预合爪爬行 --grip-preclose 的前提格）。"""
    ladder = [float(x) for x in args.creep_ladder.split(",") if x.strip()]
    print(f"P5 最小速度极限梯 {ladder}（每档窗 {args.creep_wait:.0f}s）:")
    res = []                                      # (v, onset, vmax, dw)
    for v in ladder:
        gs = robot.get_gripper_state(group)
        w0 = gs.width * 1000.0 if gs else 0.0
        if w0 < WMAX * 1000.0 - 2.0:              # 上档动过 → 回全开再静置
            robot.set_gripper_command(group, WMAX, args.speed, args.effort, False)
            settle(robot, group)
        t0 = time.perf_counter()
        ph = f"P5_lad_{v:.2f}"
        poll_span(robot, group, t0, args.creep_wait, side, ph,
                  target=WMIN, speed=v)
        step = [r for r in rows if r[1] == ph]
        onset = None
        if step:
            w_start = step[0][3]
            for r in step:
                if abs(r[4]) > 0.005 or abs(r[3] - w_start) >= 0.5:
                    onset = r[2]
                    break
        vmax = max((abs(r[4]) for r in step), default=0.0)
        dw = abs(step[-1][3] - step[0][3]) if step else 0.0
        res.append((v, onset, vmax, dw))
        if onset is not None:
            print(f"  v={v:.2f}: 启动 {onset:.2f}s | max|v|={vmax:.3f} | "
                  f"Δw={dw:.1f}mm")
        else:
            print(f"  v={v:.2f}: {args.creep_wait:.0f}s 未启动 | "
                  f"max|v|={vmax:.3f} | Δw={dw:.1f}mm")
    ok = [r for r in res if r[1] is not None]     # 梯序=速度降序
    if len(ok) >= 2:
        v_hi, t_hi = ok[0][0], ok[0][1]           # 最快档
        v_lo, t_lo = ok[-1][0], ok[-1][1]         # 最慢可执行档
        den = 1.0 / v_lo - 1.0 / v_hi
        D = (t_lo - t_hi) / den if den > 1e-9 else float("nan")   # 行程等效距离
        L = t_hi - D / v_hi if D == D else float("nan")           # 固定起步延迟
        print(f"  onset=L+D/v 拟合（两点 {v_hi:.2f}/{v_lo:.2f}）: "
              f"L={L:.2f}s D={D * 1000:.0f}mm")
        for v, onset, _, _ in res:
            if onset is None and D == D and D > 0:
                tp = L + D / v
                print(f"  外推 v={v:.2f} onset≈{tp:.1f}s → "
                      f"{'窗太短非不可执行' if tp > args.creep_wait else '应已启动却未动=疑似真低于最小速'}")
    vmin = min((r[0] for r in ok), default=None)
    if vmin is None:
        print("  ⚠ 全梯未启动 → 梯顶以下全灭：--grip-preclose 判死候选"
              "（除非外推显示窗太短——加档重跑）")
        settle(robot, group)
        return
    onset_v = next(r[1] for r in res if r[0] == vmin)
    print(f"  → 最小可执行速度 ≈ {vmin:.2f} m/s（onset {onset_v:.2f}s）")
    # 接管格：最小可执行档起步在途 → 正常速同目标重发
    robot.set_gripper_command(group, WMAX, args.speed, args.effort, False)
    settle(robot, group)
    t0 = time.perf_counter()
    robot.set_gripper_command(group, WMIN, vmin, args.effort, False)
    mv = wait_first_motion(robot, group, t0, timeout=max(10.0, onset_v * 1.5))
    if mv is None:
        print("  ⚠ 接管格起步未启动（与梯判矛盾=复现差异），跳过")
    else:
        time.sleep(0.5)                           # 最小档在途 0.5s
        stc = robot.set_gripper_command(group, WMIN, args.speed, args.effort, False)
        seg = trace_seg(robot, group, side, "P5_speedtakeover", stc, 4.0)
        win = [v for t, v in seg if 0.2 <= t <= 1.2]   # 重发后稳定窗
        vmean = sum(win) / len(win) if win else float("nan")
        if vmean >= 0.08:
            verdict = "速度接管（新令生效）→ 爬行可行：最小档起步+正常速收敛"
        elif vmean <= max(0.03, vmin):
            verdict = "速度继承（行程保持旧速）→ 爬行段全程慢速，收敛等意图平稳"
        else:
            verdict = "混合/渐变（看 CSV 细判）"
        print(f"  重发后 0.2-1.2s 平均|v|={vmean:.3f} m/s（爬速 {vmin:.2f} / "
              f"正常 {args.speed}）→ {verdict}")
    settle(robot, group, need=1.2, timeout=15.0)


def phase_p3(robot, group, side):
    """P3 密发复现：0.1s 连发微摆目标 3s（模拟裸 chg 门），静默 2.5s。"""
    print("P3 密发复现（0.1s 连发 3s → 静默 2.5s）:")
    t0 = time.perf_counter()
    n_sent = 0
    while time.perf_counter() - t0 < 3.0:
        tel = time.perf_counter() - t0
        tgt = WMID + (5.0 / 1000.0) * (1 if int(tel / 0.1) % 2 else -1)  # ±5mm 微摆
        robot.set_gripper_command(group, tgt, args.speed, args.effort, False)
        n_sent += 1
        gs = robot.get_gripper_state(group)
        if gs is not None:
            age = (time.time_ns() - gs.timestamp_ns) / 1e9 if getattr(gs, "timestamp_ns", 0) else float("nan")
            rows.append([side, "P3_dense", round(tel, 3), round(gs.width * 1000, 2),
                         round(gs.velocity, 4), int(gs.is_moving), round(age, 3), "SENT"])
        time.sleep(0.1)
    gs = robot.get_gripper_state(group)
    w_dense_end = gs.width * 1000.0 if gs else float("nan")
    fm3 = poll_span(robot, group, time.perf_counter(), 2.5, side, "P3_silence")
    gs = robot.get_gripper_state(group)
    w_end = gs.width * 1000.0 if gs else float("nan")
    print(f"  密发 {n_sent} 条（±5mm 摆动目标）→ 密发末宽度 {w_dense_end:.1f}mm | "
          f"静默 2.5s → {w_end:.1f}mm（补走 {abs(w_end - w_dense_end):.1f}mm）")


sides = ["left", "right"] if args.side == "both" else [args.side]
for side in sides:
    group = getattr(G1JointGroup, f"{side}_gripper")
    print(f"\n──── {side} 爪 ────")
    gs = robot.get_gripper_state(group)
    print(f"起始宽度 {(gs.width * 1000 if gs else float('nan')):.1f} mm")
    for fn, ph in ((phase_p1, "p1"), (phase_p2, "p2"), (phase_p4, "p4"),
                   (phase_p5, "p5"), (phase_p3, "p3")):
        if _run(ph):
            fn(robot, group, side)

out_p = out_d / f"{tag}_{args.side}.csv"
with out_p.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["side", "phase", "t_rel_s", "width_mm", "velocity_mps",
                "is_moving", "state_ts_age_s", "status"])
    w.writerows(rows)
print(f"\n曲线已落盘: {out_p}")
print("判读: P1 左右爪 onset 对比（10-13 左爪=0.91s）；P2 停顿>0.3s=重启律实锤"
      "（174428 已判=证伪·在途同向平滑）；P4 停顿>0.3s=在途反向重启实锤（事件门"
      "反向豁免判死）；P5 梯=固件最小可执行速度+接管格（≥0.08=接管→--grip-preclose "
      "可行；全梯未启动=preclose 判死候选）；"
      "P3 密发期冻结+静默期补走=165556 阶梯机制现场复现。")
