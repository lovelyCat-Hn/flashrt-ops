#!/usr/bin/env python
"""batch/单点 tfs 语义黑盒探针——判 singorix WBCS 对 tfs 的真实执行方式。

背景（10-29）：singorix_target_guide.md §5 给出两种模式的判别性预测——
  OVERRIDE + LinearInterpolate 单点 tfs=T：无左侧起点 → 等待≈T 后直接跳到该点
  PROVERRIDE + LinearInterpolate 单点 tfs=T：从当前 command 起点平滑运动 T 时长
  LinearInterpolate 批次：按 tfs 时间轴逐区间插值（点距粗也会被 wbc_rate 细插）
判据（各条件打印 Δ 的 25/50/90% 到达时刻，均相对本段起点）：
  A 单点 tfs=2s   到达≈立刻=ASAP；≈2s 且平滑=PROVERRIDE；先等≈2s 再跳=OVERRIDE
  B 批次 [0.5,1s] 第二点贴 1s 到达=时间轴被执行；贴 0.5s/立刻=未按时间轴
  C 单点 tfs=0    今日 fr1 同款（预期立刻到达；30Hz 臂上大步=削顶踢的源头形态）
  D 批次全 0 tfs  G1 式 ASAP 直发假设（10-29 用户提议）：两点distinct目标但
                  都不设到达时刻——若 SDK/采样器塌缩到末点=一次大跳（比 fr1
                  更糟）；若逐点 ASAP 快翻=stop-go；若报错=SDK 拒收

安全：只动 head_joint1/2，幅度 ≤ --amp（默认 0.03 rad，远小于限位）；--exec
交互确认 + start_controller；结束回初始位；退出走优雅三连（10-28 教训）。
get_joint_states 返回规范序非请求序——按 joint_name 对号（10-29 实测坑）。

用法：
  PYTHONPATH=/data/galbot/lib \\
  ~/miniforge3/envs/flash_pyrt311/bin/python scripts/s1/probes/batch_tfs_probe.py --exec
"""
from __future__ import annotations

import argparse
import sys
import time

AP = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--exec", dest="do_exec", action="store_true", help="真机执行（默认 dry 打印计划）")
AP.add_argument("--amp", type=float, default=0.03, help="head_joint1 摆幅 rad（默认 0.03，硬上限 0.05）")
AP.add_argument("--log-hz", type=float, default=100.0, help="每条件状态突发采样 Hz")
AP.add_argument("--joint", default="head_joint1", help="被测关节（默认 head_joint1，最轻）")
args = AP.parse_args()

amp = min(abs(args.amp), 0.05)
if not args.do_exec:
    print(f"（dry：将对 {args.joint} 做 A/B/C/D 四条件 tfs 探针，幅 {amp} rad；加 --exec 执行）")
    sys.exit(0)

from galbot_sdk.s1 import GalbotRobot, JointCommand

robot = GalbotRobot()
if not robot.init(set()):
    raise SystemExit("robot.init 失败（服务在跑吗？）")
time.sleep(2)

jn = [args.joint]
try:
    print("\n⚠⚠ 将小幅度摆动头部关节（幅 ≤0.05 rad）。急停已释放？手不离急停。")
    if input("确认继续？(y/N) ").strip().lower() != "y":
        raise SystemExit("未确认，退出。")
    robot.start_controller("all")
    print("✓ start_controller(all)")
    time.sleep(1)

    p0 = float(robot.get_joint_positions(joint_names=jn)[0])
    print(f"初始 {jn[0]} = {p0:.4f} rad；幅 {amp} rad")

    def burst(sec: float) -> list[tuple]:
        """sec 秒内突发采样 (t, pos, |acc|, |cur|)。"""
        out, dl = [], time.perf_counter() + sec
        while time.perf_counter() < dl:
            st = {s.joint_name: s for s in robot.get_joint_states(joint_names=jn)}[jn[0]]
            out.append((time.perf_counter(), float(st.position),
                        abs(float(st.acceleration)), abs(float(st.current))))
            time.sleep(1.0 / args.log_hz)
        return out

    def report(tag: str, rec: list[tuple], t0: float, delta: float,
               p_start: float) -> None:
        hit = [False, False, False]
        t_hit = [None, None, None]
        max_acc = max_cur = 0.0
        for t, p, a, c in rec:
            f = (p - p_start) / delta if delta else 0.0
            for i, th in enumerate((0.25, 0.5, 0.9)):
                if not hit[i] and f >= th:
                    hit[i], t_hit[i] = True, t - t0
            max_acc, max_cur = max(max_acc, a), max(max_cur, c)
        lag = [f"{x*1000:.0f}ms" if x is not None else "—" for x in t_hit]
        print(f"  {tag}: 25/50/90% 到达 = {lag[0]}/{lag[1]}/{lag[2]} | "
              f"max|acc|={max_acc:.1f} rad/s² max|cur|={max_cur:.2f} A")

    def one(target: float, tfs: float, label: str, dur: float,
            delta: float | None = None) -> None:
        ps = float(robot.get_joint_positions(joint_names=jn)[0])
        c = JointCommand()
        c.position = target
        c.velocity = c.acceleration = c.effort = 0.0
        t0 = time.perf_counter()
        robot.set_joint_commands([c], joint_names=jn, time_from_start_s=tfs)
        report(label, burst(dur), t0, target - ps if delta is None else delta, ps)

    def batch(targets: list[float], tfss: list[float], label: str, dur: float,
              delta: float | None = None) -> None:
        from galbot_sdk.s1 import Trajectory, TrajectoryPoint
        ps = float(robot.get_joint_positions(joint_names=jn)[0])
        tr = Trajectory()
        tr.joint_names = jn
        pts = []
        for tgt, tf in zip(targets, tfss):
            pt = TrajectoryPoint()
            pt.time_from_start_second = tf
            c = JointCommand()
            c.position = tgt
            c.velocity = c.acceleration = c.effort = 0.0
            pt.joint_command_vec = [c]   # ⚠ SDK getter 返副本：vec/points 勿 append，整体赋值
            pts.append(pt)
        tr.points = pts
        t0 = time.perf_counter()
        robot.set_joint_commands_batch(tr)
        report(label, burst(dur), t0,
               targets[-1] - ps if delta is None else delta, ps)

    print("\n[A] 单点 tfs=2.0s（判 OVERRIDE 等待跳 vs PROVERRIDE 平滑）")
    one(p0 + amp, 2.0, "A 去 +amp tfs=2s", 3.0)
    print("[B] 批次 2 点 tfs=[0.5, 1.0]（判时间轴是否被执行）")
    batch([p0 + amp, p0 + amp * 0.5], [0.5, 1.0], "B 批次 0.5/1s", 2.0)
    print("[C] 单点 tfs=0（回零；今日 fr1 同款参照）")
    one(p0, 0.0, "C 回零 tfs=0", 1.0)
    print("[D] 批次 2 点全 tfs=0、目标不同（G1 式 ASAP 直发假设）")
    batch([p0 + amp * 0.5, p0 + amp], [0.0, 0.0], "D 全零tfs批", 1.5)
    print("\n[回位] 批次 2 点 tfs=[0.5, 1.0] 回初始")
    batch([p0 + amp * 0.5, p0], [0.5, 1.0], "回位", 2.0, delta=amp)
    print(f"\n终位置 {robot.get_joint_positions(joint_names=jn)[0]:.4f}（初始 {p0:.4f}）")
finally:
    try:
        robot.request_shutdown()
        robot.wait_for_shutdown()
    except Exception:
        pass
    try:
        robot.destroy()
    except Exception:
        pass
print("✓ 探针完成，已优雅退出")
