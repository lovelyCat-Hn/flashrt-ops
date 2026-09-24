#!/usr/bin/env python
"""G1 夹爪点动工具：只动夹爪，绝不触碰任何臂/腿/头关节。

用途：摆位/上电后单独开闭夹爪（如 task0 起始需要双爪闭合而臂已就位不宜再动）。
默认闭合（pct 0）；--open 张开；--pct 任意百分比。反馈滞后 ~6.3s（2026-09-23
实测），发令后固定等待 8s 再读终态，判"没动"前别提前下结论。

用法:
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py --left     # 只闭合左爪
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py --right --open
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py            # 双爪闭合
  ~/holy/run.sh ~/holy/scripts/inference/g1_grip.py --left --pct 50
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
side = ap.add_mutually_exclusive_group()
side.add_argument("--left", action="store_true", help="只动左爪（默认双爪）")
side.add_argument("--right", action="store_true", help="只动右爪（默认双爪）")
ap.add_argument("--open", dest="open_", action="store_true", help="张开（等价 --pct 100）")
ap.add_argument("--pct", type=float, default=0.0, help="目标宽度百分比 0=闭合 100=张开（默认 0）")
ap.add_argument("--speed", type=float, default=0.05, help="夹爪线速度 m/s（config [gripper].speed 同款）")
ap.add_argument("--effort", type=float, default=30.0, help="夹爪力矩 N（config [gripper].effort 同款）")
ap.add_argument("--wmin", type=float, default=0.0005, help="0%% 对应宽度 m（2026-09-23 标定）")
ap.add_argument("--wmax", type=float, default=0.1200, help="100%% 对应宽度 m（2026-09-23 标定）")
args = ap.parse_args()

sys.stdout.reconfigure(line_buffering=True)

pct = 100.0 if args.open_ else max(0.0, min(args.pct, 100.0))
sides = ["left_gripper", "right_gripper"]
if args.left:
    sides = ["left_gripper"]
elif args.right:
    sides = ["right_gripper"]
target = args.wmin + pct / 100.0 * (args.wmax - args.wmin)

from galbot_sdk.g1 import GalbotRobot, G1JointGroup, ControlStatus  # noqa: E402

robot = GalbotRobot()
robot.init()
time.sleep(2)
print(f"✓ SDK 初始化完成。目标：{'/'.join(sides)} → {pct:.0f}%（{target * 1000:.1f} mm），"
      f"速度 {args.speed} m/s，力矩 {args.effort} N")

for s in sides:
    st = robot.get_gripper_state(getattr(G1JointGroup, s))
    cur = f"{st.width * 1000:.1f} mm" if st is not None else "读取失败（查 RT/急停）"
    print(f"  {s} 当前: {cur}")

input("⚠ 回车下发夹爪指令（只动夹爪，臂不动）...")
for s in sides:
    status = robot.set_gripper_command(getattr(G1JointGroup, s), target,
                                       args.speed, args.effort, False)
    ok = status == ControlStatus.SUCCESS
    print(f"  {s} 指令 {'SUCCESS ✅' if ok else f'❌ {status}'}")

print("等待反馈消化（滞后 ~6.3s，固定 8s）...")
for i in range(4, 0, -1):
    time.sleep(2)
    line = []
    for s in sides:
        st = robot.get_gripper_state(getattr(G1JointGroup, s))
        line.append(f"{s}={st.width * 1000:.1f}mm(moving={st.is_moving})" if st else f"{s}=读失败")
    print(f"  [{i * 2}s] {'  '.join(line)}")

robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
sys.stdout.flush()
os._exit(0)                              # SDK 残留线程不退，强杀（行缓冲已刷）
