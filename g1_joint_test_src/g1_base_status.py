"""
电流基线测量：在机器人完全静止时，记录每个关节的电流/力矩基线
运行 10 秒取平均，作为后续"突增检测"的参考点
"""
import time
from galbot_sdk.g1 import GalbotRobot

robot = GalbotRobot()
try:
    robot.init()
    time.sleep(5)

    joint_names = robot.get_joint_names(
        only_active_joint=True, joint_groups=["left_arm"]
    )
    print(f"开始测量 left_arm 基线电流（10 秒）...")
    print(f"  期间请勿触碰机器人，保持完全静止\n")

    samples = {n: [] for n in joint_names}
    for _ in range(100):  # 100 次 × 100ms = 10 秒
        states = robot.get_joint_states(
            joint_groups=["left_arm"], joint_names=[]
        )
        for n, s in zip(joint_names, states):
            samples[n].append(abs(s.current))
        time.sleep(0.1)

    print(f"{'joint':<18}{'mean(A)':>10}{'max(A)':>10}{'std(A)':>10}")
    print("-" * 48)
    for n, vals in samples.items():
        mean = sum(vals) / len(vals)
        mx = max(vals)
        # 简易标准差
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = variance ** 0.5
        print(f"{n:<18}{mean:>10.3f}{mx:>10.3f}{std:>10.3f}")

    print(
        "\n→ 建议阈值: baseline_mean + 2.0A (突增检测) "
        "/ 8.0A (绝对上限)"
    )

finally:
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
