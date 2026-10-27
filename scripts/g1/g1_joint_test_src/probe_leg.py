"""
最小代码 + 限位校验：精确控制指定关节
"""
import time
from galbot_sdk.g1 import GalbotRobot

robot = GalbotRobot()
try:
    robot.init()
    time.sleep(2)

    # === 最小代码 ===
    target_names = ["leg_joint1"]
    target_pos   = [0.05]

    # === 启动控制器 ===
    if not robot.get_active_controller("leg"):
        robot.start_controller("leg")

    # === 发送 ===
    status = robot.set_joint_positions(
        joint_positions=target_pos,
        joint_names=target_names,
    )
    print(f"set_joint_positions → {status}")

finally:
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
