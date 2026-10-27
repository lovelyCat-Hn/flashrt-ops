import time
from galbot_sdk.g1 import GalbotRobot, GalbotMotion

robot = GalbotRobot()
motion = GalbotMotion()    # 独立单例
try:
    robot.init()
    motion.init()           # 注意：必须 init
    time.sleep(5)

    # 启动所有需要的控制器
    robot.start_controller("all")

    # 移到 SDK 预定义的零位（带碰撞检查）
    status = motion.move_whole_body_joint_zero(
        is_blocking=True,
        leg_head_speed_rad_s=0.2,
        leg_head_timeout_s=15.0,
    )
    print(f"move_whole_body_joint_zero → {status}")

finally:
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()