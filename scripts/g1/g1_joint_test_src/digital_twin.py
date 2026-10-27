"""
================================================================================
G1 数字孪生：真实机器人 ↔ MuJoCo 双向同步
================================================================================

【前置】
- mujoco 已装
- LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1（解决 aarch64 TLS 溢出）
- OMP_NUM_THREADS=1
- 已设置 PYTHONPATH 含 /data/galbot/lib

【用法】
  python3 digital_twin.py

【交互】
  - m: 切到镜像模式（真实→仿真，默认）
  - i: 切到试探模式（用户控制仿真关节）
  - show: 打印当前 sim_joint_angles
  - set <joint> <angle>: 修改试探模式的某个关节角度
  - 关闭窗口或 Ctrl+C: 退出

【模式说明】
  镜像模式: data.qpos ← 真实机器人当前关节角度（每帧同步）
  试探模式: data.qpos ← sim_joint_angles 字典（用户可改）
================================================================================
"""
import time
import threading
import mujoco
import mujoco.viewer
from galbot_sdk.g1 import GalbotRobot


# ============ 配置 ============
MJCF_PATH = "/home/galbot/galbot_one_golf_description/mjcf/galbot_one_golf_fixed_base.xml"
JOINT_GROUPS = ["head", "left_arm", "right_arm", "leg"]
SYNC_HZ = 30

# SDK 标准 home（21 个关节）
sim_joint_angles = {
    "head_joint1": 0.0,
    "head_joint2": 0.0,
    "left_arm_joint1": 2.0,
    "left_arm_joint2": -1.5,
    "left_arm_joint3": -0.6,
    "left_arm_joint4": -1.7,
    "left_arm_joint5": 0.0,
    "left_arm_joint6": -0.8,
    "left_arm_joint7": 0.0,
    "right_arm_joint1": -2.0,
    "right_arm_joint2": 1.5,
    "right_arm_joint3": 0.6,
    "right_arm_joint4": 1.7,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": 0.8,
    "right_arm_joint7": 0.0,
    "leg_joint1": 0.5,
    "leg_joint2": 1.5,
    "leg_joint3": 1.0,
    "leg_joint4": 0.0,
    "leg_joint5": 0.0,
}

# 全局状态
mode = "mirror"
mode_lock = threading.Lock()


def input_thread():
    """后台监听键盘输入切换模式/调整试探角度"""
    global mode, sim_joint_angles
    print("\n=== 键盘指令 ===")
    print("  m           - 切到镜像模式")
    print("  i           - 切到试探模式")
    print("  show        - 显示当前 sim_joint_angles")
    print("  set <j> <a> - 试探模式下设关节角度")
    print("  例如: set left_arm_joint1 1.5")
    print("================\n")
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue
        with mode_lock:
            if line == 'm':
                mode = "mirror"
                print("→ 镜像模式（真实→仿真）")
            elif line == 'i':
                mode = "interactive"
                print("→ 试探模式（仿真独立）")
            elif line == 'show':
                print("\n当前 sim_joint_angles:")
                for n, a in sim_joint_angles.items():
                    print(f'  "{n}": {a:+.4f},')
                print()
            elif line.startswith('set '):
                parts = line.split()
                if len(parts) == 3:
                    _, name, angle = parts
                    if name in sim_joint_angles:
                        try:
                            sim_joint_angles[name] = float(angle)
                            print(f"→ {name} = {float(angle):+.4f}")
                        except ValueError:
                            print("✗ 角度必须是数字")
                    else:
                        print(f"✗ 未知关节: {name}")
                else:
                    print("用法: set <joint_name> <angle>")


def main():
    global mode

    # 1. 加载 MJCF
    print(f"加载 MJCF: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity = (0.0, 0.0, 0.0)  # 关闭重力
    print(f"✓ 模型: {model.njnt} 关节, {model.nbody} bodies")

    # 2. 关节映射
    urdf_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                        for i in range(model.njnt)]
    sdk_to_mj = {}
    for sdk_name in sim_joint_angles.keys():
        if sdk_name in urdf_joint_names:
            sdk_to_mj[sdk_name] = urdf_joint_names.index(sdk_name)
    print(f"✓ SDK→MJ 映射: {len(sdk_to_mj)}/{len(sim_joint_angles)} 个")
    if len(sdk_to_mj) < len(sim_joint_angles):
        missing = set(sim_joint_angles.keys()) - set(sdk_to_mj.keys())
        print(f"  未映射: {missing}")

    # 3. 启动键盘监听
    threading.Thread(target=input_thread, daemon=True).start()

    # 4. 启动 SDK
    robot = GalbotRobot()
    try:
        if not robot.init():
            print("✗ init 失败")
            return
        print("✓ SDK init 成功")
        time.sleep(5)

        # 5. viewer 主循环
        print(f"\n启动 MuJoCo viewer（{SYNC_HZ} Hz）...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            frame = 0
            while viewer.is_running():
                t0 = time.time()

                with mode_lock:
                    current_mode = mode

                if current_mode == "mirror":
                    # === 镜像模式：从真实机器人读 ===
                    real_names = robot.get_joint_names(
                        only_active_joint=True, joint_groups=JOINT_GROUPS
                    )
                    # 按名字显式读取（顺序有保证）；group 模式返回顺序可能与 names 不一致
                    real_pos = robot.get_joint_positions(joint_names=real_names)
                    pos_dict = dict(zip(real_names, real_pos))
                    for sdk_name, mj_id in sdk_to_mj.items():
                        if sdk_name in pos_dict:
                            data.qpos[mj_id] = pos_dict[sdk_name]
                            data.qvel[mj_id] = 0
                else:
                    # === 试探模式：用 sim_joint_angles ===
                    for sdk_name, mj_id in sdk_to_mj.items():
                        if sdk_name in sim_joint_angles:
                            data.qpos[mj_id] = sim_joint_angles[sdk_name]
                            data.qvel[mj_id] = 0

                # 关键：只算运动学，不积分动力学
                mujoco.mj_forward(model, data)
                viewer.sync()

                frame += 1
                sleep_t = 1.0 / SYNC_HZ - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

            print(f"\n✓ Viewer 关闭，共 {frame} 帧")

    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        print("✓ SDK 关停")


if __name__ == "__main__":
    main()
