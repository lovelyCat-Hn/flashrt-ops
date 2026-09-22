"""
【测试 A】Windows 静态自检：不联网，只摆固定 home 姿态，只 mj_forward
预期：机器人纹丝不动，终端每秒打印的 qpos 漂移恒为 0
  - 漂移 > 0 或机器人瘫倒 → Windows 上跑的不是修正版代码（查文件是否保存/路径对不对）
  - 漂移恒 0 且姿态保持 → viewer 没问题，继续【测试 B】
"""
import time
import mujoco
import mujoco.viewer

MJCF_PATH = r"D:\work\project_file\galbot_one_golf_description\mjcf\galbot_one_golf_fixed_base.xml"

HOME = {
    "head_joint1": 0.0, "head_joint2": 0.0,
    "left_arm_joint1": 2.0, "left_arm_joint2": -1.5, "left_arm_joint3": -0.6,
    "left_arm_joint4": -1.7, "left_arm_joint5": 0.0, "left_arm_joint6": -0.8, "left_arm_joint7": 0.0,
    "right_arm_joint1": -2.0, "right_arm_joint2": 1.5, "right_arm_joint3": 0.6,
    "right_arm_joint4": 1.7, "right_arm_joint5": 0.0, "right_arm_joint6": 0.8, "right_arm_joint7": 0.0,
    "leg_joint1": 0.5, "leg_joint2": 1.5, "leg_joint3": 1.0, "leg_joint4": 0.0, "leg_joint5": 0.0,
}

model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

for n, v in HOME.items():
    if n in names:
        data.qpos[names.index(n)] = v
mujoco.mj_forward(model, data)
snapshot = data.qpos.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        viewer.sync()
        if time.time() - t0 >= 1.0:
            print(f"qpos 最大漂移: {abs(data.qpos - snapshot).max():.6f}  (应恒为 0)")
            t0 = time.time()
        time.sleep(0.03)
