"""
================================================================================
G1 末端 IK 运动脚本：用 SDK 自带求解器让夹爪到达目标位姿
================================================================================

【做什么】
  用 SDK 求解器驱动末端运动，支持"相对当前末端平移"（最常用）、
  "绝对坐标"和只解不动的 dry-run。
  手臂：GalbotMotion 规划器（IK+碰撞检查+执行一步到位）。
  leg ：规划器不收 leg，走混合路径——SDK IK 解 5 个关节角 → 直控执行
        （与 move_whole_body_joint_zero 的 leg 路径同款）。

【接口链】
  get_end_effector_pose_on_chain()  读当前末端位姿（move 的基准）
  set_end_effector_pose()           IK解算+轨迹规划+碰撞检查+执行（一步到位）
  inverse_kinematics()              只解算不执行（solve 命令，验证可达性）
  move_whole_body_joint_zero()      回 home

【坐标系】
  base_link（机器人底盘原点，z 向上），单位米；与 Windows viewer 世界系同源。
  位姿格式 7 维: [x, y, z, qx, qy, qz, qw]（位置 + 四元数）。

【用法】
  python3 jetson_ik_move.py

  进交互后：
    pose                      打印各链当前末端位姿（双臂 + leg）
    move left_arm|right_arm dx dy dz   臂末端相对当前平移 (dx,dy,dz) 米
    move leg dx dy dz         躯干相对当前平移（驮全身，单步限 0.05m）
    goto left_arm|right_arm|leg x y z  末端到绝对坐标（保持当前朝向）
    rot left_arm|right_arm|leg r p y   末端朝向相对旋转（度，RPY=侧倾/俯仰/偏航，
                                       base_link 系，位置不变，单次限 30°）
    solve left_arm|right_arm|leg x y z 只解算不执行（打印解出的关节角）
    home                      回 SDK 预定义零位（带碰撞检查）
    q                         退出

【安全】
  - 启动需 y 确认；单步平移限幅 0.15 m，超出拒绝
  - 每次执行均 enable_collision_check=True（自碰 + 已加载障碍物）
  - 阻塞执行 + 30s 超时，完成前不会接受下一条命令

【前置】
  ~/.bashrc 已含 LD_PRELOAD=libgomp.so.1 和 OMP_NUM_THREADS=1；
  PYTHONPATH=/data/galbot/lib（或运行时指定）。
================================================================================
"""
import math
import time

from galbot_sdk.g1 import (ControlStatus, GalbotMotion, GalbotRobot,
                           MotionStatus, Parameter)

# 单条 move/goto 的单步平移限幅（米）：leg 驮着整个上半身，限得更死
MAX_STEP = {"left_arm": 0.15, "right_arm": 0.15, "leg": 0.05}
MOVE_TIMEOUT_S = 30.0  # 单次执行超时

CHAINS = ("left_arm", "right_arm", "leg")
N_JOINTS = {"left_arm": 7, "right_arm": 7, "leg": 5}


def fmt_pose(pose):
    """位姿列表 → 可读字符串。7 维按 [x,y,z,qx,qy,qz,qw] 解析，附 RPY 欧拉角。"""
    if pose and len(pose) == 7:
        r, p, y = quat_to_rpy_deg(pose[3:7])
        return (f"pos=({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}) m  "
                f"quat=({pose[3]:+.3f}, {pose[4]:+.3f}, {pose[5]:+.3f}, {pose[6]:+.3f})"
                f"  [RPY=({r:+.1f}°, {p:+.1f}°, {y:+.1f}°)]")
    return str(pose)


def quat_mul(a, b):
    """四元数乘法，统一 SDK 约定 [qx,qy,qz,qw]。q = a⊗b 表示先做 b 再做 a（世界系）。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_from_axis_deg(axis, deg):
    """绕单位化 axis 轴转 deg 度的四元数（返回 [qx,qy,qz,qw]）。"""
    h = math.radians(deg) / 2
    n = math.sqrt(sum(v * v for v in axis)) or 1.0
    s = math.sin(h) / n
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h))


def quat_to_rpy_deg(q):
    """[qx,qy,qz,qw] → RPY 欧拉角（度），世界系 X(roll)-Y(pitch)-Z(yaw) 依次旋转。"""
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def quat_angle_deg(q0, q1):
    """两个朝向之间的夹角（度）。"""
    dot = sum(a * b for a, b in zip(q0, q1))
    return math.degrees(2 * math.acos(min(1.0, abs(dot))))


def get_current_pose(motion, chain):
    """读某链末端（*_end_effector_mount_link）当前位姿；失败返回 None。"""
    status, pose = motion.get_end_effector_pose(f"{chain}_end_effector_mount_link")
    if status != MotionStatus.SUCCESS or not pose:
        print(f"✗ 读取 {chain} 末端位姿失败: {motion.status_to_string(status)}")
        return None
    return list(pose)


def fk_chain(motion, chain, angles):
    """用 FK 验算一组关节角对应的末端位姿；失败返回 None。"""
    st, pose = motion.forward_kinematics(
        f"{chain}_end_effector_mount_link", "base_link", {chain: list(angles)})
    if st != MotionStatus.SUCCESS:
        return None
    return list(pose)[:7]


def check_step(cur, target, chain):
    """与当前位姿的平移距离限幅校验（leg 的限幅更小）。"""
    if len(cur) != 7 or len(target) != 7:
        return False
    dist = sum((target[i] - cur[i]) ** 2 for i in range(3)) ** 0.5
    limit = MAX_STEP[chain]
    if dist > limit:
        print(f"⚠ 单步平移 {dist:.3f} m 超过 {chain} 限幅 {limit} m，"
              f"请拆成多次小步（安全考虑，已拒绝）")
        return False
    return True


def execute_ee_pose(motion, robot, chain, target):
    """执行末端运动。手臂走规划器（IK+碰撞检查）；leg 规划器不收执行，
    用 SDK IK 解角度后走直控（move_whole_body_joint_zero 的 leg 同款路径）。"""
    if chain == "leg":
        return _execute_leg(motion, robot, target)
    status = motion.set_end_effector_pose(
        target_pose=target,
        # 帧名 = 链名（官方教程同款写法；实测 "EndEffector" 和 link 名都会 INVALID_INPUT）
        end_effector_frame=chain,
        reference_frame="base_link",
        enable_collision_check=True,
        is_blocking=True,
        timeout=MOVE_TIMEOUT_S,
        params=Parameter(),   # 必须显式传，缺省会被参数校验拒绝
    )
    if status != MotionStatus.SUCCESS:
        print(f"✗ 执行失败: {motion.status_to_string(status)}")
        return False
    new_pose = get_current_pose(motion, chain)
    print(f"✓ 完成，当前末端: {fmt_pose(new_pose) if new_pose else '?'}")
    return True


def _execute_leg(motion, robot, target):
    """leg 链：SDK IK 解 5 个关节角 → set_joint_positions 按名字直控执行。"""
    status, joints = motion.inverse_kinematics(
        target_pose=target, chain_names=["leg"],
        target_frame="EndEffector", reference_frame="base_link",
        enable_collision_check=True)
    if status != MotionStatus.SUCCESS:
        print(f"✗ leg IK 无解: {motion.status_to_string(status)}")
        return False
    names = _chain_joints("leg")
    angles = list(joints.get("leg") or next(iter(joints.values())))
    if len(angles) != len(names):
        print(f"✗ IK 解出 {len(angles)} 角 vs 关节名 {len(names)} 个，放弃执行")
        return False
    # 求解器对够不着的目标可能原样返回初值（假成功），FK 验算真实残差
    fk_pose = fk_chain(motion, "leg", angles)
    if fk_pose is None:
        print("✗ 解算结果 FK 验算失败，放弃执行")
        return False
    resid = sum((fk_pose[i] - target[i]) ** 2 for i in range(3)) ** 0.5
    if resid > 0.01:
        print(f"⚠ 解算器返回的解实际误差 {resid * 1000:.0f} mm —— 目标超出 leg "
              f"工作空间（升降臂只能前后/上下，不能横移 y），已拒绝执行")
        return False
    ang = quat_angle_deg(fk_pose[3:7], target[3:7])
    if ang > 5:
        print(f"⚠ 解算出的姿态残差 {ang:.1f}° —— 目标朝向超出 leg 能力，已拒绝执行")
        return False
    print("→ leg 目标关节角: " + ", ".join(f"{a:+.4f}" for a in angles))
    cstatus = robot.set_joint_positions(
        joint_positions=angles, joint_names=names,
        is_blocking=True, speed_rad_s=0.2, timeout_s=MOVE_TIMEOUT_S)
    if cstatus != ControlStatus.SUCCESS:
        print(f"✗ leg 直控执行失败: {cstatus}")
        return False
    new_pose = get_current_pose(motion, "leg")
    print(f"✓ 完成，当前 leg 末端: {fmt_pose(new_pose) if new_pose else '?'}")
    return True


def parse_floats(parts, n):
    """解析 n 个浮点数；失败打印提示并返回 None。"""
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        print(f"✗ 需要 {n} 个数字")
        return None
    if len(vals) != n:
        print(f"✗ 需要正好 {n} 个数字")
        return None
    return vals


def confirm_safety():
    print("注意：本脚本会让真机手臂运动！")
    print("  1) 急停按钮已释放且触手可及")
    print("  2) 手臂前后左右无人员/障碍物/易碎品")
    while True:
        key = input("确认以上两点，是否继续？(y/n): ").strip().lower()
        if key == "y":
            return True
        if key == "n":
            return False
        print("请输入 y 或 n")


def main():
    if not confirm_safety():
        print("已取消")
        return

    robot = GalbotRobot()
    motion = GalbotMotion()
    if not robot.init():
        print("✗ robot.init 失败")
        return
    if not motion.init():
        print("✗ motion.init 失败（运动学模块未起来）")
        return
    print("✓ SDK 初始化完成，等待控制器就绪...")
    time.sleep(5)
    robot.start_controller("all")

    print("\n=== 末端 IK 运动交互 ===")
    print("  pose                  打印各链当前末端位姿（双臂 + leg）")
    print("  move left_arm|right_arm dx dy dz   臂末端相对当前平移（米，小步）")
    print("  move leg dx dy dz     躯干相对当前平移（驮全身，单步限 0.05m）")
    print("  goto left_arm|right_arm|leg x y z  末端到绝对坐标（保持当前朝向）")
    print("  rot left_arm|right_arm|leg r p y   末端朝向相对旋转（度，单次限30°）")
    print("  solve left_arm|right_arm|leg x y z 只解算不执行（验证可达性）")
    print("  home                  回零位")
    print("  q                     退出")
    print("========================\n")

    for c in CHAINS:
        p = get_current_pose(motion, c)
        print(f"{c} 当前末端: {fmt_pose(p) if p else '?'}")
    print()

    while True:
        try:
            line = input("ik> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "q":
            break

        elif cmd == "pose":
            for c in CHAINS:
                p = get_current_pose(motion, c)
                print(f"{c}: {fmt_pose(p) if p else '?'}")

        elif cmd in ("move", "goto", "solve") and len(parts) >= 2:
            chain = parts[1].lower()
            if chain not in CHAINS:
                print(f"✗ 链名须是 {' 或 '.join(CHAINS)}")
                continue
            need = 3
            nums = parse_floats(parts[2:2 + need], need)
            if nums is None:
                print(f"用法: {cmd} left_arm|right_arm|leg dx dy dz" if cmd == "move"
                      else f"用法: {cmd} left_arm|right_arm|leg x y z")
                continue

            if cmd == "solve":
                cur = get_current_pose(motion, chain)
                if cur is None:
                    continue
                target = nums + cur[3:7]          # 只给位置，朝向沿用当前
                status, joints = motion.inverse_kinematics(
                    target_pose=target,
                    chain_names=[chain],
                    target_frame="EndEffector",
                    reference_frame="base_link",
                    enable_collision_check=True,
                )
                if status != MotionStatus.SUCCESS:
                    print(f"✗ 无解: {motion.status_to_string(status)}"
                          f"（目标可能不可达/碰撞）")
                    continue
                print(f"✓ 可达，{chain} 关节解：")
                for name, angles in joints.items():
                    for jn, a in zip(_chain_joints(chain), angles):
                        print(f"    {jn} = {a:+.4f}")
                sol = list(next(iter(joints.values())))
                fk_pose = fk_chain(motion, chain, sol)
                if fk_pose:
                    resid = sum((fk_pose[i] - target[i]) ** 2
                                for i in range(3)) ** 0.5
                    if resid > 0.01:
                        print(f"⚠ 注意：该解实际误差 {resid * 1000:.0f} mm，"
                              f"目标可能超出此链工作空间（执行会被拒绝）")

            else:
                cur = get_current_pose(motion, chain)
                if cur is None:
                    continue
                if cmd == "move":
                    target = [cur[i] + nums[i] for i in range(3)] + cur[3:7]
                else:  # goto
                    target = nums + cur[3:7]
                if not check_step(cur, target, chain):
                    continue
                label = (f"{chain} 平移 "
                         f"({nums[0]:+.3f}, {nums[1]:+.3f}, {nums[2]:+.3f})"
                         if cmd == "move" else f"{chain} 到 "
                         f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})")
                print(f"→ {label} ...")
                execute_ee_pose(motion, robot, chain, target)

        elif cmd == "rot" and len(parts) == 5:
            chain = parts[1].lower()
            if chain not in CHAINS:
                print(f"✗ 链名须是 {' 或 '.join(CHAINS)}")
                continue
            degs = parse_floats(parts[2:5], 3)
            if degs is None:
                print("用法: rot <链名> roll pitch yaw（度，相对当前朝向）")
                continue
            if max(abs(d) for d in degs) > 30:
                print("⚠ 单次旋转超过 30°，请分次小步（安全考虑，已拒绝）")
                continue
            cur = get_current_pose(motion, chain)
            if cur is None:
                continue
            # 世界系依次绕 X(roll)→Y(pitch)→Z(yaw) 旋转，再复合到当前朝向前
            dq = quat_mul(quat_from_axis_deg((0, 0, 1), degs[2]),
                          quat_from_axis_deg((0, 1, 0), degs[1]))
            dq = quat_mul(dq, quat_from_axis_deg((1, 0, 0), degs[0]))
            tq = quat_mul(dq, cur[3:7])
            n = math.sqrt(sum(v * v for v in tq))
            target = cur[:3] + [v / n for v in tq]
            print(f"→ {chain} 旋转 RPY({degs[0]:+g}°, {degs[1]:+g}°, "
                  f"{degs[2]:+g}°)，位置不变 ...")
            execute_ee_pose(motion, robot, chain, target)

        elif cmd == "home":
            print("→ 回零位（腿/头直控 + 双臂规划，带碰撞检查）...")
            status = motion.move_whole_body_joint_zero(
                is_blocking=True,
                leg_head_speed_rad_s=0.2,
                leg_head_timeout_s=15.0,
            )
            print(f"{'✓ 已回零位' if status == MotionStatus.SUCCESS else '✗ 失败: ' + motion.status_to_string(status)}")

        else:
            print("✗ 未知命令（pose / move / goto / solve / home / q）")

    print("\n=== 安全关闭 ===")
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("✓ 已退出")


def _chain_joints(chain):
    """链名 → 该链关节名列表（leg 5 个，臂 7 个）。"""
    return [f"{chain}_joint{i}" for i in range(1, N_JOINTS[chain] + 1)]


if __name__ == "__main__":
    main()
