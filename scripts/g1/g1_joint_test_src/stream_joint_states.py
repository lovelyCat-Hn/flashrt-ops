"""
day1/1.5 — 非阻塞调用的安全模式（dI/dt 版）
按电流变化率而非绝对值判断碰撞
"""
import time
from collections import deque
from galbot_sdk.g1 import (
    GalbotRobot, ControlStatus, LogLevel, TrajectoryControlStatus
)

# ---------- 阈值（这是关键设计） ----------
RATE_WINDOW_MS = 100       # 用最近 100ms 算 dI/dt
DI_DT_THRESHOLD = 30.0     # A/s，单关节变化率阈值
CONFIRM_FRAMES = 3         # 连续 3 帧超阈值才判定（@20Hz = 0.15s）
POS_DEV_THRESHOLD = 0.3    # rad
MONITOR_HZ = 20
MAX_WAIT_S = 30.0
MONITOR_DT = 1.0 / MONITOR_HZ

# 每个关节的窗口大小 = RATE_WINDOW_MS / MONITOR_DT
WINDOW_SIZE = int(RATE_WINDOW_MS / MONITOR_DT * 1.0) + 1


# ---------- 默认测试关节组（命令行可覆盖） ----------
TEST_GROUPS = ["left_arm"]


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--groups", nargs="+", default=TEST_GROUPS,
        help="要测试的关节组（默认 left_arm）；可选: head left_arm right_arm leg 等",
    )
    return p.parse_args()


# ---------- 安全还原函数：best-effort，永不抛异常 ----------
def safe_restore(robot, joint_groups, original_pos, timeout_s=20.0):
    """
    尽力把机器人还原到 original_pos。
    - 内部 try/except 兜底，绝对不抛异常
    - 自动确保控制器在线
    - 失败也不抛，只打印
    """
    print(f"\n→ [safe_restore] 还原到起始位置 (timeout={timeout_s}s)...")
    try:
        # 先确保控制器在线
        for g in joint_groups:
            try:
                if not robot.get_active_controller(g):
                    print(f"    启动控制器 {g}...")
                    robot.start_controller(g)
            except Exception as e:
                print(f"    启动 {g} 控制器异常: {e}")

        # 还原（阻塞，确保到位）
        status = robot.set_joint_positions(
            joint_positions=original_pos,
            joint_groups=joint_groups,
            joint_names=[],
            is_blocking=True,
            speed_rad_s=0.1,
            timeout_s=timeout_s,
        )
        if status == ControlStatus.SUCCESS:
            print(f"  ✓ [safe_restore] 已还原到起始位置")
        else:
            print(f"  ⚠ [safe_restore] 还原返回 {status}（可能未完全到位）")
    except Exception as e:
        print(f"  ✗ [safe_restore] 还原异常: {e}")

# ---------- 方向配置（先用 probe_direction.py 探针后填写） ----------
# 每关节方向系数：+1 = 沿关节正方向；-1 = 反方向
# 把下面全部填成 +1，跑 probe_direction.py 看实际方向，再回来改
DIRECTION_SIGN = {
    "left_arm_joint1": +1,
    "left_arm_joint2": +1,
    "left_arm_joint3": +1,
    "left_arm_joint4": +1,
    "left_arm_joint5": +1,
    "left_arm_joint6": +1,
    "left_arm_joint7": +1,
}
OFFSET_RAD = 0.3   # 通用偏移幅度（如果只想动某些关节，下面用列表指定）


def main():
    # 解析命令行参数（--groups head left_arm ...）
    args = parse_args()
    print(f"→ 测试关节组: {args.groups}")

    robot = GalbotRobot()
    try:
        if not robot.init():
            exit("✗ init 失败")
        time.sleep(5)

        # 关节组可通过命令行参数覆盖：--groups head left_arm
        joint_groups = args.groups
        joint_names = robot.get_joint_names(
            only_active_joint=True, joint_groups=joint_groups
        )
        original_pos = robot.get_joint_positions(
            joint_groups=joint_groups, joint_names=[]
        )

        # ---------- 1. 发非阻塞指令 ----------
        # 用 DIRECTION_SIGN 表对每个关节分别应用方向系数
        target = [
            original_pos[i] + OFFSET_RAD * DIRECTION_SIGN.get(n, +1)
            for i, n in enumerate(joint_names)
        ]
        print(f"→ 发非阻塞指令: 目标偏移 ±{OFFSET_RAD} rad（按方向表）")
        print(f"  起点: {[f'{p:+.3f}' for p in original_pos]}")
        print(f"  目标: {[f'{p:+.3f}' for p in target]}")
        print(f"  方向: {[DIRECTION_SIGN.get(n, +1) for n in joint_names]}")

        # [诊断 ①] 检查每个关节组的活动控制器
        print(f"\n  [诊断 ①] 活动控制器检查:")
        for g in joint_groups:
            active = robot.get_active_controller(g)
            print(f"    {g}: 活动控制器 = {active!r}")
            if not active:
                print(f"    ⚠ {g} 无活动控制器，尝试启动...")
                st = robot.start_controller(g)
                print(f"      start_controller({g}) -> {st}")

        status = robot.set_joint_positions(
            joint_positions=target,
            joint_groups=joint_groups,
            joint_names=[],
            is_blocking=False,
            speed_rad_s=0.1,
            timeout_s=20.0,
        )
        if status != ControlStatus.SUCCESS:
            exit(f"✗ 下发失败: {status}")
        print(f"  ✓ 指令已下发 (status={status})")

        # [诊断 ②] 验证轨迹是否真的启动（位置变化 + WBC 错误日志）
        print(f"\n  [诊断 ②] 验证轨迹是否真的启动:")
        for label, delay in [("立即", 0.0), ("0.5s后", 0.5), ("2s后", 2.0)]:
            if delay > 0:
                time.sleep(delay)
            p = robot.get_joint_positions(joint_groups=joint_groups, joint_names=[])
            d = max(abs(a - b) for a, b in zip(p, target))
            print(f"    [{label:>6}] dev_to_target={d:.4f}rad  pos[0]={p[0]:+.4f}")

        errors = robot.get_log_information(
            timewindow_s=3, log_level=LogLevel.ERROR
        )
        if errors:
            print(f"  ⚠ WBC 错误日志: {errors}")
        else:
            print(f"  ✓ WBC 最近 3s 无 ERROR")

        # ---------- 2. 监控循环 ----------
        # 每个关节的电流历史窗口
        current_hist = {n: deque(maxlen=WINDOW_SIZE) for n in joint_names}
        # 每个关节的"超阈值"计数器
        spike_count = {n: 0 for n in joint_names}

        print(f"\n→ 监控中 (max {MAX_WAIT_S}s) "
              f"[dI/dt 阈值={DI_DT_THRESHOLD}A/s, "
              f"窗口={RATE_WINDOW_MS}ms, "
              f"确认={CONFIRM_FRAMES}帧]")
        t_start = time.time()

        while time.time() - t_start < MAX_WAIT_S:
            t0 = time.time()

            # ① 轨迹完成？
            statuses = robot.check_trajectory_execution_status(
                joint_groups=joint_groups
            )
            # 注意：TrajectoryControlStatus 是纯枚举，直接 == 比较；
            # 完成态叫 COMPLETED（SUCCESS 是 ControlStatus 的成员，不一样）。
            abnormal_states = {
                TrajectoryControlStatus.ERROR,
                TrajectoryControlStatus.STOPPED_UNREACHED,
                TrajectoryControlStatus.INVALID_INPUT,
                TrajectoryControlStatus.DATA_FETCH_FAILED,
            }
            if statuses:
                # 异常状态 → 立即停止
                bad = [s for s in statuses if s in abnormal_states]
                if bad:
                    print(f"\n  ✗ 异常轨迹状态: {bad}")
                    robot.stop_trajectory_execution()
                    break
                # 全部完成
                if all(s == TrajectoryControlStatus.COMPLETED for s in statuses):
                    print(f"\n  ✓ 轨迹完成 ({time.time()-t_start:.2f}s)")
                    break
                # 否则（RUNNING 等）继续监控，不 break

            # ② 取当前状态
            current = robot.get_joint_positions(
                joint_groups=joint_groups, joint_names=[]
            )
            states = robot.get_joint_states(
                joint_groups=joint_groups, joint_names=[]
            )

            # ③ 位置偏差（辅助信号，不作为主判断）
            max_dev = max(abs(c - t) for c, t in zip(current, target))
            pos_abnormal = max_dev > POS_DEV_THRESHOLD

            # ④ 核心：dI/dt 检测
            collided = False
            di_dt_max = 0.0
            worst_joint = None
            for n, s in zip(joint_names, states):
                current_hist[n].append(s.current)

                if len(current_hist[n]) < WINDOW_SIZE:
                    continue  # 还没攒够窗口数据

                # 算 dI/dt
                oldest = current_hist[n][0]
                newest = current_hist[n][-1]
                di_dt = abs(newest - oldest) / (RATE_WINDOW_MS / 1000.0)
                di_dt_max = max(di_dt_max, di_dt)

                # 计数器
                if di_dt > DI_DT_THRESHOLD:
                    spike_count[n] += 1
                    if spike_count[n] >= CONFIRM_FRAMES:
                        collided = True
                        worst_joint = n
                else:
                    spike_count[n] = max(0, spike_count[n] - 1)

            # ⑤ 错误日志（兜底）
            errors = robot.get_log_information(
                timewindow_s=2, log_level=LogLevel.ERROR
            )
            log_abnormal = bool(errors)

            # 综合判定：dI/dt 持续 + 位置偏差 + WBC 错误，三选一即停
            if collided or (pos_abnormal and di_dt_max > DI_DT_THRESHOLD * 0.6):
                reason = []
                if collided:
                    reason.append(f"dI/dt 突增 ({worst_joint}: max|ΔI/Δt|={di_dt_max:.1f}A/s)")
                if pos_abnormal:
                    reason.append(f"位置偏差 {max_dev:.3f}rad")
                if log_abnormal:
                    reason.append(f"WBC 错误: {errors}")
                print(f"\n  ✗ 异常: {' | '.join(reason)}")
                robot.stop_trajectory_execution()
                break

            # 实时显示
            cur_str = " ".join(
                f"{n.split('_joint')[1]}={s.current:+.2f}"
                for n, s in zip(joint_names, states)
            )
            print(
                f"  [{time.time()-t_start:5.2f}s] "
                f"dev={max_dev:.3f} max_dI/dt={di_dt_max:5.1f}A/s "
                f"[{cur_str}]",
                end="\r", flush=True,
            )

            # 节流
            sleep_t = MONITOR_DT - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
        else:
            print(f"\n  ⚠ 超时 {MAX_WAIT_S}s，强制停止")
            robot.stop_trajectory_execution()

        # ---------- 3. 还原（正常流程） ----------
        safe_restore(robot, joint_groups, original_pos, timeout_s=15.0)

    finally:
        # [兜底] 无论 try 块里发生了什么（异常 / break / Ctrl+C），
        # 都先尝试还原，再关停 SDK。
        # safe_restore 内部自带 try/except，不会再抛异常。
        safe_restore(robot, joint_groups, original_pos, timeout_s=20.0)

        print(f"\n→ 关停 SDK...")
        try:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()
        except Exception as e:
            print(f"  关停异常: {e}")


if __name__ == "__main__":
    main()
