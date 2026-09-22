"""
================================================================================
joint 状态实时查看工具
================================================================================

【用途】
  通过 SDK 的 DDS 缓存实时读取并显示 Galbot G1 各关节的位置/速度/电流等状态。
  数据来自 GalbotRobot 单例的 get_joint_positions() / get_joint_states()，
  内部是订阅 WBC topic 的缓存，几乎是实时的（采样延迟 < 1 个缓存周期）。

【前置】
  - 机器人端 WBC / 传感器链路已就绪
  - 急停已释放
  - PYTHONPATH 含 /data/galbot/lib（~/.bashrc 已自动配置）
  - 无需 conda / venv

【五种输出模式】

  ① --mode table   （默认）表格模式，列对齐，最清晰
     输出：
       [15:37:33.681] joint 状态
       joint                pos(rad)   pos(deg)
       --------------------------------------------
       head_joint1           +0.0000      +0.00°
       left_arm_joint1       +2.0000    +114.59°
       left_arm_joint2       -1.5500     -88.80°
       ...

  ② --mode details  完整状态表（5 列：pos/vel/acc/effort/current）
     输出：
       [15:37:33.681] joint 完整状态
                          pos(rad)  vel(r/s)  acc(r/s²)  eff(N·m)   cur(A)
       joint              ---------  ---------  ---------  ---------  ---------
       head_joint1          +0.0000    +0.0000    +0.0000    +0.0000    +0.0341
       left_arm_joint1      +2.0000    +0.0000    +0.0000    +0.0000    +3.3840
       ...

  ③ --mode grouped  按关节组分块（最容易扫读）
     输出：
       [15:37:33.681]
       ▸ head (2 关节)
         head_joint1          +0.0000      +0.00°
         head_joint2          +0.0000      +0.00°
       ▸ left_arm (7 关节)
         left_arm_joint1      +2.0000    +114.59°
         ...

  ④ --mode live     单行紧凑（高频观测 / 不重定向时）
     输出：
       1=+2.000 2=-1.550 3=-0.550 4=-1.700 5=+0.000 6=-0.800 7=+0.000 ...

  ⑤ --mode append   每帧一行（适合重定向到文件，不丢精度）
     输出：
       [15:37:33.681] +0.0000 +0.0000 +2.0000 -1.5500 -0.5500 -1.7000 ...
       [15:37:33.782] +0.0000 +0.0000 +2.0000 -1.5500 -0.5500 -1.7000 ...


【命令行参数】

  --groups  G1 G2 ...    要监控的关节组（默认 head left_arm right_arm leg）
  --hz      N            刷新频率（次/秒），默认 10
  --mode    MODE         table / details / grouped / live / append
  --duration SECS        运行时长，到点自动退出，0=无限（默认 0）
  --csv     FILE         把数据同时写入 CSV（含时间戳）
  --once                  只读一次就退出（用于快照 / 调试）


【常用用法速查】

  # 1. 看一眼当前姿态（最常用，快照）
  python3 read_joint_pos.py --mode grouped --once

  # 2. 实时表格观测（推荐，自动清屏）
  python3 read_joint_pos.py --mode table --hz 10

  # 3. 实时完整状态（含电流，调试驱动器用）
  python3 read_joint_pos.py --mode details --hz 5

  # 4. 高频观测（100Hz 关节状态）
  python3 read_joint_pos.py --mode live --hz 100

  # 5. 保存数据到 CSV（带时间戳）
  python3 read_joint_pos.py --mode table --csv joints_$(date +%H%M%S).csv

  # 6. 重定向日志（推荐 append 模式，stdout 纯净）
  python3 read_joint_pos.py --mode append --hz 50 > joints.log

  # 7. 只看一个组
  python3 read_joint_pos.py --mode grouped --groups left_arm --once

  # 8. 只跑 30 秒自动退出
  python3 read_joint_pos.py --mode table --duration 30


【输出示例（grouped 模式）】

  $ python3 read_joint_pos.py --mode grouped --once

    [15:37:33.681]

    ▸ head (2 关节)
      head_joint1            +0.0000      +0.00°
      head_joint2            +0.0000      +0.00°

    ▸ left_arm (7 关节)
      left_arm_joint1        +2.0000    +114.59°
      left_arm_joint2        -1.5500     -88.80°
      left_arm_joint3        -0.5500     -31.50°
      left_arm_joint4        -1.7000     -97.40°
      left_arm_joint5        +0.0000      +0.00°
      left_arm_joint6        -0.8000     -45.84°
      left_arm_joint7        +0.0000      +0.00°

    ▸ right_arm (7 关节)
      right_arm_joint1       -2.0000    -114.59°
      ...

    ▸ leg (5 关节)
      leg_joint1             +0.3000     +17.19°
      leg_joint2             +1.2000     +68.75°
      ...


【退出方式】

  - Ctrl+C：自动 request_shutdown → wait_for_shutdown → destroy）
  - --duration SECS：到点自动退出
  - --once：只读一次就退出


【提示】

  - pos 单位是 rad；显示同时给出 deg 便于直观判断
  - 关节电流（cur）静止时不为 0 是正常的（承重关节需保持扭矩）
  - 静止时 velocity / acceleration 应接近 0；非零可能是关节在动
  - stdout 是数据流，stderr 是 meta 信息（日志/启动/退出）—— 重定向时只存数据
================================================================================
"""
import argparse
import signal
import sys
import time
from datetime import datetime
from galbot_sdk.g1 import GalbotRobot


def parse_args():
    p = argparse.ArgumentParser(
        description="实时流式输出 joint 位置（多种可读模式）"
    )
    p.add_argument("--groups", nargs="+",
                   default=["head", "left_arm", "right_arm", "leg"],
                   help="要监控的关节组")
    p.add_argument("--hz", type=float, default=10.0,
                   help="刷新频率（次/秒）")
    p.add_argument("--mode",
                   choices=["table", "details", "grouped", "live", "append"],
                   default="table",
                   help="table=对齐表格, details=5 字段完整状态表, "
                        "grouped=按关节组分块, live=单行紧凑, append=每帧一行")
    p.add_argument("--duration", type=float, default=0.0,
                   help="运行时长（秒），0=无限")
    p.add_argument("--csv", type=str, default=None,
                   help="可选：把数据追加写到这个 CSV 文件")
    p.add_argument("--once", action="store_true",
                   help="只读一次就退出（用于快照）")
    return p.parse_args()


# ---------- 工具函数 ----------
def clear_screen():
    """ANSI 清屏 + 移到顶部（只在 TTY 时用）"""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def format_table_row(name: str, pos_rad: float, max_name_len: int) -> str:
    """格式化表格的一行：joint name + pos(rad) + pos(deg)"""
    return (f"  {name:<{max_name_len}}  "
            f"{pos_rad:+10.4f} rad  "
            f"{pos_rad * 57.2958:+9.2f}°")


def format_details_row(name: str, s, max_name_len: int) -> str:
    """完整状态表的一行（5 字段）"""
    return (f"  {name:<{max_name_len}}  "
            f"{s.position:+9.4f}  "
            f"{s.velocity:+9.4f}  "
            f"{s.acceleration:+9.4f}  "
            f"{s.effort:+9.4f}  "
            f"{s.current:+9.4f}")


def print_table_header(joint_names, ts_str: str):
    """打印表格头部"""
    max_name_len = max(len(n) for n in joint_names)
    print(f"\033[1m  [{ts_str}] joint 状态\033[0m")
    print(f"  {'joint':<{max_name_len}}  {'pos(rad)':>10}  {'pos(deg)':>10}")
    print(f"  {'-' * max_name_len}  {'-' * 10}  {'-' * 10}")


def print_details_header(joint_names, ts_str: str):
    """打印完整状态表头部"""
    max_name_len = max(len(n) for n in joint_names)
    print(f"\033[1m  [{ts_str}] joint 完整状态\033[0m")
    print(f"  {'':>{max_name_len}}  {'pos(rad)':>9}  {'vel(r/s)':>9}  "
          f"{'acc(r/s²)':>9}  {'eff(N·m)':>9}  {'cur(A)':>9}")
    print(f"  {'joint':<{max_name_len}}  {'-' * 9}  {'-' * 9}  "
          f"{'-' * 9}  {'-' * 9}  {'-' * 9}")


def group_joints_by_group(joint_names):
    """按前缀分组，返回 [(group_name, [joint_names_in_group]), ...]"""
    groups = {}
    order = []
    for n in joint_names:
        # 提取 group（去掉 _jointN 后缀）
        for prefix in ["left_arm", "right_arm", "head", "leg", "chassis"]:
            if n.startswith(prefix):
                if prefix not in groups:
                    groups[prefix] = []
                    order.append(prefix)
                groups[prefix].append(n)
                break
        else:
            # 自定义命名
            custom = n.rsplit("_joint", 1)[0] if "_joint" in n else n
            if custom not in groups:
                groups[custom] = []
                order.append(custom)
            groups[custom].append(n)
    return [(g, groups[g]) for g in order]


# ---------- 主循环 ----------
def main():
    args = parse_args()
    period = 1.0 / args.hz
    csv_fh = None

    robot = GalbotRobot()
    frame = 0  # 提前初始化:提前退出(sys.exit)时 finally 不会 UnboundLocalError
    try:
        if not robot.init():
            print("✗ robot.init() 失败", file=sys.stderr)
            sys.exit(1)
        print("✓ init 成功", file=sys.stderr)

        # 给 WBC/DDS 链路留启动时间
        time.sleep(5)

        joint_names = robot.get_joint_names(
            only_active_joint=True, joint_groups=args.groups
        )
        if not joint_names:
            print("✗ 关节名为空，请检查 WBC/急停", file=sys.stderr)
            sys.exit(1)

        # CSV 文件
        if args.csv:
            csv_fh = open(args.csv, "a", buffering=1)
            csv_fh.write("timestamp_ns," + ",".join(joint_names) + "\n")
            print(f"→ CSV 输出: {args.csv}", file=sys.stderr)

        is_tty = sys.stdout.isatty()
        print(
            f"→ 开始监控 {len(joint_names)} 个关节 @ {args.hz} Hz "
            f"| mode={args.mode} | Ctrl+C 退出",
            file=sys.stderr,
        )

        signal.signal(signal.SIGINT, signal.default_int_handler)

        t_start = time.time()
        frame = 0

        # 一次性快照模式
        if args.once:
            _print_once(robot, joint_names, args, frame=1)
            return

        while True:
            t0 = time.time()
            ts_ns = time.time_ns()
            ts_str = datetime.fromtimestamp(ts_ns / 1e9).strftime("%H:%M:%S.%f")[:-3]

            positions = robot.get_joint_positions(
                joint_names=joint_names
            )

            # CSV
            if csv_fh:
                csv_fh.write(f"{ts_ns}," + ",".join(f"{p:.6f}" for p in positions) + "\n")

            # 各种模式
            if args.mode == "table":
                if is_tty and frame % 5 == 0:
                    clear_screen()
                print_table_header(joint_names, ts_str)
                max_name_len = max(len(n) for n in joint_names)
                for n, p in zip(joint_names, positions):
                    print(format_table_row(n, p, max_name_len))
                print(flush=True)

            elif args.mode == "details":
                if is_tty and frame % 5 == 0:
                    clear_screen()
                states = robot.get_joint_states(
                    joint_names=joint_names
                )
                print_details_header(joint_names, ts_str)
                max_name_len = max(len(n) for n in joint_names)
                for n, s in zip(joint_names, states):
                    print(format_details_row(n, s, max_name_len))
                print(flush=True)

            elif args.mode == "grouped":
                if is_tty and frame % 5 == 0:
                    clear_screen()
                print(f"\033[1m  [{ts_str}]\033[0m")
                grouped = group_joints_by_group(joint_names)
                # 建立 name → position 映射
                pos_dict = dict(zip(joint_names, positions))
                for group_name, names_in_group in grouped:
                    print(f"\n  \033[1m▸ {group_name}\033[0m ({len(names_in_group)} 关节)")
                    max_name_len = max(len(n) for n in names_in_group)
                    for n in names_in_group:
                        p = pos_dict[n]
                        print(format_table_row(n, p, max_name_len))
                print(flush=True)

            elif args.mode == "live":
                line = " ".join(
                    f"{n.split('_joint')[-1] if '_joint' in n else n}={p:+.3f}"
                    for n, p in zip(joint_names, positions)
                )
                end_char = "\r" if is_tty else "\n"
                print(line, end=end_char, flush=True)

            else:  # append
                line = (
                    f"[{ts_str}] "
                    + " ".join(f"{p:+.4f}" for p in positions)
                )
                print(line, flush=True)

            frame += 1

            # 时长到点退出
            if args.duration and (t0 - t_start) >= args.duration:
                print(f"\n→ 已运行 {args.duration}s，自动退出", file=sys.stderr)
                break

            # 节流
            sleep_t = period - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n→ 收到 Ctrl+C，准备退出", file=sys.stderr)

    finally:
        print("→ 关停中...", file=sys.stderr)
        try:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()
        except Exception as e:
            print(f"  关停异常: {e}", file=sys.stderr)
        if csv_fh:
            csv_fh.close()
        print(f"✓ 已退出，共输出 {frame} 帧", file=sys.stderr)


def _print_once(robot, joint_names, args, frame):
    """一次性快照输出"""
    ts_ns = time.time_ns()
    ts_str = datetime.fromtimestamp(ts_ns / 1e9).strftime("%H:%M:%S.%f")[:-3]

    if args.mode in ("table", "grouped"):
        positions = robot.get_joint_positions(
            joint_names=joint_names
        )
        if args.mode == "table":
            print_table_header(joint_names, ts_str)
            max_name_len = max(len(n) for n in joint_names)
            for n, p in zip(joint_names, positions):
                print(format_table_row(n, p, max_name_len))
        else:  # grouped
            print(f"  [{ts_str}]")
            grouped = group_joints_by_group(joint_names)
            pos_dict = dict(zip(joint_names, positions))
            for group_name, names_in_group in grouped:
                print(f"\n  ▸ {group_name} ({len(names_in_group)} 关节)")
                max_name_len = max(len(n) for n in names_in_group)
                for n in names_in_group:
                    print(format_table_row(n, pos_dict[n], max_name_len))

    elif args.mode == "details":
        states = robot.get_joint_states(
            joint_names=joint_names
        )
        print_details_header(joint_names, ts_str)
        max_name_len = max(len(n) for n in joint_names)
        for n, s in zip(joint_names, states):
            print(format_details_row(n, s, max_name_len))

    else:  # live / append
        positions = robot.get_joint_positions(
            joint_names=joint_names
        )
        line = " ".join(
            f"{n.split('_joint')[-1] if '_joint' in n else n}={p:+.3f}"
            for n, p in zip(joint_names, positions)
        )
        print(f"[{ts_str}] {line}")


if __name__ == "__main__":
    main()
