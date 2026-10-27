"""
Windows 端 viewer v2：实时镜像 + 离线姿态设计 双模式

【两种模式】
  mirror 实时镜像: 持续用真机关节角覆盖仿真（数字孪生监控）
  free   冻结设计: 启动时抓一帧真机姿态作起点，之后不再覆盖，
                   可在 viewer 里随意摆弄机械臂，**绝不影响真机**
                   （本脚本没有任何向 Jetson/真机发送指令的代码路径）

【用法】
  python windows_viewer.py                 # 默认 mirror 模式
  python windows_viewer.py --mode free     # 启动即进入冻结设计模式

【终端命令】（在运行脚本的终端里输入，回车生效）
  m                     切到实时镜像
  f                     切到冻结设计（停止真机覆盖）
  p                     打印当前仿真姿态（21 关节 dict，可直接粘进脚本）
  set <关节名> <弧度>    精确设置关节，如: set left_arm_joint1 1.5
  ik left|right x y z   逆运动学：让该侧夹爪TCP到达世界坐标(x,y,z)米
                        （自动解 7 个关节角；自动切到 free 模式；未收敛则还原）
  q                     退出

【滑条面板（推荐）】
  python windows_viewer.py --mode free --panel
  会弹出 21 关节滑条窗口（范围=模型限位），拖滑条 = 精确动对应关节；
  「读入当前仿真姿态」把仿真姿态读回滑条；
  「仿真同步到真机」把最新真机姿态写入仿真+滑条；
  「对比 仿真vs真机」在终端逐关节打印差值（>0.02rad 标⚠️），调试一致性用。
  注：viewer 自带 Ctrl+拖拽在 launch_passive 下部分版本无效，滑条是可靠方式。

【核心规则】只调 mj_forward（摆姿态），绝不调 mj_step（跑物理）！
"""
import argparse
import json
import socket
import threading
import time
import mujoco
import mujoco.viewer
import numpy as np

JETSON_IP = "192.168.1.88"
JETSON_PORT = 9999
MJCF_PATH = r"D:\work\project_file\galbot_one_golf_description\mjcf\galbot_one_golf_fixed_base.xml"

SDK_NAMES = [
    'head_joint1', 'head_joint2',
    'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
    'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6', 'left_arm_joint7',
    'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3',
    'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6', 'right_arm_joint7',
    'leg_joint1', 'leg_joint2', 'leg_joint3', 'leg_joint4', 'leg_joint5',
]

# ============ 线程间共享状态 ============
mode = "mirror"          # "mirror" | "free"
pending_sets = {}        # set 命令暂存，viewer 线程统一应用
lock = threading.Lock()
quit_flag = False
first_frame_applied = False   # 第一帧真机姿态是否已写入（面板等它到位再初始化）
latest_real = {}              # 真机最新姿态（每帧更新，free 模式也更新，供对比/同步用）


def solve_ik_pos(model, data, body_name, target, pairs,
                 max_iter=150, tol=0.003):
    """位置级逆运动学（阻尼最小二乘）。
    pairs: [(qpos下标, (下限, 上限)), ...] 只动这些关节；原地修改 data.qpos。
    返回 (是否收敛, 末次位置误差)。"""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return False, -1.0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    lam2 = 0.05 ** 2
    qidx = [p[0] for p in pairs]
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        err = target - data.xpos[bid]
        if np.linalg.norm(err) < tol:
            return True, float(np.linalg.norm(err))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        J = jacp[:, qidx]                      # 全是铰链关节: dof 下标 == qpos 下标
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), err)
        n = np.linalg.norm(dq)
        if n > 0.25:                           # 单步限幅，防甩
            dq *= 0.25 / n
        for k, (qi, (lo, hi)) in enumerate(pairs):
            data.qpos[qi] = min(max(data.qpos[qi] + dq[k], lo), hi)
    mujoco.mj_forward(model, data)
    return False, float(np.linalg.norm(target - data.xpos[bid]))


def input_thread(data, model, sdk_to_mj, limits):
    """终端命令线程"""
    global mode, quit_flag
    print("\n=== 终端命令 ===")
    print("  m  实时镜像 | f  冻结设计 | p  打印当前姿态")
    print("  set <关节名> <弧度>   如: set left_arm_joint1 1.5（超限位会被拒绝）")
    print("  ik left|right x y z  让该侧夹爪TCP到达世界坐标(米)，自动切冻结模式")
    print("  q  退出")
    print("================\n")
    while not quit_flag:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        if cmd == 'm':
            with lock:
                mode = "mirror"
            print("→ 实时镜像模式（真机数据持续覆盖）")
        elif cmd == 'f':
            with lock:
                mode = "free"
            print("→ 冻结设计模式（真机不再覆盖，可自由拖拽）")
        elif cmd == 'p':
            with lock:
                print("\nsim_pose = {")
                for n in SDK_NAMES:
                    if n in sdk_to_mj:
                        print(f'    "{n}": {data.qpos[sdk_to_mj[n]]:+.4f},')
                print("}\n")
        elif cmd == 'set' and len(parts) == 3:
            name, angle = parts[1], parts[2]
            if name in sdk_to_mj:
                try:
                    v = float(angle)
                except ValueError:
                    print("✗ 角度必须是数字（弧度）")
                    v = None
                if v is not None:
                    lo, hi = limits.get(name, (-3.14, 3.14))
                    if hi > lo and not (lo - 1e-6 <= v <= hi + 1e-6):
                        print(f"⚠ {v:+.4f} 超出 {name} 限位 [{lo:+.3f}, {hi:+.3f}]，已拒绝")
                    else:
                        with lock:
                            pending_sets[name] = v
                        print(f"→ {name} = {v:+.4f}")
            else:
                print(f"✗ 未知关节: {name}")
        elif cmd == 'ik' and len(parts) == 5:
            side = parts[1].lower()
            if side not in ("left", "right"):
                print("✗ 用法: ik left|right x y z（单位: 米，世界坐标系）")
            else:
                try:
                    target = np.array([float(parts[2]), float(parts[3]),
                                       float(parts[4])])
                except ValueError:
                    print("✗ 坐标必须是数字（米，世界坐标系）")
                    target = None
                if target is not None:
                    body_name = f"{side}_gripper_tcp_link"
                    prefix = f"{side}_arm"
                    # IK 期间锁住主循环，避免并发改 qpos
                    with lock:
                        mode = "free"          # 否则镜像下一帧就把结果覆盖了
                        backup = data.qpos.copy()
                        pairs = [(sdk_to_mj[n], limits[n])
                                 for n in SDK_NAMES if n.startswith(prefix)]
                        ok, errv = solve_ik_pos(model, data, body_name,
                                                target, pairs)
                        if ok:
                            for qi, _ in pairs:
                                data.qvel[qi] = 0
                    if ok:
                        print(f"✓ IK 收敛（TCP 距目标 {errv * 1000:.1f} mm）")
                        for n in [x for x in SDK_NAMES if n.startswith(prefix)]:
                            print(f"    {n} = {data.qpos[sdk_to_mj[n]]:+.4f}")
                        print("→ 已切到冻结模式；滑条请点「读入当前仿真姿态」刷新")
                    else:
                        with lock:
                            data.qpos[:] = backup
                            mujoco.mj_forward(model, data)
                        print(f"✗ IK 未收敛（末次误差 {errv * 1000:.1f} mm），"
                              f"姿态已还原（目标可能超出台面可达范围）")
        elif cmd == 'q':
            quit_flag = True
            print("→ 退出")
        else:
            print("✗ 未知命令（m / f / p / set / q）")


def panel_thread(data, model, sdk_to_mj):
    """滑条控制面板（tkinter 独立线程）：21 关节各一根滑条，范围 = 模型限位"""
    import tkinter as tk

    # 等第一帧真机姿态写入 qpos 后再建滑条（离线最多等 3 秒）
    t0 = time.time()
    while not first_frame_applied and not quit_flag and time.time() - t0 < 3.0:
        time.sleep(0.05)

    root = tk.Tk()
    root.title("关节滑条（冻结模式下用；镜像模式会被真机数据覆盖）")

    guard = {"syncing": True}   # 程序设值时跳过回调，避免误写 pending_sets

    def make_cb(name):
        def cb(val):
            if not guard["syncing"]:
                with lock:
                    pending_sets[name] = float(val)
        return cb

    sliders = {}
    col_of = {"head": 0, "left_arm": 0, "right_arm": 1, "leg": 1}
    row = {0: 0, 1: 0}
    for prefix in ["head", "left_arm", "right_arm", "leg"]:
        col = col_of[prefix]
        base = col * 3
        tk.Label(root, text=f"— {prefix} —").grid(
            row=row[col], column=base, columnspan=3, sticky="w", padx=4)
        row[col] += 1
        for n in [x for x in SDK_NAMES if x.startswith(prefix)]:
            jid = sdk_to_mj[n]
            lo, hi = model.jnt_range[jid]
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                lo, hi = -3.14, 3.14
            s = tk.Scale(root, from_=lo, to=hi, resolution=0.01, length=230,
                         label=n, orient="horizontal", command=make_cb(n))
            s.set(float(data.qpos[jid]))
            # 两端标出限位数值
            tk.Label(root, text=f"{lo:+.2f}", fg="#666").grid(
                row=row[col], column=base, sticky="e")
            s.grid(row=row[col], column=base + 1, sticky="ew", padx=2)
            tk.Label(root, text=f"{hi:+.2f}", fg="#666").grid(
                row=row[col], column=base + 2, sticky="w")
            sliders[n] = s
            row[col] += 1

    def read_back():
        guard["syncing"] = True
        for n, s in sliders.items():
            s.set(float(data.qpos[sdk_to_mj[n]]))
        guard["syncing"] = False

    def apply_real():
        """把最新真机姿态写入仿真 + 滑条（free 模式下 = 重新选起点）"""
        if not latest_real:
            print("✗ 尚未收到真机数据")
            return
        guard["syncing"] = True
        with lock:
            for n, v in latest_real.items():
                if n in sliders:
                    sliders[n].set(float(v))
                    pending_sets[n] = float(v)
        guard["syncing"] = False
        print("→ 仿真已同步到真机最新姿态")

    def compare():
        """仿真 vs 真机 逐关节对比，输出到终端（不改变任何状态）"""
        if not latest_real:
            print("✗ 尚未收到真机数据")
            return
        with lock:
            real = dict(latest_real)
        print(f"\n{'关节':20s} {'仿真':>9s} {'真机':>9s} {'差值':>8s}")
        worst = 0.0
        for n in SDK_NAMES:
            if n not in sdk_to_mj or n not in real:
                continue
            s_val = float(data.qpos[sdk_to_mj[n]])
            r_val = float(real[n])
            d = s_val - r_val
            worst = max(worst, abs(d))
            flag = "  ⚠️超差" if abs(d) > 0.02 else ""
            print(f"{n:20s} {s_val:+9.4f} {r_val:+9.4f} {d:+8.4f}{flag}")
        print(f"最大偏差: {worst:.4f} rad (≈{worst / 3.14159 * 180:.2f}°)"
              f"  [超差阈值 0.02 rad ≈ 1.15°]\n")

    btn_row = max(row.values()) + 1
    tk.Button(root, text="读入当前仿真姿态", command=read_back).grid(
        row=btn_row, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
    tk.Button(root, text="仿真同步到真机", command=apply_real).grid(
        row=btn_row, column=3, columnspan=3, sticky="ew", padx=4, pady=4)
    tk.Button(root, text="对比 仿真vs真机（输出到终端）", command=compare).grid(
        row=btn_row + 1, column=0, columnspan=6, sticky="ew", padx=4, pady=4)

    guard["syncing"] = False
    root.mainloop()


def main():
    global mode, quit_flag, first_frame_applied
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mirror", "free"], default="mirror",
                    help="mirror=实时镜像 free=冻结设计（默认 mirror）")
    ap.add_argument("--panel", action="store_true",
                    help="弹出 21 关节滑条控制面板")
    args = ap.parse_args()
    mode = args.mode

    # 1. 加载模型（物理禁用但保留碰撞计算：接触只提示不响应）
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_GRAVITY |
                               mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    # 注意：不禁用 mjDSBL_CONTACT —— mj_forward 会算出接触点（data.ncon），
    # 用于穿透报警；因为没有 mj_step，接触不会产生物理弹开。
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)]
    sdk_to_mj = {n: joint_names.index(n) for n in SDK_NAMES if n in joint_names}
    # 各关节限位（供滑条范围 + set 命令校验用）
    limits = {}
    for n in SDK_NAMES:
        if n in sdk_to_mj:
            lo, hi = model.jnt_range[sdk_to_mj[n]]
            lo, hi = float(lo), float(hi)
            limits[n] = (lo, hi) if hi > lo else (-3.14, 3.14)
    print(f"模型: {model.njnt} joints | SDK 关节映射 {len(sdk_to_mj)}/21")

    # 2. 连接 Jetson
    sock = None
    try:
        sock = socket.create_connection((JETSON_IP, JETSON_PORT), timeout=5)
        sock.settimeout(5.0)
        print(f"✓ 已连接 Jetson {JETSON_IP}:{JETSON_PORT}")
    except OSError as e:
        print(f"⚠ 连不上 Jetson（{e}），以纯离线模式继续（set 命令可用）")

    # 3. 启动终端命令线程 + 滑条面板（可选）
    threading.Thread(target=input_thread, args=(data, model, sdk_to_mj, limits),
                     daemon=True).start()
    if args.panel:
        threading.Thread(target=panel_thread, args=(data, model, sdk_to_mj),
                         daemon=True).start()

    # 4. 主循环
    buf = b""
    frame = 0
    offline = sock is None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 尝试在 3D 窗口里画出接触点/接触力（版本不支持则静默跳过，
        # 也可在 viewer 左侧 Visualization 面板手动勾选 Contact point/force）
        opt = getattr(viewer, "opt", None)
        if opt is not None:
            try:
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            except Exception:
                pass

        prev_in_collision = False
        while viewer.is_running() and not quit_flag:
            t0 = time.time()

            # 应用 set 命令（离线/冻结/镜像模式下都可用）
            with lock:
                if pending_sets:
                    for n, v in pending_sets.items():
                        data.qpos[sdk_to_mj[n]] = v
                        data.qvel[sdk_to_mj[n]] = 0
                    pending_sets.clear()
                    mujoco.mj_forward(model, data)
                    viewer.sync()

            # 接收真机数据（冻结模式下照收不覆盖，保持链路活着）
            if not offline:
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        print("⚠ Jetson 断开，转纯离线模式")
                        offline = True
                    else:
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            msg = json.loads(line.decode("utf-8"))
                            with lock:
                                latest_real.update(msg["joints"])
                                # mirror 持续覆盖；free 只抓第一帧作起点，之后冻结
                                if mode == "mirror" or not first_frame_applied:
                                    for n, p in msg["joints"].items():
                                        if n in sdk_to_mj:
                                            data.qpos[sdk_to_mj[n]] = p
                                            data.qvel[sdk_to_mj[n]] = 0
                                    if not first_frame_applied:
                                        first_frame_applied = True
                                        if mode == "free":
                                            print("✓ 已抓取真机姿态作为设计起点（已冻结，可自由摆弄）")
                except socket.timeout:
                    pass
                except (ConnectionResetError, json.JSONDecodeError) as e:
                    print(f"⚠ {e}，转纯离线模式")
                    offline = True

            # 只摆姿态，绝不 mj_step
            mujoco.mj_forward(model, data)
            viewer.sync()
            frame += 1

            # 碰撞报警（只提示不阻挡）：接触点数变化时打印涉及的身体部件
            in_collision = data.ncon > 0
            if in_collision != prev_in_collision:
                if in_collision:
                    pairs = []
                    for ci in range(min(data.ncon, 5)):
                        b1 = model.geom_bodyid[data.contact[ci].geom1]
                        b2 = model.geom_bodyid[data.contact[ci].geom2]
                        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or "?"
                        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or "?"
                        pairs.append(f"{n1} ↔ {n2}")
                    print(f"⚠️ 自碰报警: {data.ncon} 个接触点 — " + "；".join(pairs)
                          + "（仅提示，不会物理阻挡）")
                else:
                    print("✓ 碰撞解除")
                prev_in_collision = in_collision

            # 无数据可收时（离线）限制刷新率
            if offline:
                sleep_t = 1 / 30 - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    if sock:
        sock.close()
    print(f"✓ 退出，共渲染 {frame} 帧")


if __name__ == "__main__":
    main()
