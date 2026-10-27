"""S1 端：读真机关节角，通过 TCP 推流给 Windows 渲染端（自 G1 jetson_sender.py 迁移）

与 G1 版协议完全同构（JSON 行、按关节名键控、:9999 @30Hz），Windows 端
s1_viewer.py 配对。本脚本纯只读（get_* only），不含任何指令路径。

S1 与 G1 差异（全部按 docs/s1-joint-api-quickref.md 核过）：
  - SDK: galbot_sdk.s1（1.10.0，本机 /data/galbot/lib），init(set()) 写法照抄
    ~/yuqiz/s1_teleop 真机跑通代码
  - 关节组: head/left_arm/right_arm/torso（G1 的 leg 组不存在）→ 17 关节
  - 夹爪不在此推流（S1 夹爪是独立 API + 物理宽度，URDF 基座亦无夹爪几何；
    待端工具描述并入模型后再扩展）
  - ⚠ 读关节数据一律按显式名字（group 模式与 names 配对会全错，G1 教训）

用法（本机，flash_pyrt311 已含 SDK 3.11 绑定）:
  PYTHONPATH=/data/galbot/lib ~/miniforge3/envs/flash_pyrt311/bin/python s1_sender.py
  # Windows 端: python s1_viewer.py --host <本机IP>（hostname -I 第一个值别用错网卡）
"""
import json
import time
import socket
from galbot_sdk.s1 import GalbotRobot

HOST = "0.0.0.0"   # 监听所有网卡
PORT = 9999        # 与 G1 孪生同端口（不同时跑两套才不冲突）
JOINT_GROUPS = ["head", "left_arm", "right_arm", "torso"]   # 17 关节，无 leg
SEND_HZ = 30


def main():
    robot = GalbotRobot()
    server = None
    conn = None
    frame = 0
    try:
        if not robot.init(set()):          # yuqiz s1_teleop 同款（enabled_modules 空=不拉感知）
            print("✗ init 失败")
            return
        print("✓ SDK init 成功")
        time.sleep(5)

        # 启动时读一次关节名（应为 17 个；此后每帧按这串显式名字读，
        # 返回顺序与本次调用配对——勿改 group 模式读法）
        joint_names = robot.get_joint_names(
            only_active_joint=True, joint_groups=JOINT_GROUPS
        )
        print(f"发送 {len(joint_names)} 个 body 关节 @ {SEND_HZ} Hz: {joint_names}")

        # TCP 服务端
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"等待 Windows 客户端连接 {HOST}:{PORT} ...")

        period = 1.0 / SEND_HZ
        conn, addr = server.accept()
        print(f"✓ 客户端已连接: {addr}")
        conn.settimeout(5.0)

        while True:
            t0 = time.time()
            try:
                pos = robot.get_joint_positions(joint_names=joint_names)
                msg = json.dumps({
                    "ts": time.time_ns(),
                    "joints": dict(zip(joint_names, pos)),
                }) + "\n"
                conn.sendall(msg.encode("utf-8"))
                frame += 1

                if frame % 30 == 0:
                    print(f"  已发送 {frame} 帧 @ {SEND_HZ} Hz")

            except (BrokenPipeError, ConnectionResetError):
                print("⚠ 客户端断开，等待重连...")
                conn.close()
                conn, addr = server.accept()
                print(f"✓ 重连成功: {addr}")

            sleep_t = period - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print(f"\n→ Ctrl+C，共发送 {frame} 帧")

    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        if server:
            server.close()
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()


if __name__ == "__main__":
    main()
