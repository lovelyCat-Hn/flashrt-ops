"""
Jetson 端：读真机关节角，通过 TCP 推流给 Windows 渲染端
"""
import json
import time
import socket
from galbot_sdk.g1 import GalbotRobot

HOST = "0.0.0.0"   # 监听所有网卡
PORT = 9999
JOINT_GROUPS = ["head", "left_arm", "right_arm", "leg"]
SEND_HZ = 30


def main():
    robot = GalbotRobot()
    server = None
    conn = None
    frame = 0
    try:
        if not robot.init():
            print("✗ init 失败")
            return
        print("✓ SDK init 成功")
        time.sleep(5)

        # 读一次关节名（启动时确认 21 个）
        joint_names = robot.get_joint_names(
            only_active_joint=True, joint_groups=JOINT_GROUPS
        )
        print(f"发送 {len(joint_names)} 个 body 关节 @ {SEND_HZ} Hz")

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

        frame = 0
        while True:
            t0 = time.time()
            try:
                pos = robot.get_joint_positions(
                    joint_names=joint_names
                )
                msg = json.dumps({
                    "ts": time.time_ns(),
                    "joints": dict(zip(joint_names, pos)),
                }) + "\n"
                conn.sendall(msg.encode("utf-8"))
                frame += 1

                # 每 30 帧打印一次状态
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
            conn.close()
        except Exception:
            pass
        server.close()
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()


if __name__ == "__main__":
    main()
