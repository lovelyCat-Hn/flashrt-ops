"""假数据发送端：不开真机/不碰 SDK，供 s1_viewer.py 离线开发与链路自检
（自 G1 jetson_constant_sender.py 迁移，17 关节 S1 布局，全部零位）

用法（任何有 python3 的机器）:
  python3 s1_constant_sender.py [--port 9999]

viewer 端连上后应看到 S1 直立零位姿态（torso_lift_joint1=0 m 是躯干降到底，
模型里若显得"缩着"属正常）。⚠ 零位只是链路自检口径，不代表真机安全姿态。
"""
import argparse
import json
import socket
import time

SDK_NAMES = (
    [f"left_arm_joint{i}" for i in range(1, 8)]
    + [f"right_arm_joint{i}" for i in range(1, 8)]
    + ["torso_joint1", "head_joint1", "head_joint2"]
)
SEND_HZ = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"假数据端就绪 {args.host}:{args.port}（{len(SDK_NAMES)} 关节全零 @ {SEND_HZ} Hz）")

    conn = None
    period = 1.0 / SEND_HZ
    frame = 0
    try:
        conn, addr = server.accept()
        print(f"✓ 客户端已连接: {addr}")
        conn.settimeout(5.0)
        while True:
            t0 = time.time()
            try:
                msg = json.dumps({
                    "ts": time.time_ns(),
                    "joints": {n: 0.0 for n in SDK_NAMES},
                }) + "\n"
                conn.sendall(msg.encode("utf-8"))
                frame += 1
                if frame % 30 == 0:
                    print(f"  已发送 {frame} 帧")
            except (BrokenPipeError, ConnectionResetError):
                print("⚠ 客户端断开，等待重连...")
                conn.close()
                conn, addr = server.accept()
                print(f"✓ 重连成功: {addr}")
            sleep_t = period - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print(f"\n→ 退出，共发送 {frame} 帧")
    finally:
        if conn:
            conn.close()
        server.close()


if __name__ == "__main__":
    main()
