"""
【测试 B】Jetson 诊断发送端：不初始化 SDK，往 9999 端口发 30Hz 固定关节值
21 个值各不相同（0.00 ~ 0.20），姿态稳定但能一眼看出配对是否串位
用法：先停掉 jetson_sender.py（占着 9999 端口），再跑本脚本
"""
import json
import socket
import time

PORT = 9999
SEND_HZ = 30

SDK_NAMES = [
    'head_joint1', 'head_joint2',
    'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
    'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6', 'left_arm_joint7',
    'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3',
    'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6', 'right_arm_joint7',
    'leg_joint1', 'leg_joint2', 'leg_joint3', 'leg_joint4', 'leg_joint5',
]
# 第 i 个关节 → i/100，值恒定且各不相同
FIXED_JOINTS = {n: i / 100.0 for i, n in enumerate(SDK_NAMES)}

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)
print(f"等待连接 0.0.0.0:{PORT} ...（发送恒定假数据）")

conn, addr = server.accept()
print(f"✓ 客户端: {addr}")
msg = json.dumps({"ts": 0, "joints": FIXED_JOINTS}) + "\n"
frame = 0
try:
    while True:
        try:
            conn.sendall(msg.encode())
            frame += 1
            if frame % 30 == 0:
                print(f"  已发 {frame} 帧（内容恒定不变）")
            time.sleep(1.0 / SEND_HZ)
        except (BrokenPipeError, ConnectionResetError):
            print("⚠ 断开，等待重连")
            conn, addr = server.accept()
            print(f"✓ 重连: {addr}")
except KeyboardInterrupt:
    pass
finally:
    conn.close()
    server.close()
