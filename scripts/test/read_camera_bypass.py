#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_camera_bypass.py — 绕过 galbot_sdk，用机器自带 embosa Python 绑定直读相机话题。

背景（本机 camera transport 不匹配）：SDK 1.8.1 的 participant 与 rc88 capture
守护进程配对失败，相机话题收不到；但同版本(9.1.2)的 embosa Python 绑定 +
LARGE_DATA_TRANSPORT(6) reader 是 fusion/swallows 实证能配对的路径
（relocalization_server 也用该绑定正常收 lidar）。

依赖（机器上现成，无需安装）：
  - 绑定:   /userdata/update/manual_update/lib/python3.8.10  (py3.8, .so.9.1.2)
  - proto:  /data/galbot/lib/python3/site-packages/galbot     (sensor_proto/image_pb2)
  - 注意:   系统 protobuf 4.x 与旧式生成代码不兼容，必须设
            PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python（脚本已自动设置）。

用法:
  python3 read_camera_bypass.py                         # 默认读左臂 color，10s
  python3 read_camera_bypass.py --list                  # 列出已知话题
  python3 read_camera_bypass.py TOPIC [TOPIC...] --seconds 8 --save /tmp/x.jpg
  python3 read_camera_bypass.py --transport 5           # 实验其它 transport
"""
import os

# 必须在 import 任何 protobuf 之前设置（纯 Python 实现兼容旧式 _pb2 生成代码）
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import time
import argparse

BINDING_DIR = "/userdata/update/manual_update/lib/python3.8.10"
PROTO_PKG = "/data/galbot/lib/python3/site-packages"
for p in (BINDING_DIR, PROTO_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

KNOWN_TOPICS = [
    "/left_arm_camera/color/image_raw",
    "/right_arm_camera/color/image_raw",
    "/front_head_camera/left_color/image_raw",
    "/front_head_camera/right_color/image_raw",
    "/left_arm_camera/depth/image_raw",
    "/right_arm_camera/depth/image_raw",
    "/front_head_camera/depth/image_raw",
    "/left_front_surround/color/image_raw",
    "/right_front_surround/color/image_raw",
    "/left_rear_surround/color/image_raw",
    "/right_rear_surround/color/image_raw",
]

# 话题名 -> 简短输出名
SHORT = {
    "/left_arm_camera/color/image_raw": "left_color",
    "/right_arm_camera/color/image_raw": "right_color",
    "/front_head_camera/left_color/image_raw": "head_left",
    "/front_head_camera/right_color/image_raw": "head_right",
    "/left_arm_camera/depth/image_raw": "left_depth",
    "/right_arm_camera/depth/image_raw": "right_depth",
    "/front_head_camera/depth/image_raw": "head_depth",
}


def stamp_str(msg):
    try:
        ts = msg.header.timestamp
        return "%d.%09d" % (ts.seconds, ts.nanos)
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("topics", nargs="*", default=None,
                    help="相机话题名，缺省 = /left_arm_camera/color/image_raw")
    ap.add_argument("--list", action="store_true", help="列出已知话题名并退出")
    ap.add_argument("--seconds", type=float, default=10.0, help="接收时长(秒)")
    ap.add_argument("--save-dir", default=None,
                    help="保存各话题首帧到此目录（颜色=jpg, 深度=raw16）")
    ap.add_argument("--transport", type=int, default=None,
                    help="覆盖 transport_type（缺省 LARGE_DATA_TRANSPORT）")
    ap.add_argument("--depth", type=int, default=4, help="reader 队列深度")
    args = ap.parse_args()

    if args.list:
        for t in KNOWN_TOPICS:
            print(t)
        return
    topics = args.topics or ["/left_arm_camera/color/image_raw"]

    import embosa_python as ep
    from galbot.sensor_proto import image_pb2

    transport = args.transport if args.transport is not None else ep.LARGE_DATA_TRANSPORT
    print("transport_type = %d (LARGE_DATA=%d SHM_ONLY=%d DEFAULT_UDP_SHM=%d)"
          % (transport, ep.LARGE_DATA_TRANSPORT, ep.SHM_ONLY_TRANSPORT,
             ep.DEFAULT_UDP_SHM_TRANSPORT))

    ep.EmbosaInit()
    node = ep.CreateNode("camera_bypass_reader")
    qos = ep.QosConf.CreateIntraCoreQos(
        transport,
        ep.DATA_SHARING_AUTO,
        ep.BEST_EFFORT_RELIABILITY,
        ep.KEEP_LAST_HISTORY,
        args.depth,
        ep.VOLATILE_DURABILITY,
    )

    readers = []
    stats = {t: {"n": 0, "first": None, "last": None, "bytes": 0, "saved": False}
             for t in topics}

    def make_cb(topic):
        def on_msg(msg):
            s = stats[topic]
            now = time.time()
            if s["first"] is None:
                s["first"] = now
                print("[%s] 首帧: enc=%s %dx%d %d字节 ts=%s"
                      % (SHORT.get(topic, topic), msg.encoding,
                         msg.width, msg.height, len(msg.data), stamp_str(msg)))
                if args.save_dir:
                    ext = "raw16" if "16" in (msg.encoding or "") else "jpg"
                    path = os.path.join(
                        args.save_dir,
                        "%s_%s.%s" % (SHORT.get(topic, topic.strip("/").replace("/", "_")),
                                      msg.encoding or "frame", ext))
                    try:
                        with open(path, "wb") as f:
                            f.write(msg.data)
                        print("[%s] 已保存首帧 -> %s" % (SHORT.get(topic, topic), path))
                    except OSError as e:
                        print("[%s] 保存失败: %s" % (topic, e))
            s["n"] += 1
            s["last"] = now
            s["bytes"] += len(msg.data)
        return on_msg

    for t in topics:
        r = node.CreateSerializationReader(image_pb2.Image, t, make_cb(t), qos)
        readers.append(r)
    print("已创建 %d 个 reader，等待配对/收帧 (%.0fs)..." % (len(readers), args.seconds))

    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        for r in readers:
            try:
                node.DeleteSerializationReader(r)
            except Exception:
                pass
        ep.Clear()

    print("\n===== 结果 =====")
    any_data = False
    for t in topics:
        s = stats[t]
        if s["n"] >= 2:
            fps = (s["n"] - 1) / (s["last"] - s["first"])
            any_data = True
            print("%-24s %6d 帧  %.1f fps  %.1f KB/帧"
                  % (SHORT.get(t, t), s["n"], fps, s["bytes"] / s["n"] / 1024.0))
        else:
            print("%-24s %6d 帧  ✗ 未收到数据" % (SHORT.get(t, t), s["n"]))
    if not any_data:
        print("\n提示: 0 帧时查 /userdata/log/embosa/embosa_hpu_camera_bypass_reader_*"
              "_statistics.log 看 reader 是否配对（match 状态），"
              "或换 --transport 5 / 0 试验。")
    # embosa C 层线程可能残留，直接退出保证干净
    sys.stdout.flush()
    os._exit(0 if any_data else 1)


if __name__ == "__main__":
    main()
