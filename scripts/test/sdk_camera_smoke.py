#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sdk_camera_smoke.py — 用 galbot_sdk 一次性抓全部相机并保存，验证链路。"""
import os
import sys
import time

sys.path.insert(0, "/data/galbot/lib")
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sdk_cam_test"
os.makedirs(OUT, exist_ok=True)

COLOR_SET = [
    (SensorType.HEAD_LEFT_CAMERA, "head_left"),
    (SensorType.HEAD_RIGHT_CAMERA, "head_right"),
    (SensorType.LEFT_ARM_CAMERA, "left_color"),
    (SensorType.RIGHT_ARM_CAMERA, "right_color"),
]
DEPTH_SET = [
    (SensorType.LEFT_ARM_DEPTH_CAMERA, "left_depth"),
    (SensorType.RIGHT_ARM_DEPTH_CAMERA, "right_depth"),
]

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def decode_rgb(msg):
    arr = np.frombuffer(msg["data"], np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    robot = GalbotRobot()
    enable = {s for s, _ in COLOR_SET} | {s for s, _ in DEPTH_SET}
    ok = robot.init(enable)
    print("init:", ok)
    if not ok:
        os._exit(1)
    time.sleep(5)

    results = []
    for st, name in COLOR_SET:
        try:
            msg = robot.get_rgb_data(st)
            if not msg or "data" not in msg:
                print("%-11s ✗ 无数据" % name)
                results.append((name, False))
                continue
            img = decode_rgb(msg)
            if img is None:
                print("%-11s ✗ JPEG 解码失败 (%d 字节)" % (name, len(msg["data"])))
                results.append((name, False))
                continue
            path = os.path.join(OUT, name + ".jpg")
            cv2.imwrite(path, img)
            print("%-11s ✓ %dx%d %d字节 -> %s" % (name, img.shape[1], img.shape[0],
                                                  len(msg["data"]), path))
            results.append((name, True))
        except Exception as e:
            print("%-11s ✗ 异常: %s" % (name, e))
            results.append((name, False))

    for st, name in DEPTH_SET:
        try:
            msg = robot.get_depth_data(st)
            if not msg or "data" not in msg:
                print("%-11s ✗ 无数据" % name)
                results.append((name, False))
                continue
            h, w = msg.get("height", 0), msg.get("width", 0)
            scale = msg.get("depth_scale", 1)
            arr = np.frombuffer(msg["data"], np.uint16)
            okk = h > 0 and w > 0 and arr.size == h * w
            if okk:
                np.save(os.path.join(OUT, name + ".npy"),
                        arr.reshape(h, w).astype(np.float32) / scale)
                print("%-11s ✓ %dx%d scale=%s %d字节" % (name, w, h, scale, len(msg["data"])))
            else:
                print("%-11s ✗ 尺寸不符 h=%s w=%s size=%d" % (name, h, w, arr.size))
            results.append((name, okk))
        except Exception as e:
            print("%-11s ✗ 异常: %s" % (name, e))
            results.append((name, False))

    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    good = sum(1 for _, okk in results if okk)
    print("==== %d/%d 路正常 ====" % (good, len(results)))
    sys.stdout.flush()
    os._exit(0 if good == len(results) else 1)


if __name__ == "__main__":
    main()
