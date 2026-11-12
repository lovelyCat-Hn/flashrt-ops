#!/usr/bin/env python3
"""读取左臂摄像头图像（只读零运动，不起控制器、不发任何指令）。

用法：
  conda activate flash_pyrt311
  PYTHONPATH=/userdata/update/manual_update/lib \\
      python scripts/s1/probes/read_left_arm_camera.py --n 3 --out /tmp/left_arm_cam

要点：
  - robot.init 只带 {LEFT_ARM_CAMERA}（pyi：只初始化所需传感器，减开销）。
  - 取图路径与 run_s1_loop_traj.py 主循环 grab_views 同源：
    get_rgb_data(JPEG, once=True) 等一帧新鲜 JPEG → cv2.imdecode。
  - 存盘 = 相机 JPEG 原始字节（零重编码，保真）；顺带解码打印分辨率/
    帧间像素差（相邻帧逐字节相同 = 流冻结疑点，同主循环冻结守卫判据）。
  - 退出走 SDK 优雅三连（request_shutdown/wait_for_shutdown/destroy），
    缺 = 退出段 Segmentation fault（10-28 replay 实证）。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from galbot_sdk.s1 import GalbotRobot, RgbOutputFormat, SensorType


def main() -> None:
    ap = argparse.ArgumentParser(
        description="读左臂摄像头图像（只读零运动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--n", type=int, default=1, help="抓帧数（默认 1）")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="帧间隔秒（默认 0=连抓；once=True 自身阻塞等新鲜帧，"
                         "调用耗时即帧到达间隔）")
    ap.add_argument("--out", default="/tmp/left_arm_cam", help="存图目录")
    args = ap.parse_args()

    robot = GalbotRobot()
    inited = False
    try:
        if not robot.init({SensorType.LEFT_ARM_CAMERA}):
            sys.exit("robot.init 失败（排查 RT 相机服务/采集栈）")
        inited = True
        time.sleep(2)  # 等 DMA 订阅起来、首帧就绪
        os.makedirs(args.out, exist_ok=True)

        prev = None
        for i in range(args.n):
            t0 = time.time()
            d = robot.get_rgb_data(SensorType.LEFT_ARM_CAMERA,
                                   RgbOutputFormat.JPEG, True)
            dt = (time.time() - t0) * 1000
            if not d or not d.get("data"):
                sys.exit(f"取图失败（第 {i + 1}/{args.n} 帧，耗时 {dt:.0f} ms）")
            jpg = bytes(d["data"])
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                sys.exit("JPEG 解码失败（数据非 JPEG？）")

            path = os.path.join(args.out, f"left_{i:02d}.jpg")
            with open(path, "wb") as f:
                f.write(jpg)

            line = (f"[{i + 1}/{args.n}] {img.shape[1]}x{img.shape[0]} "
                    f"{len(jpg) / 1024:.1f} KiB  取图+解码 {dt:.0f} ms  → {path}")
            if prev is not None:
                if np.array_equal(prev, img):
                    line += "  ⚠ 与上一帧逐字节相同（流冻结疑点）"
                else:
                    diff = float(np.abs(prev.astype(np.int16) -
                                        img.astype(np.int16)).mean())
                    line += f"  帧间像素差 {diff:.2f}（>0 即活流）"
            print(line, flush=True)
            if i == 0:  # 第一帧打印一次完整元数据键供判读
                meta = {k: v for k, v in d.items()
                        if k not in ("data",) and np.isscalar(v)}
                print(f"    元数据: {meta}")
                print(f"    header: {d.get('header')}")
            prev = img
            if args.interval > 0 and i + 1 < args.n:
                time.sleep(args.interval)

        print(f"\n完成：{args.n} 帧 → {args.out}/")
    finally:
        if inited:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()


if __name__ == "__main__":
    main()
