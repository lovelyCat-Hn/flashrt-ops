#!/usr/bin/env python
"""场景摆位对位检查：抓三路相机当前帧，与训练集参考帧拼图对比（不加载模型）。

背景（2026-09-23 真机）：头部构图已接近训练，但模型仍输出均值（臂 14 维
97% 贴零、夹爪 ≈均值）——腕部视角才是决定性的：训练起点爪几乎压在盒沿
正上方，蓝盒+白色空气开关占腕部视野下半很大比例；摆远了盒子缩到视野
边缘，模型看「爪下无盒」就不出抓取动作（伸臂待机）。本脚本用于摆位
迭代：跑一次十几秒，看拼图调机器人/盒子位置，直到构图 ≈ 训练帧。

输出 /tmp/scene_check/：
  head_right.png / left_arm.png / right_arm.png     当前帧原图
  compare_left.png / compare_right.png / compare_head.png
                                                    左=当前 右=训练 并排拼图
参考帧在 docs/scene_reference/（episode0 起点±，随库入库）。

用法:
  ~/holy/run.sh ~/holy/scripts/inference/g1_scene_check.py
"""
import os
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/data/galbot/lib")
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402

REF_DIR = pathlib.Path("/home/galbot/holy/docs/scene_reference")
OUT_DIR = pathlib.Path("/tmp/scene_check")
# (SDK 相机, 当前帧文件名, 训练参考帧, 拼图文件名)
CAMS = [
    (SensorType.HEAD_RIGHT_CAMERA, "head_right.png", "head_right_train.png", "compare_head.png"),
    (SensorType.LEFT_ARM_CAMERA, "left_arm.png", "left_arm_train.png", "compare_left.png"),
    (SensorType.RIGHT_ARM_CAMERA, "right_arm.png", "right_arm_train.png", "compare_right.png"),
]


def grab(robot, cam, tag):
    """重启后相机服务可能慢：6 次×2s 重试。"""
    for k in range(6):
        d = robot.get_rgb_data(cam)
        if d and d.get("data"):
            img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        print(f"  {tag} 第{k + 1}次取图空，等 2s 重试", flush=True)
        time.sleep(2)
    return None


def side_by_side(cur, ref):
    """左=当前 右=训练，等高拼图，中间画分隔线。"""
    h = 480
    a = cv2.resize(cur, (int(cur.shape[1] * h / cur.shape[0]), h))
    b = cv2.resize(ref, (int(ref.shape[1] * h / ref.shape[0]), h))
    sep = np.full((h, 4, 3), 255, np.uint8)
    return np.hstack([a, sep, b])


def _kb(t, v, tb):
    if t is KeyboardInterrupt:
        print("\n⛔ Ctrl-C 退出", flush=True)
        r = globals().get("robot")
        if r is not None:
            try:
                r.request_shutdown(); r.wait_for_shutdown(); r.destroy()
            except Exception:
                pass
        sys.stdout.flush(); os._exit(130)
    sys.__excepthook__(t, v, tb)


sys.excepthook = _kb

print("== 场景对位检查（只取图，不加载模型）==", flush=True)
robot = GalbotRobot()
if not robot.init(set(c for c, *_ in CAMS)):
    raise SystemExit("robot.init 失败")
time.sleep(5)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for cam, cur_name, ref_name, cmp_name in CAMS:
    img = grab(robot, cam, cur_name)
    if img is None:
        print(f"❌ {cur_name} 取图失败（重启后服务未就绪？稍等重跑）", flush=True)
        continue
    cv2.imwrite(str(OUT_DIR / cur_name), img)
    ref_p = REF_DIR / ref_name
    if ref_p.exists():
        ref = cv2.imread(str(ref_p))
        cv2.imwrite(str(OUT_DIR / cmp_name), side_by_side(img, ref))
        print(f"{cur_name} ✓ → 拼图 {OUT_DIR / cmp_name}"
              f"（左=当前 / 右=训练）", flush=True)
    else:
        print(f"{cur_name} ✓ → {OUT_DIR / cur_name}"
              f"（无参考帧 {ref_p}）", flush=True)

print(f"""
对位判据（腕部两张最重要）：
  ✅ 蓝盒占视野下半较大比例、白色空气开关清晰可见，盒沿在爪正前下方
  ❌ 盒子缩在视野边缘/很远、看到的多是空桌板和背景 → 挪机器人向前
     或把盒子往桌沿挪（训练起点爪几乎压在盒沿正上方）
对齐后再跑探针看 chunk：~/holy/run.sh ~/holy/scripts/inference/run_g1_inference.py
合格信号：臂 14 维呈 -1.2~1.6 结构化数值（贴零占比 << 50%）、夹爪维 ≈0%""",
      flush=True)

robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
sys.stdout.flush(); os._exit(0)   # SDK 残留线程，干净退出
