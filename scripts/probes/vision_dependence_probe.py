#!/usr/bin/env python
"""视觉依赖度探针（只读，零运动）：模型到底用不用相机画面定抓取点？

背景（2026-10-10）：真机 pick 表现为"往固定位置伸手，物件在哪都一样"——
怀疑视觉通路（部署喂图 or ckpt 本身）没有参与动作生成。本探针在真实机器人
上同一状态、同一 prompt 下只改图像输入，对比输出 chunk：

  A1  真实三视图            ——基准
  A2  真实三视图（重新取图）  ——确定性基线（与 A1 应几乎相同）
  B   全黑三视图            ——剜掉视觉：B≈A1 ⇒ 模型无视图像（ckpt/训练侧病）
  C   head↔left_arm 交换    ——槽位敏感度：C≈A1 ⇒ 模型分不出视图（同样指向无视视觉）

同时把 A1 实际喂进模型的 224² 三帧落盘（probe_saw_A1_*.png），人工核验
管线送图内容（是否黑帧/陈旧帧/错相机——部署侧病）。

判读表（打印在结果后）：
  (A1,A2) 差异大        → 引擎非确定，B/C 结论作废先查引擎
  (A1,B) 几乎相同       → 视觉未参与动作 ⇒ ckpt/训练侧（对照同事 lerobot 配方）
  (A1,B) 差异显著       → 视觉在参与 ⇒ 固定伸手是 OOD 场景下塌缩到边际轨迹，
                           走场景复位+数据多样性路线
  (A1,C) 几乎相同且(A1,B)显著 → 视图可互换，怀疑槽位映射训练/部署不一致

用法（SDK 会 start_controller，急停可及）：
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \\
  ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/probes/vision_dependence_probe.py
只调 get_rgb_data / get_joint_positions，不发任何运动/夹爪指令。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/galbot/lib")
sys.path.insert(0, os.path.expanduser("~/holy/scripts/inference"))

import cv2  # noqa: E402
import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402
import g1_config  # noqa: E402

try:
    _data, _p = g1_config.load()
except Exception:
    _data = {}
_run = (_data or {}).get("run", {})
CKPT = _run.get("ckpt") or os.path.expanduser("~/holy/models/pi05_g1_onlypick_deploy")
PROMPT = _run.get("prompt") or "Left arm pick up A. Right arm pick up A."
print(f"[probe] ckpt={CKPT}\n[probe] prompt={PROMPT}")

# 与 run_g1_loop.py 同源常量（manifest 权威）
import json  # noqa: E402
mf = json.load(open(os.path.join(CKPT, "flashrt_deploy.json")))
CAM_MAP = mf.get("camera_map", {"image": "HEAD_RIGHT_CAMERA",
                                "wrist_image": "LEFT_ARM_CAMERA",
                                "wrist_image_right": "RIGHT_ARM_CAMERA"})
VIEWS = int(mf.get("views", 3))
ACTION_DIM = int(mf.get("action_dim", 16))
GRIP_WMIN = mf.get("gripper", {}).get("width_min")
GRIP_WMAX = mf.get("gripper", {}).get("width_max")

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
STATE_NAMES = (RIGHT + ["right_gripper_joint1"] + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])
GRIP_IDX = (7, 15)

EVID = os.path.expanduser("~/holy/evidence/20261010_pick_aim_diff")
os.makedirs(EVID, exist_ok=True)


def grab_views(robot) -> dict:
    out = {}
    for key, cam in list(CAM_MAP.items())[:VIEWS]:
        d = robot.get_rgb_data(getattr(SensorType, cam))
        if not d or not d.get("data"):
            raise SystemExit(f"取图失败: {cam}")
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"解码失败: {cam}")
        out[key] = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
    return out


def state_now(robot, ns) -> np.ndarray:
    vals = robot.get_joint_positions([], STATE_NAMES)
    if not vals or len(vals) != len(STATE_NAMES):
        raise SystemExit(f"关节读取失败（{len(vals) if vals else 0} 维）")
    st = np.array(vals, dtype=np.float32)
    if GRIP_WMIN is not None and GRIP_WMAX is not None:
        for i in GRIP_IDX:
            st[i] = float(np.clip((st[i] - GRIP_WMIN)
                                  / (GRIP_WMAX - GRIP_WMIN + 1e-9) * 100.0,
                                  0.0, 100.0))
    return normalize_state(st, ns)


def predict(model, obs, ns, robot) -> np.ndarray:
    return np.asarray(model.predict(obs, prompt=PROMPT,
                                    state=state_now(robot, ns)))


def dump_views(obs: dict, tag: str):
    canvas = np.full((224, 224 * len(obs) + 8 * (len(obs) - 1), 3), 255, np.uint8)
    x = 0
    for k, im in obs.items():
        canvas[:, x:x + 224] = im
        cv2.putText(canvas, k, (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)
        x += 232
    p = os.path.join(EVID, f"probe_saw_{tag}.png")
    cv2.imwrite(p, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[probe] 已落盘模型实际输入({tag}) → {p}")


def cmp_chunks(a, b, la, lb):
    def group_cos(x, y, idx):
        u, v = x[:, idx].ravel(), y[:, idx].ravel()
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        return c, float(np.abs(u - v).max())
    arm = list(range(0, 7)) + list(range(8, 15))
    rows = []
    for name, idx in [("R臂", range(0, 7)), ("L臂", range(8, 15)),
                      ("爪", [7, 15]), ("全臂", arm)]:
        c, d = group_cos(a, b, list(idx))
        rows.append(f"    {name}: cos {c:+.4f}  max|Δ| {d*1000:.1f} mrad")
    print(f"  {la} vs {lb}:")
    print("\n".join(rows))
    return rows


print("== SDK 初始化（只读取图/读关节，无运动）==")
robot = GalbotRobot()
sensors = {getattr(SensorType, c) for c in list(CAM_MAP.values())[:VIEWS]}
if not robot.init(sensors):
    raise SystemExit("robot.init 失败")
time.sleep(3)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")

model = flash_rt.load_model(CKPT, config="pi05", num_views=VIEWS,
                            cache_frames=1, action_dim=ACTION_DIM)
ns = model._pipe.norm_stats
obsA1 = grab_views(robot)
dump_views(obsA1, "A1")
chunkA1 = predict(model, obsA1, ns, robot)          # 吸收惰性构建
chunkA1 = predict(model, grab_views(robot), ns, robot)
time.sleep(0.5)
chunkA2 = predict(model, grab_views(robot), ns, robot)
time.sleep(0.5)
black = {k: np.zeros_like(v) for k, v in obsA1.items()}
chunkB = predict(model, black, ns, robot)
time.sleep(0.5)
swap = dict(obsA1)
if "image" in swap and "wrist_image" in swap:
    swap["image"], swap["wrist_image"] = swap["wrist_image"], swap["image"]
chunkC = predict(model, swap, ns, robot)

print(f"\nchunk shape {chunkA1.shape}（步×维，首 25 步参与对比）")
n = min(25, chunkA1.shape[0])
cmp_chunks(chunkA1[:n], chunkA2[:n], "A1", "A2(重取图)")
cmp_chunks(chunkA1[:n], chunkB[:n], "A1", "B(全黑图)")
cmp_chunks(chunkA1[:n], chunkC[:n], "A1", "C(head↔left交换)")
print("\n判读：A1≈A2 才有效；(A1,B)几乎相同=模型无视视觉(ckpt/训练侧)；"
      "(A1,B)差异大=视觉在参与(场景/数据侧)；"
      "(A1,C)几乎相同=视图可互换(槽位映射疑点)")
