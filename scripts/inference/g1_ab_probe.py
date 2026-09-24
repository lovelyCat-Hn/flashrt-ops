#!/usr/bin/env python
"""A/B 判别探针（只读，不动机器人）：live 观测 vs 数据集 ep0 帧 0。

起始位（预热后）两路输入应几乎同分布。判读：
  - live chunk 右臂维 ≈ 数据集 actions[0]（近零混合符号）→ 模型输入干净，
    真机右臂下压 = 闭环演化产物（漂移/场景偏）
  - live chunk 右臂维持续单向下压、背离数据集 → 右臂输入被污染
    （对照同屏图像定位：右腕相机流内容?）
"""
import json, os, pathlib, sys, time
os.environ.setdefault("FVK_PI05_RTX_FORCE_BF16", "1")
os.environ.setdefault("PI05_NO_GRAPH", "1")
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")

import cv2
import numpy as np

sys.path.insert(0, "/home/galbot/holy/scripts/inference")

CKPT = pathlib.Path("/home/galbot/holy/models/pi05_g1_ft")
mf = json.loads((CKPT / "flashrt_deploy.json").read_text())
CAM_MAP = mf.get("camera_map", {"image": "HEAD_RIGHT_CAMERA",
                                "wrist_image": "LEFT_ARM_CAMERA",
                                "wrist_image_right": "RIGHT_ARM_CAMERA"})
PROMPT = "Left arm pick up A. Right arm pick up A."

RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
STATE_NAMES = (RIGHT + ["right_gripper_joint1"] + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)] + ["head_joint1", "head_joint2"])
GRIP_IDX = (7, 15)
WMIN, WMAX = 0.0005, 0.1200

from galbot_sdk.g1 import GalbotRobot, SensorType  # noqa: E402
import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402


def grab(robot, cam, tries=6):
    for k in range(tries):
        d = robot.get_rgb_data(getattr(SensorType, cam))
        if d and d.get("data"):
            img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        print(f"  {cam} 第{k+1}次空，2s 重试", flush=True)
        time.sleep(2)
    raise SystemExit(f"取图失败: {cam}")


robot = GalbotRobot()
sensors = {getattr(SensorType, c) for c in CAM_MAP.values()}
robot.init(sensors)
time.sleep(5)
print("✓ SDK init（未 start_controller，纯只读）", flush=True)

print("== 流活性预检（2s 两抓对比）==", flush=True)
live_raw = {}
frozen = []
for key, cam in CAM_MAP.items():
    a = grab(robot, cam)
    time.sleep(2)
    b = grab(robot, cam)
    diff = float(np.mean(cv2.absdiff(a, b)))
    print(f"  {cam}: 像素均差 {diff:.2f} {'⚠ 冻结!' if diff < 0.5 else '✓ 在动'}",
          flush=True)
    live_raw[key] = b
    if diff < 0.5:
        frozen.append(cam)
if frozen:
    print(f"⛔ 冻结流: {frozen} —— 先排查 RT 相机服务，本次探针中止", flush=True)
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    sys.stdout.flush()
    os._exit(2)

vals = robot.get_joint_positions([], STATE_NAMES)
st = np.array(vals, dtype=np.float32)
for i in GRIP_IDX:
    st[i] = float(np.clip((st[i] - WMIN) / (WMAX - WMIN + 1e-9) * 100.0, 0.0, 100.0))

robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
# SDK 用完即关（相机/关节读取不再需要；模型推理不占 SDK）
print("✓ 相机/state 采集完成，SDK 已关。加载模型（~30s）...", flush=True)
print("23 维 live state（夹爪已换算%）:")
print("  " + np.round(st, 3).tolist().__str__())

t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
ns = model._pipe.norm_stats
print(f"模型加载 {time.time()-t0:.1f}s", flush=True)

obs = {}
for key, img in live_raw.items():
    obs[key] = np.ascontiguousarray(
        cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB))
state_n = normalize_state(st, ns)

chunks = []
for r in range(3):
    t = time.perf_counter()
    c = np.asarray(model.predict(obs, prompt=PROMPT, state=state_n))
    chunks.append(c)
    print(f"predict#{r} {1000*(time.perf_counter()-t):.0f} ms", flush=True)

d = np.load("/tmp/ep0_ref.npz", allow_pickle=True)
ds_act0 = d["actions"][0]          # 数据集 ep0 帧 0 的 action（臂维=delta）
ds_st0 = d["states"][0]

# 数据集帧 0 三相机参考图：按 npz 视频寻址从 mp4 抽帧
ds_root = str(d["dataset_root"])
tmpl = str(d["video_template"])
ds_imgs = {}
for ci, (key, vk) in enumerate(zip(["image", "wrist_image", "wrist_image_right"],
                                   d["cams"])):
    path = os.path.join(ds_root, tmpl.format(
        video_key=str(vk), chunk_index=int(d["vid_chunk"][0][ci]),
        file_index=int(d["vid_file"][0][ci])))
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(d["vid_frame"][0][ci]))
    ok, fr = cap.read()
    cap.release()
    assert ok, f"数据集参考帧抽取失败: {path}"
    ds_imgs[key] = fr

out = pathlib.Path("/tmp/ab_probe"); out.mkdir(exist_ok=True)
for key, img in live_raw.items():
    ref_bgr = ds_imgs[key]
    h = 360
    a = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    b = cv2.resize(ref_bgr, (int(ref_bgr.shape[1] * h / ref_bgr.shape[0]), h))
    sep = np.full((h, 4, 3), 255, np.uint8)
    cv2.imwrite(str(out / f"cmp_{key}.png"), np.hstack([a, sep, b]))

print("\n== 数据集 ep0 帧0（参考）vs live chunk ×3 —— 臂 delta 维 ==")
print(f"{'右臂 j1-7':10s} 数据集: " + " ".join(f"{v:+.3f}" for v in ds_act0[:7]))
print(f"{'左臂 j1-7':10s} 数据集: " + " ".join(f"{v:+.3f}" for v in ds_act0[8:15]))
print(f"夹爪维(R,L) 数据集: {ds_act0[7]:.2f} / {ds_act0[15]:.2f}")
for r, c in enumerate(chunks):
    print(f"\npredict#{r}:")
    print(f"  右臂: " + " ".join(f"{v:+.3f}" for v in c[:7]))
    print(f"  左臂: " + " ".join(f"{v:+.3f}" for v in c[8:15]))
    print(f"  夹爪(R,L): {c[7]:.2f} / {c[15]:.2f}")
    cos_r = float(np.dot(c[:7], ds_act0[:7]) /
                  (np.linalg.norm(c[:7]) * np.linalg.norm(ds_act0[:7]) + 1e-9))
    cos_l = float(np.dot(c[8:15], ds_act0[8:15]) /
                  (np.linalg.norm(c[8:15]) * np.linalg.norm(ds_act0[8:15]) + 1e-9))
    print(f"  与数据集 cos：右臂 {cos_r:+.3f} | 左臂 {cos_l:+.3f}")

print("\n== state 对比（raw 数据集单位，live vs 数据集帧0，前 16 维）==")
diff = np.abs(st[:16] - ds_st0[:16])
print("  live     : " + " ".join(f"{v:+.2f}" for v in st[:16]))
print("  数据集帧0: " + " ".join(f"{v:+.2f}" for v in ds_st0[:16]))
print(f"  逐维|差| max {diff.max():.3f}（臂维应 <0.1；夹爪/腿/头窄维例外）")
print(f"\n同屏图已存 {out}/cmp_*.png（左=live 右=数据集帧0）")
sys.stdout.flush()
os._exit(0)
