#!/usr/bin/env python
"""pi05 S1 适配器离线校验（零 GPU 零 SDK，flash_pyrt311 可跑）——真权重加载前的逻辑闸：
① stats 读取链：flash_rt lerobot fallback 对真权重 pre/post safetensors →
   state q01 (27) / actions q01 (16)，normalize_state 输出落 [-1,1]；
② 19→27 state 重排 + to_s1 delta 加锚/爪绝对化数学（含符号对号）；
③ 帧口径判据：真数据集帧（4:3 letterbox=168 → ✓）/ 合成 16:9 帧（126 → ⚠⚠）。
跑法： ~/miniforge3/envs/flash_pyrt311/bin/python scripts/s1/probes/pi05_mapping_check.py
"""
import sys
import pathlib

import numpy as np

_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "inference"))

from policies.pi05 import (Pi05Adapter, _GRIP_DIMS, CKPT_DIR,   # noqa: E402
                           S1_TARGETS_FROM_POLICY16)

ok = True


def check(tag, cond, detail=""):
    global ok
    print(f"{'✓' if cond else '✗ FAIL'} {tag} {detail}")
    ok = ok and bool(cond)


# ── ① stats 读取链（复刻 pi05_rtx._load_norm_stats 的调用形态）──
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402

stats = load_norm_stats(pi05_candidates(pathlib.Path(CKPT_DIR)),
                        checkpoint_dir=pathlib.Path(CKPT_DIR))
check("stats 非空", stats is not None)
q01 = np.asarray(stats["state"]["q01"])
q99 = np.asarray(stats["state"]["q99"])
aq99 = np.asarray(stats["actions"]["q99"])
check("state q01 形状 (27,)", q01.shape == (27,), str(q01.shape))
check("actions q99 形状 (16,)", aq99.shape == (16,), str(aq99.shape))

n0 = normalize_state(np.zeros(27, np.float32), stats)
nmid = normalize_state(q01 + 0.5 * (q99 - q01), stats)
check("normalize(0) 有限", np.all(np.isfinite(n0)),
      f"范围 [{n0.min():.1f},{n0.max():.1f}]（常数维饱和=-1 属预期）")
check("normalize(中位) ≈0 且落 [-1,1]",
      np.all(np.abs(nmid) < 0.6), f"max|v|={np.abs(nmid).max():.2f}")

# ── ② 重排 + to_s1 数学 ──
ad = Pi05Adapter()
# state_raw 19：左臂=0.1*i+1，左爪=55%，右臂=0.2*i+2，右爪=66%，torso=0.735，head=(0.01,0.02)
raw = np.array([0.1 * i + 1 for i in range(7)] + [55.0]
               + [0.2 * i + 2 for i in range(7)] + [66.0, 0.735, 0.01, 0.02],
               dtype=np.float32)
s27 = ad.build_state27(raw)
check("s27 右臂 ← raw[8:15]", np.allclose(s27[0:7], raw[8:15]))
check("s27 右爪 ← raw[15]", float(s27[7]) == 66.0)
check("s27 左臂 ← raw[0:7]", np.allclose(s27[8:15], raw[0:7]))
check("s27 左爪 ← raw[7]", float(s27[15]) == 55.0)
check("s27 torso/head 真值", np.allclose(s27[16:19], [0.735, 0.01, 0.02]))
check("s27 swerve8=0", np.allclose(s27[19:27], 0.0))

# delta 行：全 0.05（臂）、爪绝对 30/40% → 加锚后右臂 j1 = 2.0+0.05、爪 30%
delta = np.full(16, 0.05, np.float32)
delta[7], delta[15] = 30.0, 40.0
tg = ad.to_s1(delta, raw)
check("to_s1 右臂 j1 = raw[8]+δ", abs(tg.right_arm[0] - (raw[8] + 0.05)) < 1e-6,
      f"{tg.right_arm[0]:.3f}")
check("to_s1 左臂 j7 = raw[6]+δ", abs(tg.left_arm[6] - (raw[6] + 0.05)) < 1e-6,
      f"{tg.left_arm[6]:.3f}")
check("to_s1 右爪 30% → m", abs(tg.right_grip_m - (0.007 + 0.3 * 0.103)) < 1e-9,
      f"{tg.right_grip_m:.4f} m")
check("to_s1 左爪 40% → m", abs(tg.left_grip_m - (0.007 + 0.4 * 0.103)) < 1e-9)
check("to_s1 torso/head None（policy 不覆盖）",
      tg.torso is None and tg.head is None)
# 爪维必须没吃到锚（66+30=96 就错了）
check("爪维未加锚", tg.right_grip_m < 0.1)

# ── ③ 帧口径判据 ──
import cv2  # noqa: E402

ds = cv2.imread("/tmp/ds_frame_head_right.png")   # 真数据集帧（前文抽出）
if ds is not None:
    print("--- 真数据集帧（预期 ✓ 168）：")
    ad._check_frame_caliber(cv2.cvtColor(ds, cv2.COLOR_BGR2RGB))
wide = np.random.default_rng(0).integers(60, 200, (360, 640, 3), dtype=np.uint8)
from policies.base import letterbox224  # noqa: E402

print("--- 合成 16:9 帧 letterbox 后（预期 ⚠⚠ 126）：")
ad._check_frame_caliber(letterbox224(wide))

print("\n" + ("ALL PASS" if ok else "存在 FAIL——修完再上真权重"))
sys.exit(0 if ok else 1)
