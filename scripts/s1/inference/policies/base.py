"""PolicyAdapter 抽象层——通用执行器（run_s1_loop.py）只认这里的接口与 S1Targets。

隔离原则：policy 专属的一切（引擎加载方式、观测预处理、动作语义、System 2
入口）都住在本包各适配器里；执行器只做 S1 SDK 交互、循环节拍与安全栏。

S1 事实来源：~/holy/docs/s1-joint-api-quickref.md（限位权威 =
/data/galbot/config/teleop/SingoriX/real/galbot_s1-v2.0/*.toml）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── S1 关节名（quickref §2；运行时仍以 get_joint_names 实读为准） ──
LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
TORSO_JOINTS = ["torso_lift_joint1"]   # ⚠ 运行时实名（get_joint_names 实测），config 文档写 torso_joint1 不符；升降柱，位置语义=高度米非 rad
HEAD_JOINTS = ["head_joint1", "head_joint2"]

# ── S1 19 维状态布局（quickref §3，yuqiz 采集口径） ──
S1_STATE_LAYOUT = (
    [("left_arm", n) for n in LEFT_ARM_JOINTS]
    + [("left_gripper", "left_arm_gripper")]
    + [("right_arm", n) for n in RIGHT_ARM_JOINTS]
    + [("right_gripper", "right_arm_gripper")]
    + [("torso", "torso_lift_joint1")]
    + [("head", n) for n in HEAD_JOINTS]
)

# ── 限位表（rad / % / m；左右臂 j2/j4 不对称，逐条照抄勿镜像） ──
S1_LIMITS: dict[str, tuple[float, float]] = {}
for _i, _lim in enumerate([(-2.9409, 2.9409), (-1.9808, 1.7191), (-2.9234, 2.9234),
                           (-2.5918, 1.8937), (-2.9234, 2.9234), (-0.7592, 0.7592),
                           (-1.5620, 1.5620)]):
    S1_LIMITS[LEFT_ARM_JOINTS[_i]] = _lim
for _i, _lim in enumerate([(-2.9409, 2.9409), (-1.7191, 1.9808), (-2.9234, 2.9234),
                           (-1.8937, 2.5918), (-2.9234, 2.9234), (-0.7592, 0.7592),
                           (-1.5620, 1.5620)]):
    S1_LIMITS[RIGHT_ARM_JOINTS[_i]] = _lim
S1_LIMITS["torso_lift_joint1"] = (0.0, 1.0)   # 升降柱高度，单位=米（官方 example kTorsoJt=0.60 "torso target height, meter"）
S1_LIMITS["head_joint1"] = (-0.7592, 0.7592)
S1_LIMITS["head_joint2"] = (-0.5847, 0.5847)   # ⚠ 头部限位两处口径不一，以 TOML 为准待实测定案

# ── S1 夹爪（独立 API，物理宽度米）──
GRIP_WIDTH_MIN_M = 0.007
GRIP_WIDTH_MAX_M = 0.11         # 长行程；短行程 0.076（按本体端工具核对）
GRIP_FEEDBACK_HALF_M = 0.055    # 单侧行程反馈上限（yuqiz 参考）


def gripper_pct_to_width_m(pct: float) -> float:
    """夹爪百分比（0–100，状态维口径）→ 物理宽度（m）。⚠ 线性换算待实测标定。"""
    p = min(max(float(pct), 0.0), 100.0)
    return GRIP_WIDTH_MIN_M + p / 100.0 * (GRIP_WIDTH_MAX_M - GRIP_WIDTH_MIN_M)


def gripper_width_m_to_pct(width_m: float) -> float:
    w = min(max(float(width_m), GRIP_WIDTH_MIN_M), GRIP_WIDTH_MAX_M)
    return (w - GRIP_WIDTH_MIN_M) / (GRIP_WIDTH_MAX_M - GRIP_WIDTH_MIN_M) * 100.0


@dataclass
class S1Targets:
    """一条动作步的 S1 全身目标——执行器唯一认识的动作表达。

    关节 = 绝对角（rad，base 域）；夹爪 = 物理宽度（m）；torso = 升降高度
    （**米**，升降柱非旋转关节——S1 与 G1 躯干本质不同）；head 同关节（rad）。
    arms 为 None 的组不动（部分策略只覆盖部分组）。
    """
    left_arm: np.ndarray | None = None       # (7,)
    right_arm: np.ndarray | None = None      # (7,)
    left_grip_m: float | None = None
    right_grip_m: float | None = None
    torso: float | None = None               # 升降高度（m，0–1.0）
    head: np.ndarray | None = None           # (2,)
    meta: dict[str, Any] = field(default_factory=dict)


class PolicyAdapter:
    """policy 适配器基类。子类按 policy 填四个类属性 + 实现 load/infer。"""

    name: str = "?"
    config_name: str = "?"          # flash_rt.load_model(config=...)
    action_dim: int = 0             # policy 动作语义维
    state_dim: int = 0              # policy 期望的 state 语义维（与 S1 19 维未必同）
    horizon: int = 50
    views: tuple[str, ...] = ()
    camera_map: dict[str, str] = {}     # obs 键 → galbot_sdk.s1 SensorType 名
    needs_mapping: bool = False     # False = 无 S1 语义映射 → 执行器拒绝 --exec
    notes: str = ""

    def __init__(self):
        self.model = None

    # ── 生命周期 ──
    def load(self, ckpt: str, task: str, calib_obs: dict | None = None) -> None:
        """加载引擎 + set_prompt + calibrate（图捕获仪式，需一帧观测）。"""
        raise NotImplementedError

    def infer(self, obs: dict, state_raw: np.ndarray) -> np.ndarray:
        """→ (H, action_dim) 已 unnormalize 的 policy 动作块。"""
        raise NotImplementedError

    # ── 可选能力 ──
    def subtask(self, task: str, obs: dict) -> str | None:
        """System 2 hook（fineart 有，pi05 无）。默认无。"""
        return None

    def to_s1(self, action_row: np.ndarray, state_raw: np.ndarray) -> S1Targets | None:
        """policy 动作语义 → S1Targets。无映射的 policy 返回 None（dry-run only）。"""
        return None


def letterbox224(img: np.ndarray) -> np.ndarray:
    """等比缩到 224 + 黑条（G1 线实证训练帧口径；默认观测预处理）。"""
    import cv2

    h, w = img.shape[:2]
    _s = 224.0 / max(h, w)
    _nh, _nw = round(h * _s), round(w * _s)
    _cv = np.zeros((224, 224, 3), dtype=np.uint8)
    _y, _x = (224 - _nh) // 2, (224 - _nw) // 2
    _cv[_y:_y + _nh, _x:_x + _nw] = cv2.resize(img, (_nw, _nh))
    return np.ascontiguousarray(_cv)
