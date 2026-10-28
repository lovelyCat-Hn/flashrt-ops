"""pi05 适配器（接口留位）。

警示：现有 pi05 权重 = **G1 语义**（action 16 维 / state 23 维，G1 7-DoF 双臂 +
leg 五关节布局）。S1 无 leg 组、夹爪是独立 API——语义映射未定义，
needs_mapping=False：只能 dry-run。真要在 S1 跑 pi05 需 S1 语义微调权重。
"""
from __future__ import annotations

import os

import numpy as np

from .base import PolicyAdapter

os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")

VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")


class Pi05Adapter(PolicyAdapter):
    name = "pi05"
    config_name = "pi05"
    action_dim = 16                  # G1 双臂 14 + 双爪 2（G1 语义）
    state_dim = 23                   # G1 全身（含 leg5）
    horizon = 50
    views = VIEW_KEYS
    camera_map = {"image": "HEAD_RIGHT_CAMERA",
                  "wrist_image": "LEFT_ARM_CAMERA",
                  "wrist_image_right": "RIGHT_ARM_CAMERA"}
    needs_mapping = False
    notes = "G1 语义权重；S1 无 leg 组/夹爪接口不同——映射未定义，dry-run only。"

    def load(self, ckpt: str, task: str, calib_obs: dict | None = None) -> None:
        from flash_rt.api import load_model
        from flash_rt.core.utils.actions import normalize_state

        self.model = load_model(ckpt, framework="torch", config="pi05",
                                num_views=3, cache_frames=1, action_dim=16)
        ns = self.model._pipe.norm_stats
        self._state_n = normalize_state(np.zeros(self.state_dim, np.float32), ns)
        self.model.set_prompt(task, self._state_n)
        self._obs0 = None            # calibrate 由首次 warm 时补（需要一帧观测）

    def warm(self, obs: dict) -> None:
        if self._obs0 is None:
            self.model.pipeline.calibrate_with_real_data([obs])
            self._obs0 = obs

    def infer(self, obs: dict, state_raw: np.ndarray) -> np.ndarray:
        self.warm(obs)
        out = self.model.predict(obs, state=self._state_n)
        acts = out["actions"] if isinstance(out, dict) else out
        return np.asarray(acts, dtype=np.float32)
