"""fineart_vla 适配器（FlashRT 引擎线已验证：①②③ 全 PASS，BENCHMARKS §十二/十三/十四）。

语义警示：action 14 维 = **ALOHA 口径**（双臂各 5 关节+夹爪），state 14 维同口径。
S1 本体是 19 维（7-DoF 双臂+夹爪%+torso+head）——**语义映射未定义**（微调前），
因此 needs_mapping=False：本适配器只能 dry-run（打印动作），拒绝 --exec。
"""
from __future__ import annotations

import os

import numpy as np

from .base import PolicyAdapter

# chunk 走 env（模块导入时读）——run_s1_loop.py 顶部已设；此处兜底
os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")

VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")
CKPT_DIR = os.path.expanduser("~/holy/models/fineart_vla_midtrained")


class FineartVlaAdapter(PolicyAdapter):
    name = "fineart_vla"
    config_name = "fineart_vla"
    action_dim = 14                  # ALOHA 双臂口径（非 S1 语义）
    state_dim = 14
    horizon = 50
    views = VIEW_KEYS
    camera_map = {"image": "HEAD_RIGHT_CAMERA",
                  "wrist_image": "LEFT_ARM_CAMERA",
                  "wrist_image_right": "RIGHT_ARM_CAMERA"}
    needs_mapping = False
    CKPT_DIR = CKPT_DIR
    notes = ("ALOHA 14 维语义（mid-trained 基座）；S1 映射=微调适配（本轮不碰）。"
             "System 2 generate_subtask 可用（可降频摊销）。")

    def load(self, ckpt: str, task: str, calib_obs: dict | None = None) -> None:
        from flash_rt.api import load_model

        self.model = load_model(ckpt, framework="torch", config="fineart_vla",
                                num_views=3, num_steps=10, action_dim=14)
        # state 归一化 = golden_engine 同款（q01/q99，preprocessor safetensors 同源）
        from safetensors import safe_open

        with safe_open(os.path.join(ckpt, "policy_preprocessor_step_3_normalizer_processor.safetensors"),
                       framework="pt", device="cpu") as f:
            q01 = f.get_tensor("observation.state.q01").numpy().astype(np.float32)
            q99 = f.get_tensor("observation.state.q99").numpy().astype(np.float32)
        self._q01, self._q99 = q01, q99
        raw = np.zeros(self.state_dim, np.float32)   # ⚠ S1 19 维→ALOHA 14 维映射未定，占位零态
        norm = (2.0 * (raw - q01) / (q99 - q01) - 1.0).astype(np.float32)
        self.model.set_prompt(task, norm)
        self._norm_state = norm
        if calib_obs is not None:            # rtx 系 infer 前必做（图捕获仪式）
            self.model.pipeline.calibrate_with_real_data([calib_obs])

    def infer(self, obs: dict, state_raw: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        noise = rng.standard_normal((self.horizon, 32)).astype(np.float32)
        out = self.model.infer(obs, noise=noise)
        acts = out["actions"] if isinstance(out, dict) else out
        return np.asarray(acts, dtype=np.float32)

    def subtask(self, task: str, obs: dict) -> str | None:
        return self.model.pipeline.generate_subtask(task, obs)
