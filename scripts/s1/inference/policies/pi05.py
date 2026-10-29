"""pi05 适配器——S1 语义真权重（同事 lerobot 管线微调，clean_ready_224_1028）。

权重：/home/galbot/model_weights/pi05/S1_run_20261028_121132_b3f637a6_010000
（lerobot 导出：config.json + model.safetensors + policy_pre/postprocessor；
与 fineart_vla_midtrained 同布局——flash_rt torch 前端直读已验证的形态）。

语义链（10-29 对质定案，三处源码交叉）：
  action 16 = [right_arm_joint1-7, right_arm_gripper, left_arm_joint1-7,
               left_arm_gripper]（= 数据集 action_feature_names，S1 运行时实名）
  state 27  = 前 16 与 action 同序同位 + [torso, head1, head2, swerve×8]
              （数据集 info.json names；swerve=底盘轮编码器）
  use_relative_actions=True + exclude "gripper"
    → 模型输出（flash_rt unnormalize 后）= **臂维 delta（相对 t0 state）
       + 爪维绝对（0-100%）**
  还原语义 = lerobot absolute_actions_processor（src/lerobot/processor/
    relative_action_processor.py::to_absolute_actions）：**absolute[k] =
    s_t0 + delta[k] 逐行同锚**（state[..., :16] 广播到整个 horizon，非 cumsum；
    锚=喂给 policy 的 state 前 16 位，与训练时 delta = action - state 同一基准）。
  归一 = QUANTILES：2(x−q01)/(q99−q01)−1（normalize_processor.py:404）——
    与 flash_rt q01_q99 同式；stats 由 flash_rt 自动从 lerobot pre/post
    safetensors 对读（core/utils/norm_stats.py lerobot fallback，action 取
    postprocessor、state 取 preprocessor）。

⚠ 训练 task 字符串 = "test"（tasks.parquet 全库唯一任务）——推理 prompt 用
  同串最 in-distribution（--task test）。
⚠ 训练帧口径 = 4:3 内容 letterbox 成 224²（数据集实测内容 224×168 + 上下
  28px 黑条）——load 时对首帧实测内容高体检口径（168=匹配 / 126=16:9 源需裁）。
⚠ swerve8 维喂 0：训练里 swerve1/3/5/7≈0、swerve4≈307（常数，归一后饱和到
  与 0 同一离散 bin）、swerve2/6/8 狂转（spread 160-253，模型只能学会忽略）；
  torso/head 喂真实读数（19 维 state 里有）。
"""
from __future__ import annotations

import os

import numpy as np

from .base import PolicyAdapter, gripper_pct_to_width_m

# chunk 走 env（模块导入时读）——run_s1_loop.py 顶部已设；此处兜底
os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")

VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")
CKPT_DIR = ("/home/galbot/model_weights/pi05/"
            "S1_run_20261028_121132_b3f637a6_010000/pretrained_model")

# ── 19 维 S1 state_raw（run_s1_loop.read_s1_state，左臂在前）→ 27 维 policy
#    state（右臂在前）重排表：policy 维 ← state_raw 维；swerve 8 维无源喂 0。
_POLICY27_FROM_RAW19 = (                      # (policy 维, state_raw 维)
    [(i, 8 + i) for i in range(7)]            # 0-6   right_arm ← state_raw[8:15]
    + [(7, 15)]                               # 7     right_gripper ← [15]
    + [(8 + i, i) for i in range(7)]          # 8-14  left_arm ← state_raw[0:7]
    + [(15, 7)]                               # 15    left_gripper ← [7]
    + [(16, 16), (17, 17), (18, 18)])         # 16-18 torso/head ← 同位
_GRIP_DIMS = (7, 15)                          # policy action 爪维（绝对，0-100%）


class Pi05Adapter(PolicyAdapter):
    name = "pi05"
    config_name = "pi05"
    action_dim = 16
    state_dim = 27
    horizon = 50
    views = VIEW_KEYS
    camera_map = {"image": "HEAD_RIGHT_CAMERA",
                  "wrist_image": "LEFT_ARM_CAMERA",
                  "wrist_image_right": "RIGHT_ARM_CAMERA"}
    needs_mapping = True
    CKPT_DIR = CKPT_DIR
    notes = ("S1 语义（右臂在前 16 维 delta-除爪；state 27 右前+swerve0）"
             "→ --exec 可用。训练 task='test'、帧口径 4:3 letterbox。"
             "System 1 only（无 subtask）。")

    # ── 生命周期 ──
    def load(self, ckpt: str, task: str, calib_obs: dict | None = None) -> None:
        from flash_rt.api import load_model

        self.model = load_model(ckpt, framework="torch", config="pi05",
                                num_views=3, num_steps=10, action_dim=16,
                                cache_frames=1)
        # stats 由前端自动从 lerobot pre/post safetensors 对读——⚠ norm_stats
        # 住前端实例（VLAModel 包装层没有此属性，fineart 用 .pipeline 同理），
        # 在此验明再走
        self._stats = getattr(self.model.pipeline, "norm_stats", None) or {}
        stats = self._stats
        q01 = np.asarray(stats.get("state", {}).get("q01", []))
        if q01.shape != (27,):
            raise SystemExit(
                f"norm_stats['state'] q01 形状 {q01.shape} ≠ (27,)——"
                "lerobot stats 读取链断了，state 归一化不可信，拒绝加载")
        self._task = task
        # state 归一化（flash_rt 不内做；原始值喂 prompt=静默 bin 饱和）+
        # 初始 prompt（首轮真 state 在 infer 时重设）
        from flash_rt.core.utils.actions import normalize_state

        self._norm_state = normalize_state(np.zeros(27, np.float32),
                                           self._stats)
        self.model.set_prompt(task, self._norm_state)
        if calib_obs is not None:            # rtx 系 infer 前必做（图捕获仪式）
            self.model.pipeline.calibrate_with_real_data([calib_obs])
            self._check_frame_caliber(calib_obs.get("image"))

    def infer(self, obs: dict, state_raw: np.ndarray) -> np.ndarray:
        from flash_rt.core.utils.actions import normalize_state

        s27 = self.build_state27(state_raw)
        norm = normalize_state(s27, self._stats)
        self.model.set_prompt(self._task, norm)   # 每轮真 state（token 数恒定
        # → prompt_len 恒定 → exact 模式 pipeline 复用无重建税；若实测慢可
        #   FLASHRT_PI05_STATE_PROMPT_MODE=fixed 兜底，G1 ③层同款）
        rng = np.random.default_rng()
        noise = rng.standard_normal((self.horizon, 32)).astype(np.float32)
        out = self.model.infer(obs, noise=noise)
        acts = out["actions"] if isinstance(out, dict) else out
        return np.asarray(acts, dtype=np.float32)   # (50,16) 臂 delta+爪绝对

    # ── S1 语义映射（闸②依据）──
    def to_s1(self, action_row: np.ndarray, state_raw: np.ndarray):
        # 锚 = 喂给 policy 的 state 前 16 位（build_state27 同一重排）——
        # absolute_actions_processor 语义：逐行同锚加回，非 cumsum
        a = np.asarray(action_row, dtype=np.float64).copy()
        if np.max(np.abs(a[[i for i in range(16) if i not in _GRIP_DIMS]])) > 0.7:
            print(f"⚠ pi05 臂 delta 超 0.7 rad（训练 max≈0.53）: "
                  f"{np.array2string(a, precision=3, suppress_small=True)}")
        for d, raw_i in _POLICY27_FROM_RAW19[:16]:   # 锚加回（臂维）；爪维不加
            if d not in _GRIP_DIMS:
                a[d] += float(state_raw[raw_i])
        return S1_TARGETS_FROM_POLICY16(a)

    # ── 内部件 ──
    def build_state27(self, state_raw: np.ndarray) -> np.ndarray:
        """S1 19 维读数 → policy 27 维（右臂在前 + torso/head 真值 + swerve 0）。"""
        s = np.zeros(27, np.float32)
        for d, raw_i in _POLICY27_FROM_RAW19:
            s[d] = float(state_raw[raw_i])
        return s

    def _check_frame_caliber(self, img: np.ndarray) -> None:
        """首帧内容高体检对训练口径（4:3 letterbox=168；16:9 源=126 需裁）。
        只警告不改图——合成噪声帧（无黑条）静默跳过。"""
        if img is None or img.ndim != 3:
            return
        g = np.asarray(img).mean(axis=2)
        rows = np.where(g.mean(axis=1) > 5)[0]
        if len(rows) == 0 or rows[0] == 0 and rows[-1] == img.shape[0] - 1:
            return                                   # 无上下黑条（合成帧/全幅）→ 跳过
        h = int(rows[-1] - rows[0] + 1)
        if 160 <= h <= 176:
            print(f"✓ 帧口径匹配训练（内容高 {h}px ≈ 4:3 letterbox 168）")
        elif 118 <= h <= 134:
            print(f"⚠⚠ 帧口径不匹配！内容高 {h}px = 16:9 源直 letterbox；"
                  f"训练口径 4:3（168px）——需先中心裁 4:3 再 letterbox，"
                  f"否则视场/构图 OOD（G1 蠕动破案同款教训）")
        else:
            print(f"⚠ 帧内容高 {h}px（训练 168）——人工核对采集管线口径")


def S1_TARGETS_FROM_POLICY16(a16: np.ndarray):
    """policy 16 维绝对动作（右臂在前，爪=%）→ S1Targets（爪→物理宽度 m）。
    独立函数便于离线单测（不依赖已加载引擎）。"""
    from .base import S1Targets

    return S1Targets(
        right_arm=a16[0:7].astype(np.float32),
        right_grip_m=gripper_pct_to_width_m(a16[7]),
        left_arm=a16[8:15].astype(np.float32),
        left_grip_m=gripper_pct_to_width_m(a16[15]),
        torso=None, head=None)      # policy 不覆盖 torso/head → 不动
