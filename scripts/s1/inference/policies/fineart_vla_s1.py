"""fineart_vla S1 微调版适配器（S1 语义 16/27 维，--exec 可用）。

权重：fineart_vla_rtc_subtask_relative_loss_monitor_050000（11-02 加载测试绿）。
与 ALOHA 14 维 mid-trained 基座（fineart_vla.py）**语义不同源**：
  action 16 = S1 布局（右臂7+爪 + 左臂7+爪）；state 27 = S1 数据集口径
  （右前 + torso/head + swerve8）；use_relative_actions=True 除爪 →
  臂维 delta + 爪维绝对 0-100%（与 pi05 S1 同款语义链，delta 逐行同锚）。
  相机键 FlashRT 边界仍为 image/wrist_image/wrist_image_right（11-02 实证），
  ↔ S1 头/左腕/右腕。

训练特性：RTC 训练（max_delay=15、weighted 峰 delay 7-8 步 ≈ 实测 infer
232ms——训练/推理节拍自洽）；subtask_index/task_index 特征在列（System 2
文本链训过，subtask() hook 可用）；FAST 头训过但无推理面（不移植）。
⚠ 推理侧 RTC（前缀引导/--execution-horizon）是 traj 引擎的 pi05 predict
接口——适配器执行器暂无前缀引导（挂账）；裸跑=延迟 0 消费，RTC 训练对
delay 0 亦在域内（weighted 含小延迟），先跑通再接。
"""
from __future__ import annotations

import os

import numpy as np

from .base import PolicyAdapter
# S1 27 维重排表/16 维→S1Targets 换算与 pi05 同源（同一 S1 数据集布局）——
# 单一事实源，勿复制
from .pi05 import _GRIP_DIMS, _POLICY27_FROM_RAW19, S1_TARGETS_FROM_POLICY16

# chunk 走 env（模块导入时读）——run_s1_loop.py 顶部已设；此处兜底
os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")

VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")
CKPT_DIR = ("/home/galbot/model_weights/pi05/"
            "fineart_vla_rtc_subtask_relative_loss_monitor_050000/pretrained_model")
# state 归一化 stats 来源（lerobot preprocessor，q01/q99 27 维——加载时验形）
_STATS_FILE = "policy_preprocessor_step_3_normalizer_processor.safetensors"


def state_stats_ns(ckpt) -> dict:
    """preprocessor safetensors → flash_rt normalize_state 兼容 ns（(27,) 硬闸）。

    traj 引擎 fresh_obs 走 flash_rt normalize_state(ns)——同一入口喂两族
    （引擎零改）。⚠ 归一式微差已量化（11-02）：flash_rt 分母恒 +1e-6，训练
    (lerobot) 只在 denom==0 时替换——近常维 torso/head/swerve 偏差 ≤2.2e-2
    （训练 spread 25µrad 级，模型不敏感），臂/爪维逐位精确。适配器内联
    _norm_state27 是训练精确式；两路并存，真机 fineart 出异常先 A/B 这里。
    """
    from safetensors import safe_open
    with safe_open(os.path.join(str(ckpt), _STATS_FILE),
                   framework="pt", device="cpu") as f:
        q01 = f.get_tensor("observation.state.q01").numpy().astype(np.float32)
        q99 = f.get_tensor("observation.state.q99").numpy().astype(np.float32)
    if q01.shape != (27,) or q99.shape != (27,):
        raise SystemExit(
            f"state stats q01/q99 形状 {q01.shape}/{q99.shape} ≠ (27,)——"
            "不是 S1 语义微调包（ALOHA=14 维），拒绝加载")
    return {"norm_mode": "q01_q99", "state": {"q01": q01, "q99": q99}}


class FineartVlaS1Adapter(PolicyAdapter):
    name = "fineart_vla_s1"
    config_name = "fineart_vla"
    action_dim = 16                  # S1 布局（右臂在前，爪=%）
    state_dim = 27
    horizon = 50
    views = VIEW_KEYS
    camera_map = {"image": "HEAD_RIGHT_CAMERA",
                  "wrist_image": "LEFT_ARM_CAMERA",
                  "wrist_image_right": "RIGHT_ARM_CAMERA"}
    needs_mapping = True
    CKPT_DIR = CKPT_DIR
    notes = ("S1 微调版 fineart（16/27，delta-除爪同锚）→ --exec 可用。"
             "RTC 训练（峰 delay 7-8 拍）；System 2 subtask 可用（可降频摊销）。"
             "推理侧 RTC 前缀引导未接（挂账）。")

    # ── 生命周期 ──
    def load(self, ckpt: str, task: str, calib_obs: dict | None = None) -> None:
        from flash_rt.api import load_model

        self.model = load_model(ckpt, framework="torch", config="fineart_vla",
                                num_views=3, num_steps=10, action_dim=16)
        # state 归一化 stats：preprocessor safetensors 直读（fineart 路径；
        # ⚠ 前端 pipeline.norm_stats 是 pi05 前端属性，fineart 不依赖）
        ns = state_stats_ns(ckpt)
        self._q01, self._q99 = ns["state"]["q01"], ns["state"]["q99"]
        self._task = task
        # 初始 prompt（零态占位；首轮真 state 在 infer 时重设）
        self._norm_state = self._norm_state27(np.zeros(27, np.float32))
        self.model.set_prompt(task, self._norm_state)
        if calib_obs is not None:            # rtx 系 infer 前必做（图捕获仪式；
            # 漏了 infer 崩 _graph_torch_stream，11-02 实证）
            self.model.pipeline.calibrate_with_real_data([calib_obs])

    def infer(self, obs: dict, state_raw: np.ndarray) -> np.ndarray:
        s27 = np.zeros(27, np.float32)
        for d, raw_i in _POLICY27_FROM_RAW19:
            s27[d] = float(state_raw[raw_i])
        self.model.set_prompt(self._task, self._norm_state27(s27))  # 每轮真 state
        rng = np.random.default_rng()
        noise = rng.standard_normal((self.horizon, 32)).astype(np.float32)
        out = self.model.infer(obs, noise=noise)
        acts = out["actions"] if isinstance(out, dict) else out
        return np.asarray(acts, dtype=np.float32)   # (50,16) 臂 delta+爪绝对

    def subtask(self, task: str, obs: dict) -> str | None:
        return self.model.pipeline.generate_subtask(task, obs)

    # ── S1 语义映射（闸②依据；锚加回/爪不加同 pi05）──
    def to_s1(self, action_row: np.ndarray, state_raw: np.ndarray):
        a = np.asarray(action_row, dtype=np.float64).copy()
        if np.max(np.abs(a[[i for i in range(16) if i not in _GRIP_DIMS]])) > 0.7:
            print(f"⚠ fineart_s1 臂 delta 超 0.7 rad: "
                  f"{np.array2string(a, precision=3, suppress_small=True)}")
        for d, raw_i in _POLICY27_FROM_RAW19[:16]:   # 锚加回（臂维）；爪维不加
            if d not in _GRIP_DIMS:
                a[d] += float(state_raw[raw_i])
        return S1_TARGETS_FROM_POLICY16(a)

    # ── 内部件 ──
    def _norm_state27(self, s27: np.ndarray) -> np.ndarray:
        """QUANTILES 归一（lerobot normalize_processor 同式，11-02 冒烟同款）。"""
        return (2.0 * (s27 - self._q01) / (self._q99 - self._q01) - 1.0).astype(np.float32)
