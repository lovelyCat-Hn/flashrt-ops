"""π0.5 policy 加载器（policy 层统一契约：load(**overrides) -> flash_rt VLAModel）。

只固定引擎侧参数（config="pi05"、framework="torch"、可选量化门控）；
权重路径、num_views、action_dim、RTC 等由调用方注入（闭环脚本按 config/*.toml 传）。

用法：
    import sys; sys.path.insert(0, "/home/galbot/holy")
    from policies.pi05.engine import load
    pipe = load(ckpt="~/holy/models/pi05_g1_onlypick_deploy", num_views=3)
    out = pipe.infer({...})
"""
from __future__ import annotations

import os

from flash_rt.api import load_model

CONFIG = "pi05"


def load(ckpt, *, num_views=2, action_dim=None, rtc=None, rtc_horizon=None,
         rtc_max_weight=None, w8a16=False, **overrides):
    """加载 π0.5 推理管线。

    w8a16: 启用 W8A16 decoder（现场实测 −18.8ms，cos_arm 1.0000；默认关=在役 bf16 定档。
           **INT8 全量化已判死（毁动作质量），勿开**——见 BENCHMARKS/记忆）。
    """
    if w8a16:
        os.environ.setdefault("FVK_PI05_RTX_W8A16_DECODER", "1")
    return load_model(
        ckpt,
        framework="torch",
        config=CONFIG,
        num_views=num_views,
        action_dim=action_dim,
        rtc=rtc,
        rtc_horizon=rtc_horizon,
        rtc_max_weight=rtc_max_weight,
        **overrides,
    )
