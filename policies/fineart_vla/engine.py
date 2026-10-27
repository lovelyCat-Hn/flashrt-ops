"""fineart_vla 引擎加载入口（FlashRT 注册名 = "fineart_vla"）。

状态（2026-10-27）：注册骨架落地——System 1 flow 路径 = pi05 前端原样
继承（fineart 权重多出的两个 lm_head 不会被按名拉取的转换器读到）；
System 2 文本解码（generate_subtask）未移植。加载即用，冒烟脚本
见同目录 smoke_engine.py。
"""
from __future__ import annotations

from flash_rt.api import load_model

CONFIG = "fineart_vla"


def load(ckpt, *, num_views=3, action_dim=None,
         rtc=None, rtc_horizon=None, rtc_max_weight=None, **overrides):
    """加载 fineart_vla → flash_rt VLAModel。

    契约同 policies/README.md：只固定引擎侧参数（config 名 / 3 视角），
    权重路径与维数由调用方注入。fineart 上游训练配置 rtc_config=None，
    guided RTC 默认不开（那是 pi05 推理侧机制，微调适配后再议）。
    """
    return load_model(ckpt, framework="torch", config=CONFIG,
                      num_views=num_views, action_dim=action_dim,
                      rtc=rtc, rtc_horizon=rtc_horizon,
                      rtc_max_weight=rtc_max_weight, **overrides)
