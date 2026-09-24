#!/usr/bin/env python
"""state-prompt 模式 A/B 终审实验（无需真机，合成数据）。

背景（2026-09-23 破案）：state 以十进制文本拼进 prompt（format_pi05_prompt），
关节值漂→bin 数位变→prompt token 数变。state_prompt_mode 默认 "exact"=
每种长度一条 pipeline，换长即整条重建+重 autotune（~800ms）；"fixed"=
定长 200 一条 pipeline 只换 embeds。

本脚本用两组不同 state（A/B）交替推理复现/验证：
  exact: A→B 首发 793ms、切回 A 511ms（重建实锤）
  fixed: 含来回切换全部 240-253ms
用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      scripts/inference/g1_state_prompt_mode_test.py {exact|fixed}
"""
import functools
import os
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "fixed"
os.environ["FLASHRT_PI05_STATE_PROMPT_MODE"] = mode
os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
os.environ.setdefault("PI05_NO_GRAPH", "0")

import numpy as np  # noqa: E402
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

_orig_init = _fe.Pi05TorchFrontendRtx.__init__


@functools.wraps(_orig_init)
def _no_graph_init(self, *a, **kw):
    kw["use_cuda_graph"] = False
    _orig_init(self, *a, **kw)


_fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init

import flash_rt  # noqa: E402

model = flash_rt.load_model("/home/galbot/holy/models/pi05_g1_deploy",
                            config="pi05", num_views=3,
                            cache_frames=1, action_dim=16)
obs = {k: np.zeros((224, 224, 3), np.uint8)
       for k in ("image", "wrist_image", "wrist_image_right")}
PROMPT = "Left arm pick up the block. Right arm pick up the block."
st_a = np.zeros(23, np.float32)
st_a[16] = 0.3
st_b = np.zeros(23, np.float32)          # 不同关节值 → 不同 bin/数位组合
st_b[16] = 0.3
st_b[0], st_b[3], st_b[11] = -1.2, 1.8, -1.7

print(f"== mode={mode} ==", flush=True)
for tag, st in ([("A", st_a)] * 3 + [("B", st_b)] * 3 + [("A", st_a)] * 2):
    t = time.perf_counter()
    model.predict(obs, prompt=PROMPT, state=st)
    print(f"{tag} {((time.perf_counter() - t) * 1000):6.0f} ms", flush=True)
os._exit(0)   # SDK/运行时残留线程；注意 os._exit 不刷新缓冲，print 已带 flush
