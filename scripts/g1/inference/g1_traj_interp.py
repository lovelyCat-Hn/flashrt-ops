"""G1 轨迹流插值器 —— 兼容转发壳（common 整理步1，2026-10-30）。

实现已收编 common/interp/traj_interp.py（= 原 g1/s1_traj_interp 逐行副本
合一，通用数学无机体耦合）。本壳只为旧 import 路径保活：
  scripts/g1/probes/replay_stream_bench.py、swing_attrib_probe.py、
  scripts/g1/inference/run_g1_loop*.py 等仍
  `from g1_traj_interp import create_interpolator`。
新代码一律 `from common.interp.traj_interp import create_interpolator`。
"""
import pathlib
import sys

_HOLY_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
if _HOLY_ROOT not in sys.path:
    sys.path.append(_HOLY_ROOT)   # 尾部追加：勿遮蔽脚本目录同名包（holy/policies 前科 10-30）

from common.interp.traj_interp import (  # noqa: E401,F401
    BaseInterpolator,
    CubicInterpolator,
    LinearInterpolator,
    QuinticInterpolator,
    create_interpolator,
)
