# pi05（在役）

π0.5 + G1/S1 微调权重线，FlashRT 引擎加速，四台 Orin 机同款部署路径。

## 状态

**在役**。部署/验收见 [DEPLOY.md](../../DEPLOY.md)，日常配方见 [USAGE.md](../../USAGE.md)，
基准与判读史见 [BENCHMARKS.md](../../BENCHMARKS.md)。

## 权重

- 基座：`~/holy/models/pi05_lerobot_base`（14G + openpi norm_stats + tokenizer 在 `~/.cache/flash_rt/`）
- 在役 ckpt：机器人侧 config 决定指向（G1 machine3 在役 = `pi05_g1_onlypick_deploy`，
  装配/校验三步见 USAGE §3.5；**换 ckpt 必做 action stats 量级对照**，delta/绝对角不可混装）

## 引擎参数定档

| 项 | 定档 | 依据 |
|---|---|---|
| 量化 | **bf16**（INT8 全量化判死：毁动作质量；W8A16 decoder 可选，现场 −18.8ms） | BENCHMARKS/记忆 10-06/10-07 终审 |
| graph | run_* 脚本 env 门控默认开图；回退 `PI05_NO_GRAPH=1` | galbot-g1-cuda-graph-instantiate-fix |
| norm | state 调用方归一化 + norm_mode 校验（`norm_align_check.py`） | DEPLOY §4-10 |
| cuBLAS | 启动入口必挂 `cuda_warmup.py` | DEPLOY §4-8 |

## 加载

```python
from policies.pi05.engine import load   # 见 engine.py docstring
```
