---
name: galbot-pi05-cache-frames-dead
description: cache_frames=2 时序 K/V 复用对 G1 闭环判死：state 进 prompt 每轮重置前缀计数=永远全量，bench 3.8× 加速对本管线不可达；knob 保留默认 1
metadata:
  node_type: memory
  type: project
  originSessionId: 55e42dec-9ad0-4468-b717-690667d2e886
  modified: 2026-09-30T09:25:24.688Z
---

**cache_frames=2 判死（2026-09-30，闭环 A/B + 结构判读双证据）**：FlashRT 的 `cache_frames=N`（每 N 帧一次全量、中间帧仅解码器跑新噪声、视觉前缀复用旧 K/V）在本机 Orin 文档数 124/39ms→12.2Hz cos 0.991，但闭环 A/B 30 轮**全程静默无效**（每轮 387-409ms 与全量无异）。根因是结构性的：G1 的 state 以十进制文本拼进 prompt（关节值每帧都变）→ 闭环每轮 `set_prompt` → `_frame_count=0` 重置（pi05_rtx.py:1607）→ 18 层全注意力下任何 state 变化都使整条前缀 KV 失效，**每轮都全量，复用从不发生**。重置本身是正确性所需：绕过它=模型拿陈旧本体状态规划，比陈旧视觉更糟。

**bench 的 3.8× 为何不可达**：droid 基准 prompt 恒定（set_prompt 一次），前缀 KV 永远新鲜。我们的管线 prompt 每轮变 → 判死与调参无关，勿再试。knob 保留（config/BUILTIN/CLI --cache-frames，默认 1 无损）。若将来要省推理时间，正道是 W8A16-decoder（[[galbot-pi05-env-setup]] 唯一无损候选）或接受 pace 窗吞推理的现状（闭环瓶颈在执行不在推理，见 [[galbot-g1-loop-pace-tuning]]）。另 bench 换坑实录：chunk env 名是 `FLASH_RT_PI05_ACTION_CHUNK_SIZE`（默认 10）；--num-views 3 需 obs 补 `wrist_image_right`；热机 throttle 会把 p50 虚增到 659ms（冷机 343-359 才是真数）。
