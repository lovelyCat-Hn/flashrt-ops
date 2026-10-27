# fineart_vla（规划中，未接入）

Scale AI × Hugging Face 的 FineART-VLA：π0.5 + System 2 语言头（自回归子任务文本
条件化 System 1 flow-matching 动作专家）+ 可选 FAST 动作 token。双臂操作数据训练
（40,543 轨/1,718h），与 G1/S1 双臂线契合。

## 状态

**可行性评估完成（2026-10-27），未写代码**。结论：可扩展、复用度高——
fineart 直接继承 π0.5（`FineARTVLAPytorch(PI05Pytorch)`），FlashRT 走
`_template` 移植 + pi0fast 解码先例。全文见 `agent_memory/galbot-fineart-vla-flashrt-port.md`。

## 接入前置

1. **checkpoint**（卡点）：HF 发布件，含 config.json / safetensors / norm_stats——
   拿到后先核对形状与 pi05 的一致度（视觉塔/层数/头数/维度）。
2. lerobot 源码已就位：`~/lerobot/src/lerobot/policies/fineart_vla/`（架构参考，
   注意 lerobot 是 py3.12+uv 体系；引擎环境 py3.11 不受影响）。

## FlashRT 移植落点（Stage 2 时执行）

| 文件 | 说明 |
|---|---|
| `FlashRT: flash_rt/models/fineart_vla/pipeline_rtx.py` | pi05 pipeline 克隆 + lm_head 解码段 + 文本条件拼接 |
| `FlashRT: flash_rt/frontends/torch/fineart_vla_rtx.py` + `_fineart_vla_rtx_spec.py` | 前端 + 权重 spec（pi05 基础上加 lm_head） |
| `FlashRT: flash_rt/configs/fineart_vla.yaml` + api.py 注册 | 模型注册表接入 |
| ops 仓镜像 | 上述文件同步进 `hotfix_flashrt/flashrt_fineart_vla/` 随 git 走 |

本目录 `engine.py` 待 Stage 2 移植完成后按 `policies/README.md` 契约提供
（`load(**overrides)` 同款签名）。
