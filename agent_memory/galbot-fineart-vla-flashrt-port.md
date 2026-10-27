---
name: galbot-fineart-vla-flashrt-port
description: FineART-VLA（π0.5+语言头）接入 FlashRT 的可行性评估（2026-10-27）：可扩展、复用度高；源码在 ~/lerobot/policies/fineart_vla，引擎侧走 _template 移植
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-27T05:49:21.398Z
---

**FineART-VLA 接入 FlashRT 可行性评估（2026-10-27，只读调研，未动机器/未写代码）**。用户意向：在 FlashRT 上加载第二个 policy（fineart_vla，Scale AI×HF 出品，源码在 `~/lerobot/src/lerobot/policies/fineart_vla/`，README 指向 docs/source/policy_fineart_vla_README.md）。

**ops 仓已立 policy 层目录（2026-10-27 用户拍板，lerobot 风格）**：`~/holy/policies/<name>/`，契约 = 每目录 `engine.py`（`load(**overrides) -> VLAModel`，只固定引擎侧参数，权重/视角/维数由调用方注入）+ `README.md`（状态/权重/定档）。已建：`pi05/`（在役，engine.py 带 w8a16 门控、INT8 判死注释）+ `fineart_vla/`（规划位，README 含 FlashRT 移植落点表）。FlashRT 侧新模型代码仍落 FlashRT 路径、ops 仓经 `hotfix_flashrt/` 镜像随 git（环境轨/代码轨分工不变）。

**核心结论：可扩展，复用度极高**。三条依据：
1. **fineart 直接继承 π0.5**：`FineARTVLAPytorch(PI05Pytorch)` / `FineARTVLAConfig(PI05Config)` / `FineARTVLAPolicy(PI05Policy)`，同 SigLIP 视觉塔 + Gemma VLM + flow-matching 动作专家骨架；`use_native_pretrained_loader=True`。架构差异（推理期）：**System 2 语言头**（paligemma.lm_head 自回归解码子任务文本 → 条件化 System 1 动作专家）+ 可选 FAST 动作 token；knowledge_insulation / flow_num_repeats=5 / flex-attention 是**训练期**技巧，推理前向等价。
2. **FlashRT 本就是多模型引擎**：`api.py load_model(config=...)` 注册了 pi05/pi0/pi0fast/groot/groot_n17/motus/ltx25/cosmos3_*/wan22 等；`frontends/torch/_template/` 是官方新模型骨架（vision+LLM+diffusion decoder 形状），配 `docs/adding_new_model.md`、`plugin_model_template.md`、`model_runtime_api.md`；自回归解码先例 = `pi0fast.py`（decode CUDA graph 已解决）；多阶段先例 = groot。
3. **lerobot 侧反过来引用 FlashRT**：fineart_vla 里有 `_FlashRTNorms`/`use_flashrt_adarms`，拉的是 Hub `kernels` 包的 `flashrt-adarms-train`/`flashrt-flex-attention-train`——那是**训练加速内核**（GPU 机器用），与本机 Orin 部署 runtime（`~/holy/FlashRT`）是两回事，勿混淆。

**移植工作量（估）**：① `models/fineart_vla/pipeline_rtx.py` = pi05 pipeline 克隆 + lm_head 解码段 + 文本条件拼接；② `frontends/torch/fineart_vla_rtx.py` + `_fineart_vla_rtx_spec.py`（权重 spec：在 pi05 基础上加 lm_head）；③ api.py/configs 注册 `fineart_vla`；④ 精度回归（eager 参考 vs 引擎 cos）+ INT8/W8A16 套件扩展 + graph 适配。pi05 的 tokenizer（.cache/flash_rt 已有）直接复用；FAST tokenizer 仅在启用 FAST 解码路径时需要。

**待确认/风险**：① 权重是否公开可得（HF 发布，含 config/norm_stats）；② System 2 解码是否每 chunk 必跑（延迟成本——子任务文本变化慢可降频摊销）或可关；③ fineart 训练集是双臂（bimanual 40,543 ep/1,718h/533,913 子任务标注），与我们 G1/S1 双臂线契合，但 state/action 维要从 checkpoint config 实读；④ lerobot 仓是 py3.12+uv，本机引擎环境是 py3.11——FlashRT 原生实现不受影响，eager 参考跑需另配环境。建议路径：Stage0 拿 checkpoint 读 config 确认形状 → Stage1 eager 金标 → Stage2 _template bf16 移植+parity → Stage3 量化+Orin bench（入 BENCHMARKS.md）→ Stage4 接进闭环脚本（config 开关切 policy）。
