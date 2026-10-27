---
name: galbot-fineart-vla-flashrt-port
description: FineART-VLA（π0.5+语言头）接入 FlashRT 的可行性评估（2026-10-27）：可扩展、复用度高；源码在 ~/lerobot/policies/fineart_vla，引擎侧走 _template 移植
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-27T06:13:13.001Z
---

**FineART-VLA 接入 FlashRT 可行性评估（2026-10-27，只读调研，未动机器/未写代码）**。用户意向：在 FlashRT 上加载第二个 policy（fineart_vla，Scale AI×HF 出品，源码在 `~/lerobot/src/lerobot/policies/fineart_vla/`，README 指向 docs/source/policy_fineart_vla_README.md）。

**ops 仓已立 policy 层目录（2026-10-27 用户拍板，lerobot 风格）**：`~/holy/policies/<name>/`，契约 = 每目录 `engine.py`（`load(**overrides) -> VLAModel`，只固定引擎侧参数，权重/视角/维数由调用方注入）+ `README.md`（状态/权重/定档）。已建：`pi05/`（在役，engine.py 带 w8a16 门控、INT8 判死注释）+ `fineart_vla/`（规划位，README 含 FlashRT 移植落点表）。FlashRT 侧新模型代码仍落 FlashRT 路径、ops 仓经 `hotfix_flashrt/` 镜像随 git（环境轨/代码轨分工不变）。

**核心结论：可扩展，复用度极高**。三条依据：
1. **fineart 直接继承 π0.5**：`FineARTVLAPytorch(PI05Pytorch)` / `FineARTVLAConfig(PI05Config)` / `FineARTVLAPolicy(PI05Policy)`，同 SigLIP 视觉塔 + Gemma VLM + flow-matching 动作专家骨架；`use_native_pretrained_loader=True`。架构差异（推理期）：**System 2 语言头**（paligemma.lm_head 自回归解码子任务文本 → 条件化 System 1 动作专家）+ 可选 FAST 动作 token；knowledge_insulation / flow_num_repeats=5 / flex-attention 是**训练期**技巧，推理前向等价。
2. **FlashRT 本就是多模型引擎**：`api.py load_model(config=...)` 注册了 pi05/pi0/pi0fast/groot/groot_n17/motus/ltx25/cosmos3_*/wan22 等；`frontends/torch/_template/` 是官方新模型骨架（vision+LLM+diffusion decoder 形状），配 `docs/adding_new_model.md`、`plugin_model_template.md`、`model_runtime_api.md`；自回归解码先例 = `pi0fast.py`（decode CUDA graph 已解决）；多阶段先例 = groot。
3. **lerobot 侧反过来引用 FlashRT**：fineart_vla 里有 `_FlashRTNorms`/`use_flashrt_adarms`，拉的是 Hub `kernels` 包的 `flashrt-adarms-train`/`flashrt-flex-attention-train`——那是**训练加速内核**（GPU 机器用），与本机 Orin 部署 runtime（`~/holy/FlashRT`）是两回事，勿混淆。

**移植工作量（估）**：① `models/fineart_vla/pipeline_rtx.py` = pi05 pipeline 克隆 + lm_head 解码段 + 文本条件拼接；② `frontends/torch/fineart_vla_rtx.py` + `_fineart_vla_rtx_spec.py`（权重 spec：在 pi05 基础上加 lm_head）；③ api.py/configs 注册 `fineart_vla`；④ 精度回归（eager 参考 vs 引擎 cos）+ INT8/W8A16 套件扩展 + graph 适配。pi05 的 tokenizer（.cache/flash_rt 已有）直接复用；FAST tokenizer 仅在启用 FAST 解码路径时需要。

**Stage 0 卡点已解（2026-10-27 晚）**：权重已发布——**`lerobot/FineART-VLA-subtask-midtrained`**（Scale AI×HF 官方，未 gated，2026-10-23 更新，共 **9.35 GB**，model.safetensors 单文件）。README 关键参数：底座 **gemma_2b / 动作专家 gemma_300m**（= pi05 同款）；基座 = `lerobot/pi05_base` 原生 LeRobot 转换；**3 视角**（cam_high + 双腕，360×640→224²）；**state/action = 14 维、绝对角目标**（双臂口径，与 G1 双臂 14 维契合！）；**chunk 50 步 / flow 10 步推理**（= 我们 chunk 50 同款）；归一化 = **Quantiles**（stats 随仓，pre/postprocessor safetensors 齐全）；bf16 训练（权重含 FP32 张量）；knowledge_insulation=on；FAST `skip_tokens=128, max_action_tokens=384`（BPE tokenizer 随仓）。文件全：config.json/model.safetensors/pre+postprocessor/train_config.json/action_tokenizer。**网络事实：本机直连 huggingface.co 不通（Network unreachable），`hf-mirror.com` 可直连（API+raw 均验证过）；github 可直连**（lerobot 即直连克隆）。下载走 **`policies/fineart_vla/fetch_checkpoint.py`**（纯 stdlib+断点续传，2026-10-27 已启用，后台日志 ~/holy/logs/fetch_fineart.log；flash_pyrt311 无 huggingface_hub，引擎环境勿装包——numpy 红线）。另有作者实验 ckpt（pepijn223/rebot-fineart-pi052-*）与数据集 `lerobot/FineART`。paper: arxiv 2609.36416。

**Stage 0 实测（2026-10-27 晚，从小文件+权重头读出，未等大文件下完）**：
- **config.json**：`type=fineart_vla`；state [14] / action [14]；`max_state_dim=32` / `max_action_dim=32`（=pi05 的 32 维 padding 口径，action_in_proj [1024,32] 实证）；3 视角 360×640→224²；`chunk_size=50, n_action_steps=50, num_inference_steps=10`；归一化 STATE/ACTION=QUANTILES；`use_relative_actions=False`（绝对角，exclude gripper）；`rtc_config=None`（fineart 未配 RTC，我们的 guided RTC 是 pi05 推理侧机制，移植初期不开）；`tokenizer_max_length=128`；`unfreeze_lm_head=true`；FAST on（skip 128/max 384）。
- **权重头（813 张量，键名 = LeRobot pi05 布局）**：paligemma 语言模型 18 层 hidden 2048（GQA 8q/1kv，MLP 16384）+ SigLIP 视觉塔 27 层 1152（patch 14×14、256 pos）+ projector 1152→2048 + gemma_expert 18 层 hidden 1024（MLP 4096，**带 adaRMS dense 投影 [3072,1024]**——FlashRT pi05 已实现同款）+ action_in/out_proj 32 维——**与 FlashRT pi05_rtx 权重 spec 逐项同形**。
- **唯一新增张量 = 两个 lm_head**：`paligemma.lm_head` [257152,2048]（System 2 子任务文本解码）+ `gemma_expert.lm_head` [257152,1024]（FAST 动作 token 头）——引擎移植的结构性 delta 闭环实证。
- **⚠ 关节语义（用户 2026-10-27 明确提醒：关节参照系是 S1 本体）**：fineart 的 14 维 = **ALOHA 口径**（每臂 5 关节 shoulder_pan/lift/elbow_flex/wrist_angle/wrist_rotate + gripper + gripper_finger），**不是**我们 G1（7-DoF 臂+爪百分比）也不是 S1（19 维：7-DoF A21 双臂14+双爪2+torso1+head2）。fineart 是 mid-trained 基座（官方定位 "initialize adaptation to new tasks and embodiments"）——**正路 = 引擎移植 → S1 自采数据微调（yuqiz s1_collect 管线就是为这个准备的）→ 部署**；部署前必须完成 S1 关节对齐（S1_MAINTENANCE 待办）。

**待确认/风险**：① ~~权重公开可得~~已解；② ~~System 2 解码频率~~已解（2026-10-27 晚读 modeling_fineart_vla.py 实证）：**代码层面天然解耦**——`predict_action_chunk` 只吃 batch 语言 token，文本生成是独立入口 `generate_text`/`select_message`（贪心默认、KV cache 可开），**节奏完全由调用方控制**（可降频摊销）；**固定子任务外喂是原生用法**：`lang_causal_marks=None` ⇒ marks 全 False = 语言段纯 prompt = pi05 行为逐位一致，验证协议 ① 零额外机制；marks 非 None 时子任务 token 段按训练期 `_mark_target_span_causal` 布局置 causal（③ e2e 才需要）。train_config.json 另证：底座 `lerobot/pi052_base`、`enable_fast_action_loss=true`（两 lm_head 都在训）、flow_num_repeats=1、`denoise_step` 用 suffix K/V crop（`past_key_values.crop(prefix_len)`）+expert eager attention——FlashRT pi05 同款机制；③ 关节语义映射 = fineart ALOHA 14 维 ↔ S1 本体 19 维，属**训练侧适配**（微调时对齐），引擎侧只是 32 维 padding；④ lerobot py3.12+uv vs 引擎 py3.11（引擎不受影响，eager 金标需另配环境）。建议路径：~~Stage0~~已完成 → Stage1 eager 金标（GPU 机）→ Stage2 _template bf16 移植+parity → Stage3 量化+Orin bench（入 BENCHMARKS.md）→ Stage4 S1 数据微调+闭环接入。

**⚠ 目标重定位（2026-10-27 晚，用户明确）**：本轮下载权重**只为测试推理引擎能不能正常推理**（引擎能力验证），不涉部署/本体适配——ALOHA 14 维语义、S1 关节映射、微调整条线全部先不碰。验证协议（三步解耦）：① **flow 路径 parity**：喂**固定子任务文本**跳过 System 2 解码，引擎 vs eager 的 action chunk 逐维 cos ≥0.999（位级断言只在同构造内有效，见 flashrt-cross-build-bit-compare-pitfall）；② System 2 解码 parity（贪心/采样口径与 eager 对齐）；③ e2e + 延迟三口径（空载/SDK 在场/循环）。eager 参考环境：lerobot py3.12+uv，**Orin CPU eager 可跑小规模金标**（gemma_2b bf16 单次前向分钟级，够 parity 用）；装 env 的命令按惯例列清单给用户跑（galbot-user-runs-install-commands）。下载完成后的第一动作 = 读 train_config.json + usage guide 确认 System 2 解码推理频率与可否外喂子任务（决定 ① 的实现方式）。
