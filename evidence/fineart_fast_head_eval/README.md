# fineart_vla FAST 头评估（2026-10-28）

挂账「FAST 头（gemma_expert.lm_head [257152,1024]）评估」结论：**纯训练辅助损失，
官方推理面无 FAST 路径，引擎侧无需移植**。fineart 引擎能力验证的推理面
= flow（System 1）+ 子任务文本（System 2），两者均已验证（parity 金标 10-27 +
文本冒烟 10-27）。

## 代码证据（~/lerobot，read-only）

| 位置 | 事实 |
|---|---|
| `policies/fineart_vla/modeling_fineart_vla.py:227` | `sample_actions` = 纯 flow matching（euler_integrate），无 FAST 分支 |
| 同文件 `forward`（:781-864）+ `_fast_lin_ce`（:457） | FAST 只作为训练 CE 损失（`enable_fast_action_loss` + `fast_action_loss_weight` 门控） |
| 同文件 `FineARTVLAPolicy`（:718 起） | 类内仅 `__init__`/`forward`/`_embed_supervised_prefix`——无任何 FAST 推理方法；`select_action`/`predict_action_chunk` 继承自 PI05Policy（flow） |
| `processor/tokenizer_processor.py:339` `ActionTokenizerProcessorStep.__call__` | 推理批次无 ACTION 键时**显式跳过**（源码注释 "During inference, no action is available, skip tokenization"） |
| ckpt `policy_postprocessor.json` | 步骤仅 absolute_actions / device / unnormalizer——**无 FAST 解码步** |
| ckpt `train_config.json` | `enable_fast_action_loss=true` → 权重里两个 lm_head 的由来（训练配方产物），与推理无关 |

## 判定

- FAST 头存在 = mid-trained 训练配方（pi0-fast 风格动作 token CE）的辅助监督产物。
- 官方部署推理不加载/不使用它；引擎能力验证线对 FAST 的正确动作 = 不移植（移植
  反而引入官方不存在的推理面）。
- 挂账关闭，无遗留。
