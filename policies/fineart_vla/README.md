# fineart_vla（Stage 2 骨架已落地，引擎冒烟待权重落盘）

Scale AI × Hugging Face 的 FineART-VLA：π0.5 + System 2 语言头（自回归子任务文本
条件化 System 1 flow-matching 动作专家）+ 可选 FAST 动作 token。双臂操作数据训练
（40,543 轨/1,718h），与 G1/S1 双臂线契合。

## 状态

- **可行性评估完成（2026-10-27）**：可扩展、复用度高——fineart 直接继承 π0.5，
  推理期结构 delta 实证只有两个 lm_head 张量。
- **FlashRT 注册骨架落地（2026-10-27 晚）**：`load_model(config="fineart_vla")`
  已通（api 白名单 + `_PIPELINE_MAP`/`_SM87_ALLOWED` + `fineart_vla_rtx.py`
  子类）。System 1 flow 路径 = pi05 前端原样继承；`generate_subtask`
  （System 2 文本解码）未移植（Stage 2b）。
- **checkpoint**：`lerobot/FineART-VLA-subtask-midtrained`（9.35 GB）下载中，
  目的地 `~/holy/models/fineart_vla_midtrained/`（断点续传脚本
  `fetch_checkpoint.py`，走 hf-mirror）。norm_stats 用 lerobot
  pre/postprocessor safetensors 对——`flash_rt.core.utils.norm_stats`
  原生支持该格式，零适配。
- **本轮 scope（用户定）**：引擎能力验证（能不能正常推理），不是部署；
  14 维 ALOHA 关节语义 → S1 本体映射属后续微调适配，本轮不碰。

## 冒烟

```
~/miniforge3/envs/flash_pyrt311/bin/python \
    ~/holy/policies/fineart_vla/smoke_engine.py --reps 5
```

判据：加载成功 + 输出形状合理 + 全程有限值；证据落盘 `evidence/`。
二分口径 `--as-pi05`（绕过注册链直喂权重，定位"注册问题 vs 权重问题"）。

## FlashRT 落点（实际落地情况）

| 文件 | 说明 |
|---|---|
| `FlashRT: flash_rt/frontends/torch/fineart_vla_rtx.py` | `Pi05TorchFrontendRtx` 子类：num_views=3 默认 + 懒加载 text lm_head + `generate_subtask`（NotImplementedError，Stage 2b） |
| `FlashRT: flash_rt/models/fineart_vla/__init__.py` | System 2 解码段落点（decode_text_rtx.py 待建） |
| `FlashRT: flash_rt/hardware/__init__.py` | `_PIPELINE_MAP` + `_SM87_ALLOWED` 各加一行（编辑非新文件） |
| `FlashRT: flash_rt/api.py` | config 白名单加 `fineart_vla`（编辑非新文件） |
| ops 仓镜像 | 上述 4 件同步进 `hotfix_flashrt/flash_rt/...` 随 git 走，`apply_hotfix.sh` 探针/复制清单已扩展 |

权重 spec 无需新文件：`convert_pi05_safetensors` 按名拉取，lm_head 由子类
`_ensure_text_lm_head()` 单独读。

## 参考

- 架构参考源码：`~/lerobot/src/lerobot/policies/fineart_vla/`（lerobot py3.12 体系，
  引擎环境 py3.11 不受影响）。
- System 2 解码事实（读 modeling_fineart_vla.py 实证）：`predict_action_chunk`
  与文本生成天然解耦——文本生成是独立入口（`generate_text`/`select_message`，
  默认贪心），节奏由调用方控制；固定子任务外喂 = 原生用法
  （`lang_causal_marks=None` ⇒ 纯 prompt，与 pi05 行为一致）。
  flow 路径不经过任何 lm_head。
- 判读全文：`agent_memory/galbot-fineart-vla-flashrt-port.md`。
