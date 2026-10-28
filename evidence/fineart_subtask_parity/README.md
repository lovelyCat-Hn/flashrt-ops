# fineart_vla System 2 子任务生成 parity（协议②）

日期：2026-10-28　机器：S1（Jetson Orin, L4T R35.6.0）
判据：token 门（贪心文本逐位一致）+ logits cos 窄带（≥0.99999，System 1 同款结构判据；跨构造位等不可比，见 flashrt-cross-build-bit-compare-pitfall）

## 输入条件（双侧固定）

- task = `pick up the object`；3 视角噪声图 seed 8/9/10（224×224 uint8）
- 生产路径镜像：`PolicyRunner.query(kind="next_subtask")` → prompt = `<bos>User: {task}\nAssistant:`（10 token，lang_valid=10，双侧 lang_ids 逐位全同）
- ckpt = `~/holy/models/fineart_vla_midtrained`；引擎 SubtaskDecoder 与官方 `policy.generate_text`（select_message 全默认：temperature=0 贪心、kv-cache、无 loc 抑制）

## 结论（三层）

| 口径 | token 门 | logits cos | 判定 |
|---|---|---|---|
| fp32 同装置（引擎 CPU fp32 vs 参考 CPU fp32） | **PASS**：文本逐位全等，11/11 token argmax+chosen 全同 | cos_min **0.9999912** / cos_mean 0.9999977（11 步全进窄带） | **结构 parity 达成** |
| 生产 bf16 GPU（引擎） vs CPU fp32（参考） | FAIL：step3 翻转 `basket`↔`mat`（裕度 0.56 vs 0.67） | cos_mean 0.9939 / 分叉步 0.998 | 纯精度效应（结构已由上行闭环） |
| 参考 fp32 vs 参考 bf16-autocast | PASS（同文本） | — | 参考侧文本精度鲁棒 |

前缀分级（引擎 GPU bf16 vs 参考 CPU fp32）：pixel max_abs_diff=0.0；vision_feats（塔出口）cos 1−1e-11；img_emb 0.99999；prefix_vision 0.991 / prefix_lang 0.85 / prefix_last 0.987 —— 18 层 bf16 权重路线的复利偏移，fp32 化后全链回窄带，属精度域。

## 本轮修复的两个真 bug（`flash_rt/models/fineart_vla/text_decode_rtx.py`）

1. `_preprocess_images` 缺 `[0,1]→[−1,1]`（官方 modeling_pi05.py:1441）；修复前视觉输入域差一倍平移+尺度。
2. `_SiglipLayer.forward` q/k/v 直接 view 裸 LN 输出，`q_proj/k_proj/v_proj` 成死模块（权重加载正确但 forward 不调用）；修复前塔整层 cos 0.7366、vision_feats −0.15。修复后塔逐层 cos=1.0、vision_feats 1−1e-11。

## 文件清单（*.npz >9MB 未入库，仅本机）

| 文件 | 内容 |
|---|---|
| subtask_engine.json / subtask_engine_probe.json | 引擎生产口径文本（bf16 GPU，修复后） |
| subtask_ref.json / subtask_ref_bf16.json | 参考文本（fp32 / bf16-autocast，同文本） |
| subtask_compare.json | 生产口径 token 对比（FAIL @ idx3） |
| subtask_engine_cpu_fp32.json / .npz | **结构判决**：fp32 引擎 11 步逐步 cos + 文本全等 |
| subtask_logits_{engine,engine_cpu_fp32,ref_fp32,ref_bf16}.npz | 逐 step 全词表 logits (T,257152) + token ids |
| subtask_logits_compare.json | 生产口径逐步判定（argmax/chosen/cos/margin） |
| subtask_stages_{engine,ref}.npz + subtask_stages_compare.json | 前缀分级：pixel/vision_feats/img_emb/prefix_vision/prefix_lang/prefix_last |
| tower_walk_*.npz + tower_walk_compare.json | 塔逐层对拍（**修复前**存档：emb0 1.0 → l00 0.7366） |
| siglip_layer0_bisect.json | layer0 权重/子步骤分解（全部 bit-equal / 1.0） |
| log_tower_walk_postfix.txt / log_siglip_layer_capture.txt | 修复后塔复走（l00–post_ln 全 1.0）与层内全捕获（att_flat vs out_proj 输入 = 1.0，12 位全等） |

## 复现

```bash
# 参考（lerobot_cpu；HF_HUB_OFFLINE=1 必带，本机无外网路由）
/home/galbot/miniforge3/envs/lerobot_cpu/bin/python ~/holy/policies/fineart_vla/golden_subtask_ref.py
/home/galbot/miniforge3/envs/lerobot_cpu/bin/python ~/holy/policies/fineart_vla/probe_subtask_ref.py            # fp32 logits
/home/galbot/miniforge3/envs/lerobot_cpu/bin/python ~/holy/policies/fineart_vla/probe_subtask_ref.py --autocast # bf16 logits

# 引擎（flash_pyrt311）
/home/galbot/miniforge3/envs/flash_pyrt311/bin/python ~/holy/policies/fineart_vla/golden_subtask_engine.py
/home/galbot/miniforge3/envs/flash_pyrt311/bin/python ~/holy/policies/fineart_vla/probe_subtask_engine.py
/home/galbot/miniforge3/envs/flash_pyrt311/bin/python ~/holy/policies/fineart_vla/probe_subtask_stages_engine.py

# 判定 / 结构判决（lerobot_cpu）
/home/galbot/miniforge3/envs/lerobot_cpu/bin/python ~/holy/policies/fineart_vla/golden_subtask_compare.py
/home/galbot/miniforge3/envs/lerobot_cpu/bin/python ~/holy/policies/fineart_vla/probe_subtask_compare.py [--ref subtask_logits_ref_bf16.npz]
/home/galbot/miniforge3/envs/lerobot_cpu/bin/python ~/holy/policies/fineart_vla/probe_subtask_stages_compare.py
/home/galbot/miniforge3/envs/lerobot_cpu/bin/python ~/holy/policies/fineart_vla/probe_subtask_engine_cpu_fp32.py
```

参考侧两个坑（已固化在脚本内）：`select_message` 经 `config.text_tokenizer_name` 走 hub id 加载 loc tokenizer → 必须 `HF_HUB_OFFLINE=1` + `policy.config.text_tokenizer_name = <本地 paligemma_sp224 目录>`；本地路径覆盖须在 from_pretrained 前注入 config。
