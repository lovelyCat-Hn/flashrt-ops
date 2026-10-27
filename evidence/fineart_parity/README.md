# fineart_vla 引擎 parity 金标（2026-10-27）

FlashRT fineart_vla 前端 System 1 动作路径 vs 官方 lerobot 实现的数值一致性对照。
金标协议：固定合成输入双侧同源 → token 序列逐位一致（硬门）+ 动作逐维 cos 判定
（跨构造位等不可比，见记忆 flashrt-cross-build-bit-compare-pitfall）。

## 条件

- 日期：2026-10-27；机型：S1（Jetson Orin sm_87，L4T R35.6.0）
- ckpt：`~/holy/models/fineart_vla_midtrained`（ALOHA 14 维口径；
  **embedding 与 lm_head untied**）
- 引擎侧：FlashRT fineart_vla 前端（bf16、CUDA graph、num_steps=10、
  vision_pool_factor=1），flash_pyrt311 env，infer 0.32 s
- 参考侧：官方 lerobot 0.6.2 `FineARTVLAPolicy`，lerobot_cpu env，CPU eager
  两遍：fp32（309.8 s）+ CPU autocast bf16（303.3 s，OMP_NUM_THREADS=8；
  autocast 而非 .to(bf16)——sample_actions 内 noise/time 强转 fp32，
  与 bf16 权重混 dtype 必炸）
- 合成输入（双侧同公式同 seed）：raw_state = seed7 U(−2.5,2.5)×14；
  3 视角 224² 噪声图 seed 8/9/10；noise seed123 (50,32)
- task = "pick up the object"；prompt token 55，双侧逐位一致 ✅

## 文件清单

| 文件 | 内容 |
|---|---|
| engine.npz | 引擎动作 (50,14)（反归一化后）+ prompt_ids + 输入 |
| engine_meta.json | 引擎侧运行元信息 |
| reference.npz | actions_fp32 / actions_bf16 (50,14) + prompt_ids + 输入 |
| reference_meta.json | 参考侧运行元信息 |
| compare.json | 对比判定（token 门 + 三组 cos + 精度带 + 双判据结论） |
| stages_engine.npz ⚠ 22 MB 本地留存不入库 | 逐层定位引擎侧：前缀隐状态（vision 768 + lang 55）、输入端语言嵌入、10 步去噪轨迹 |
| stages_ref.npz ⚠ 同上 | 逐层定位参考侧：prefix_embs/prefix_out (1,896,2048)、x/v 轨迹 |
| stages_compare.json | 三段（A 输入嵌入/B LLM 输出/C 去噪轨迹）逐 token cos |

## 复现

```bash
# 1) 引擎侧（~1 min）
~/miniforge3/envs/flash_pyrt311/bin/python \
    ~/holy/policies/fineart_vla/golden_engine.py --dir ~/holy/evidence/fineart_parity
# 2) 参考侧 fp32+autocast bf16（CPU，~20 min；--autocast-only 只补 bf16 遍）
cd ~/lerobot && OMP_NUM_THREADS=8 ~/miniforge3/envs/lerobot_cpu/bin/python -u \
    ~/holy/policies/fineart_vla/golden_reference.py --dir ~/holy/evidence/fineart_parity
# 3) 判定（秒级；token 门 + 三组 cos + 精度带双判据）
~/miniforge3/envs/flash_pyrt311/bin/python \
    ~/holy/policies/fineart_vla/golden_compare.py --dir ~/holy/evidence/fineart_parity
```

逐层定位探针（本 parity 用于抓 untied embedding bug，工具留档）：

```bash
~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/policies/fineart_vla/probe_stages_engine.py
cd ~/lerobot && OMP_NUM_THREADS=8 ~/miniforge3/envs/lerobot_cpu/bin/python -u \
    ~/holy/policies/fineart_vla/probe_stages_ref.py
~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/policies/fineart_vla/compare_stages.py
```
