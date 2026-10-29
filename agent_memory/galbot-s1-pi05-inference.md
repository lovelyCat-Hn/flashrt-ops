---
name: galbot-s1-pi05-inference
description: S1 pi05 真权重（lerobot 导出）接入 run_s1_loop 推理线——语义链/适配器/三坑/配方，dry-run 全绿待真机 exec
metadata:
  node_type: memory
  type: project
  originSessionId: 8689eb05-8c56-416a-9e8c-9e6525e95e8c
  modified: 2026-10-29T08:09:57.629Z
---

# S1 pi05 推理线（2026-10-29 落地，接 [[galbot-fineart-vla-flashrt-port]] 的执行器底座）

**权重**：`/home/galbot/model_weights/pi05/S1_run_20261028_121132_b3f637a6_010000/pretrained_model`（同事 lerobot 管线微调，训练集=`~/datasets/clean_ready_224_1028` 同款，episodes 0-46）。lerobot 导出布局（config.json+model.safetensors+policy_pre/postprocessor）——flash_rt torch 前端直读，norm_stats 自动走 lerobot pre/post safetensors 对读兜底（`flash_rt/core/utils/norm_stats.py`，action 取 post/state 取 pre）。

**语义链（三源码交叉定案）**：
- action 16=[右臂7,右爪,左臂7,左爪]（数据集实名右臂在前）；state 27=前16与action同序+torso+head2+swerve×8
- use_relative_actions=True 排除爪 → 模型输出（unnorm 后）=**臂维 delta（相对 t0 state）+爪维绝对(0-100%)**
- 还原=lerobot `to_absolute_actions`：**absolute[k]=s_t0+delta[k] 逐行同锚广播，非 cumsum**（锚=喂 policy 的 state 前16位）；归一=QUANTILES 2(x−q01)/(q99−q01)−1 与 flash_rt q01_q99 同式
- 训练 task 字符串=**"test"**（tasks.parquet 唯一任务）→ 推理 prompt 用同串
- 训练帧口径=**4:3 letterbox 224²**（内容 224×168+上下 28px 黑条，数据集帧实测）；真机 S1 相机原生 4:3，letterbox224 后内容高 168 **精确匹配**（live dry-run 判据 ✓，适配器 load 时自动体检 `_check_frame_caliber`）
- swerve 8 维喂 0：训练里 1/3/5/7≈0、4≈307（常数→归一饱和与 0 同 bin，np.digitize 吸收）、2/6/8 狂转(spread 160-253 模型只能忽略)；torso(恒0.735)/head 喂真值

**三坑（都实咬过）**：
1. **VLAModel 包装层无 norm_stats**：load_model 返回 api.VLAModel，`norm_stats`/`calibrate_with_real_data` 住 `model.pipeline`（前端实例）——读包装层属性=静默空 dict
2. **toml ckpt 覆盖适配器 CKPT_DIR**：s1.toml 预扫填 argparse 默认，忘带 --ckpt 时加载 toml 里旧 policy 的权重（曾把 fineart 14 维 stats 喂给 pi05 护栏拦下）；现 toml 工作点已切 pi05+S1 路径+prompt="test"
3. **plan_row 块首锚定=龟速 bug**：delta-max 曾相对块首 state_raw 钳→整块 50 行钳死块首±0.05=臂速 0.05rad/块；G1 正主（run_g1_loop_traj:1830）=**链式限幅参考不重锚**（`prev+=clip(tgt−prev,±δ)`）。已修：plan_row 收 name→值 dict，执行器传演进链 chain（块界从新读数重启=重锚安全行为），回放台传每 tick 新读数字典

**状态**：dry-run 全绿（引擎加载 10-14s；infer **287-386ms**/块 50 步，exact 模式每轮 set_prompt 无重建税——state token 数恒定 prompt_len 恒定 pipeline 复用；若慢有 `FLASHRT_PI05_STATE_PROMPT_MODE=fixed` 环境钩子）。needs_mapping=True 闸②开。离线探针 `scripts/s1/probes/pi05_mapping_check.py`（stats 链+映射数学+帧口径判据）。

**待真机 exec**（用户在场手扶急停）：`PYTHONPATH=/data/galbot/lib ~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/s1/inference/run_s1_loop.py --exec --rounds 3 --rows-per-cmd 2 --frames-per-row 12`（toml 供 policy/ckpt/prompt；**R2×fr12=四角闭环静配方**，toml 默认 R1fr1=①卡卡勿直接 exec）。链路=model delta→+t0 锚→绝对目标→链钳 0.05/步→LP。爪死区 2mm、爪宽换算线性未标定（base.py 挂账）。首跑核查：模型输出 delta 量级 vs 训练 stats（max≈0.53rad）、钳位计数、枕状态 OOD（训练臂姿范围 q01-q99 已录 pi05.py 注释）。
