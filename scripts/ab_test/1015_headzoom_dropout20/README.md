# A/B：头图训练裁剪放大（sft_1012 dropout20 权重）

日期：2026-10-15 ｜ 状态：**待真机**

## 对比什么

| | A（现役） | B（本目录） |
|---|---|---|
| 脚本 | `scripts/inference/run_g1_loop.py` 原样 | 本目录 `run_g1_loop.py`（唯一差异=头图 crop） |
| 权重 | `models/pi05_g1_onlypick_deploy` | `models/pi05_g1_sft1012_dropout20_deploy` |
| 头部图像 | letterbox224 直进 | letterbox224 → 裁 [126:214, 52:172] → Lanczos4 拉回 164×224（≈1.87× 放大）→ 上下补 30px 黑条 |
| 训练数据 | 0930 job（pick_place_929/all_combin） | sft_1012（同事 clean_ready_224 4 pick 任务 + only_place；图像换成头裁剪版重导） |
| 训练增补 | — | right_arm_image_dropout 0.2（推理无影响） |

**依据**：新权重的训练视频头图经 `/home/galbot/image_preprocessor.py` 的
`head_training_crop` 处理；推理须复现同变换，否则 train/serve 图像分布失配。
stats 已核对与 onlypick **逐维逐位一致**（同一批 episode 重导，state/action
列未动，仅图像换版）→ 归一化侧零风险（10-08 新规对照通过）。

## 命令（v3 工作点，两臂只差脚本与 ckpt）

```bash
# A
LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
~/miniforge3/envs/flash_pyrt311/bin/python \
  ~/holy/scripts/inference/run_g1_loop.py \
  --ckpt ~/holy/models/pi05_g1_onlypick_deploy --exec --nav-suspend \
  --grip-state-cmd

# B
LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
~/miniforge3/envs/flash_pyrt311/bin/python \
  ~/holy/scripts/ab_test/1015_headzoom_dropout20/run_g1_loop.py \
  --ckpt ~/holy/models/pi05_g1_sft1012_dropout20_deploy --exec --nav-suspend \
  --grip-state-cmd
```

同场景同起始位姿（warmup）交替跑；`--no-head-crop` 可关 crop 自检。

## 判据

抓取成功率 / 爪相位时序 / R 臂落点偏差 / |Δcmd| 抖动谱；
加分项：`scripts/probes/vision_dependence_probe.py` 双权重对比
（量化视觉接地强度变化，dropout20 训练预期右臂腕图依赖下降）。

## 前置挂账

- ⚠ **prompt 待确认**：B 的训练 task 原句可能不是现役原句（chen 的
  clean_ready_224 任务串），须向同事要 sft_1012 的 task 字符串后再跑。

## 结论

（跑完回填）
