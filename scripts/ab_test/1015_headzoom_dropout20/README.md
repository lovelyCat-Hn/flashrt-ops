# A/B：头图训练裁剪放大（sft_1012 dropout20 权重）

日期：2026-10-15 ｜ 状态：**待真机**

## 对比什么

| | A（现役） | B（本目录） |
|---|---|---|
| 脚本 | `scripts/inference/run_g1_loop.py` 原样 | 本目录 `run_g1_loop.py`（唯一差异=头图 crop，分任务见下） |
| 全流程编排 | `scripts/inference/run_g1_task_pickplace.py` 原样 | 本目录 `run_g1_task_pickplace.py`（差异=指同目录 B loop + B ckpt + 路径深一层 + place 相关 crop） |
| 权重 | `models/pi05_g1_onlypick_deploy` | `models/pi05_g1_sft1012_dropout20_deploy` |
| 头部图像 | letterbox224 直进 | **分任务**（10-15 同事确认）：pick 任务=letterbox224 → 裁 [126:214, 52:172] → Lanczos4 拉回 164×224（≈1.87× 放大）→ 上下补 30px 黑条；place 任务=原样 letterbox 不裁 |
| 训练数据 | 0930 job（pick_place_929/all_combin） | sft_1012（同事 clean_ready_224 4 pick 任务 + only_place；**仅 pick 任务**图像换成头裁剪版重导） |
| 训练增补 | — | right_arm_image_dropout 0.2（推理无影响） |

**依据**：sft_1012 的 pick 任务训练视频头图经 `/home/galbot/image_preprocessor.py` 的
`head_training_crop` 处理，推理 pick 相须复现同变换；place 任务训练图像无裁剪
（10-15 同事确认），place 相须回原样 letterbox——B loop 据此分任务：
`--phase2-no-head-crop`（二相切相时关）/ `--no-head-crop`（单独 place），
编排器默认自动注入（`--place-head-crop` 逃生门）。
stats 已核对与 onlypick **逐维逐位一致**（同一批 episode 重导，state/action
列未动，仅图像换版）→ 归一化侧零风险（10-08 新规对照通过）。

**教训**：13:18 全流程首跑 place 相 R 爪 25 轮不放——当时 place 相吃了头裁剪
观测（train/serve 失配），主嫌疑即此；分任务修正后待复跑验证。

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

## 全流程编排（pick→折叠横移→place，二相单模型）

```bash
# B 侧全流程（place 相导航栈醒着，pace 须显式 0.43/1.65 盖水位；
# 编排器自动给 loop 注入 --phase2-no-head-crop，place 相观测回原样 letterbox）
~/miniforge3/envs/flash_pyrt311/bin/python \
  ~/holy/scripts/ab_test/1015_headzoom_dropout20/run_g1_task_pickplace.py --exec \
  --pick-args "--phase2-pace 0.43 --phase2-pace-div 1.65"

# A 侧对照 = scripts/inference/run_g1_task_pickplace.py 同参数原样
```

编排器 B 与 A 差异仅四处（指同目录 B loop / PICK_CKPT=sft1012 / 路径深一层 /
place 相 crop 分任务注入）；`--dual-load` 对本对比无意义（旧 place 权重无头
裁剪谱系）勿用；`--place-head-crop` 可强制 place 相保留裁剪（自检用）。
⚠ place 句同样在「训练 task 原句待确认」挂账内。

## 单独 place 测试（无二相）

```bash
# B 单独 place：--no-head-crop 必带（place 训练无裁剪；当前 config 已是
# place·B 三键，命令行 ckpt/prompt 可省略但显式写全更稳）
LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
~/miniforge3/envs/flash_pyrt311/bin/python \
  ~/holy/scripts/ab_test/1015_headzoom_dropout20/run_g1_loop.py \
  --ckpt ~/holy/models/pi05_g1_sft1012_dropout20_deploy --exec \
  --no-head-crop --pace 0.43 --pace-div 1.65
```

## 判据

抓取成功率 / 爪相位时序 / R 臂落点偏差 / |Δcmd| 抖动谱；
加分项：`scripts/probes/vision_dependence_probe.py` 双权重对比
（量化视觉接地强度变化，dropout20 训练预期右臂腕图依赖下降）。

## 前置挂账

- ✓ **图像处理方式**（2026-10-15 同事确认销案）：pick=头裁剪、place=原样
  letterbox——脚本已按此分任务处理。
- ⚠ **prompt 待确认**：B 的训练 task 原句可能不是现役原句（chen 的
  clean_ready_224 任务串），须向同事要 sft_1012 的 task 字符串后再跑。

## 结论

（跑完回填）
