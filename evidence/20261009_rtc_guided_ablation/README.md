# 20261009 RTC 引导（guided）质量+延迟消融

FlashRT 引擎推理侧 guided RTC（lerobot policies/rtc 移植，FlashRT 本地提交
79581a3a）的首次质量/延迟探针。复现命令见文末；原始 stdout 全文在
`summary.txt`，逐样本动作块在 `sample_*.npy`。

## 条件

- 日期：2026-10-09；机型：本机（第三台 echo，Orin sm_87，JetPack R35.6.4）
- ckpt：`~/holy/models/pi05_g1_onlypick_deploy`（10-08 only_pick 部署目录）
- 引擎：bf16（FVK_PI05_RTX_FORCE_BF16=1 语义档）、cache_frames=1 全量帧、
  无 CUDA graph、FP8 不可用自动关；chunk 50；FLASHRT_PI05_STATE_PROMPT_MODE=fixed
- 模型：单 rtc=True 实例（horizon 10，max_w 10），off=rtc_prefix=None（enable=0
  短路）；同构造内对照——构造间 autotune 重选 GEMM 算法有构造噪声
  （未归一化空间 max≈0.35，见 [[flashrt-cross-build-bit-compare-pitfall]]），
  位级断言只在同构造内成立
- 样本：N=12，种子 20261009+i；obs_A 随机图×3+state；obs_B = 图 1px 平移 +
  state 小扰动 +0.01 rad 臂维合成 BASE 漂移（前缀重锚公式覆盖）；prompt =
  部署原句 "Left arm pick up A. Right arm pick up A."
- 噪声对齐：torch.manual_seed 后 predict；B 块 off/guided 同种子
- 口径注脚：合成观测（非数据集帧、非实机相机），绝对 cos/幅度只代表该合成
  分布；接缝跳变指标的off/guided 对比在同一分布内成立

## 文件清单

- `summary.txt` —— 运行 stdout 全文（含逐样本表与汇总、延迟）
- `sample_i_A.npy / sample_i_B_off.npy / sample_i_B_on.npy`（i=0..11）——
  未归一化动作块 (50,16)，数据空间

## 指标定义

- 引导区移动门槛：行 <10 逐位有变化（逐位相同=引导未生效，硬门槛）
- cos 行<10（臂维 0-6+8-14）：guided vs off 头段方向差
- 接缝跳变（主指标）：换块模拟——A 消费 25 步后换 B，|B[0]−A[24]| 臂维
  max，数据空间；guided 的 B 以 A[25:] 重锚为前缀
- 尾段传播：行 ≥10 max |Δ|（无直接修正，attention 传播，报告不门槛）
- 夹爪哨兵：夹爪维(7/15) max |Δ|（0.8 线）

## 复现

```bash
~/miniforge3/envs/flash_pyrt311/bin/python \
    ~/holy/scripts/eval/rtc_ablation.py 12 \
    --out ~/holy/evidence/20261009_rtc_guided_ablation \
    2>&1 | tee ~/holy/evidence/20261009_rtc_guided_ablation/summary.txt
```
