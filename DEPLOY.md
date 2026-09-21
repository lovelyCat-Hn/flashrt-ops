# FlashRT + pi0.5 部署指南 v2（2026-09-21，echo 机验证全通过后定稿）

目标：在另一台 Orin 上零编译、零联网复刻 echo 机的全部验证结论。
本文件随包走，落在 `~/holy/DEPLOY.md`；一键验收脚本在 `~/holy/scripts/verify_deploy.sh`。

---

## 0. 目标机前提（硬性）

| 项 | 要求 | 检查 |
|---|---|---|
| 架构 | aarch64 | `uname -m` |
| JetPack | 5.1.x（L4T R35.x） | `cat /etc/nv_tegra_release` |
| cuBLAS 11.4 | `/usr/local/cuda-11.4/lib64/libcublas.so.11.6.6.84` 在位（软链目标） | `ls -lL` |
| 用户/家目录 | `galbot@/home/galbot`（同路径解压，否则 editable/软链全失效） | `whoami` `echo $HOME` |
| 磁盘 | ≥ 40G 空闲 | `df -h ~` |
| 红线 | **JetPack 6 / CUDA 12 不可用**（无 .so.11 运行库） | — |

## 1. 源机打包（v2 = v1 环境 + 新增资产）

**双轨管理（2026-09-21 定稿）**：环境轨走 tar（纯环境+资产，**不含 scripts**），代码轨（脚本+文档）走 git（`~/holy` 即仓库，remote 自建）。bundle 里的 scripts 视为冻结快照，上机先 `git pull` 覆盖。

```bash
cd ~
tar --exclude='miniforge3/pkgs' --exclude='holy/FlashRT/build*' \
    -cf ~/flashrt_bundle_v2.tar \
    miniforge3 holy/FlashRT holy/cuda_warmup.py holy/pytorch/dist \
    holy/models .cache/flash_rt
# 成品 ~21G，NVMe 打包 ~2 分钟；不要加 -z（safetensors 压不动，gzip 单线程白耗 1 小时；要压用 -I pigz）
```

（首版 v2 曾含 scripts，无害——git 拉取覆盖即可）

v2 相比 v1 多了三块（都是踩坑补出来的，缺一不可）：
1. `holy/models/pi05_lerobot_base/` — 权重 14G + **`assets/physical-intelligence/libero/norm_stats.json`**（openpi 官方 q01/q99 统计，HF 仓不带！）
2. `.cache/flash_rt/paligemma_tokenizer.model` — 4.26MB PaliGemMa tokenizer（HF google/paligemma 是 gated 仓，GCS 慢时用 `curl -C -` 续传）
3. `holy/scripts/` — 全部验证/基准脚本 + 本指南

## 2. 目标机部署

```bash
# ① 解压（必须在 ~ 下！解到别的目录 mv 也行，但最终位置必须是 /home/galbot/{miniforge3,holy,.cache}）
cd ~ && tar -xf flashrt_bundle_v2.tar

# ② 拉代码轨（脚本+文档以 git 为准，覆盖 bundle 里的冻结快照）
#    新机首次连 GitHub 需先配认证（二选一）：
#    a) SSH: ssh-keygen -t ed25519 -N "" → 把 ~/.ssh/id_ed25519.pub 加到 GitHub → Settings → SSH keys
#    b) HTTPS: GitHub → Settings → Developer settings → Fine-grained token（Contents: RW）
git clone git@github.com:lovelyCat-Hn/flashrt-ops.git /tmp/ops
cp -r /tmp/ops/scripts/. ~/holy/scripts/
cp /tmp/ops/*.md ~/holy/

# ③ 一键验收
bash ~/holy/scripts/verify_deploy.sh
```

（tar 里解出的 scripts 是打包时的冻结快照，第②步用 git 版覆盖；今后改动只进 git，
重打 bundle 时不要再把 scripts 打进去。echo 机已把 ~/holy 建成仓库，老机直接 `git pull` 即可。）

不需要 conda init、不需要 pip、不需要联网——env 自包含（含 numpy 1.26.4 + opencv 4.10 钉版），所有编译产物随包。
如果目标机已有 miniconda/anaconda：**不要跑 miniforge 的 conda init**（双 init 打架），用绝对路径即可。

## 3. 日常使用姿势

```bash
# FlashRT/pi0.5 一律绝对路径调 python（本系交互 shell 无 conda 命令）
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD     # 清 bashrc 的 /data/galbot/lib 泄漏（跑纯 FlashRT 时）
~/miniforge3/envs/flash_pyrt311/bin/python ~/holy/scripts/load_pi05_int8.py
```

**要同时用 GalbotSDK 时（同进程已验证可行）**：不要 unset，改带上 SDK 路径
```bash
LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
~/miniforge3/envs/flash_pyrt311/bin/python 脚本.py
```
（该目录无 CUDA 库不劫持 torch；副作用是旧 libcurl 毒化 HTTPS——SDK+推理进程不需要网络，可接受）

## 4. 已知问题与处置表（echo 机实踩）

| # | 问题 | 现象 | 处置 |
|---|---|---|---|
| 1 | 解压位置错 | editable .pth 失效，`No module named flash_rt` | 必须落 `~/holy`、`~/miniforge3`；错了 `mv` 到位即可 |
| 2 | 无 conda 命令 | `conda: command not found` | 用 `~/miniforge3/envs/flash_pyrt311/bin/python`；或 `source ~/miniconda3/etc/profile.d/conda.sh`（若存在 miniconda） |
| 3 | 环境变量泄漏 | bashrc 125/132 行把 `/data/galbot/lib` 注入 LD/PYTHONPATH | 纯 FlashRT 先 unset；用 SDK 时显式带上（见上） |
| 4 | norm_stats 缺失 | `FileNotFoundError: norm_stats not found` | HF 仓不带，必须 openpi GCS 版（包里已带）；勿用 lerobot 数据集 stats（min/max 假充 q01/q99 有 1~5% 漂移） |
| 5 | tokenizer 缺失 | `paligemma_tokenizer.model not found` / `could not parse ModelProto` | 放 `~/.cache/flash_rt/`；下载 4.26MB 完整（截断会 parse 失败），慢用 `curl -C -` 续传 |
| 6 | CUDA graph 段错误 | 首次 predict 100% 段错误，gdb 定位 `cudaGraphInstantiate` → tegra 驱动内部 | **驱动层缺陷**（r35.5），与代码无关。绕法 `use_cuda_graph=False`（脚本已内置 PI05_NO_GRAPH=1 monkeypatch）；新驱动版本可重测（verify 脚本会探） |
| 7 | numpy 被升级 | 装任何带 numpy 依赖的包后 torch/flash_rt 报 ABI 错 | **红线：钉 `numpy==1.26.4`**；opencv 用 `==4.10.0.84`（5.x 要 numpy≥2） |
| 8 | cuBLAS 阵发坏窗口 | 首调用 ALLOC_FAILED / 偶发失败 | `~/holy/cuda_warmup.py` 暖场兜底，启动入口必加 |
| 9 | INT8 无需校准 | — | 权重 scale 加载时按行静态算，激活 scale 运行时动态；FP8 那套校准不适用于 Orin |

## 5. echo 机基准（对照新机复测）

| 场景（eager，无 graph） | 延迟 | 精度 |
|---|---|---|
| 2 视角 BF16 / 编码器INT8 / 全INT8（零图） | 231 / 184 / 161 ms | cos 0.9999+ |
| 3 视角 全INT8（零图） | 190 ms | — |
| 3 视角 全INT8（真机服务并发） | 234 ms | — |
| 3 视角 BF16（真机服务并发） | 304 ms | — |
| 真机相机图 A/B（同噪声同图） | — | cos 0.954，对照混沌底 0.25 → INT8 无损 |

定档：`FVK_PI05_RTX_FORCE_INT8=1`（全 INT8）。INT8 开关：`FVK_PI05_RTX_INT8_ENCODER_ONLY=1`（保守档）/ `FVK_PI05_RTX_INT8_VISION=1` **禁用**（cos 0.991→0.282）。

## 6. 新机待验证清单（echo 机未覆盖）

- [ ] graph 抓图（驱动版本不同可能不崩；`PI05_NO_GRAPH=0 bash` 重跑 bench）
- [ ] 相机视角映射：HEAD_LEFT→image / LEFT_ARM→wrist_image / RIGHT_ARM→wrist_image_right（看 `/tmp/cam_samples.png` 确认视场）
- [ ] 动作空间映射：pi05 输出 (10,7) 是 LIBERO 臂空间，与 G1 臂零位/量纲/夹爪语义**不是恒等映射**——先开环观察再闭环
- [ ] SDK 运行时版本（echo 机是 1.8.1）
