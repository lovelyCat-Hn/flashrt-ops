---
name: galbot-flashrt-env-migration-pack
description: FlashRT+torch Orin 环境整包迁移清单：同路径 tar 解压即用、目标机零编译；软链和 editable 记录随包走
metadata: 
  node_type: memory
  type: project
  originSessionId: 688dacf4-1a8a-4f2b-b926-938022b6de08
  modified: 2026-09-22T01:46:04.904Z
---

pi0.5 推理环境（FlashRT 0.2.0 + torch 2.4.0a0 自编 wheel + py3.11 conda env）打包迁移到另一台 Orin 的标准配方（2026-09-21 定稿）。核心思路：**所有编译产物已就绪，目标机零编译**；用"相同绝对路径解压"让 conda、editable 安装、cuBLAS 软链全部免修。

**目标机 4 前提**：Orin 家族 sm_87（AGX Orin/NX/Nano）；JetPack 5.1.x 且存在 `/usr/local/cuda-11.4`（软链目标+系统 .so.11 运行库）；同用户同家目录（galbot//home/galbot）；aarch64。

**① 源机打包**（原始~3G，tar.gz 约 1~1.5G；pkgs 是下载缓存可排除）：
```bash
cd ~
tar --exclude='miniforge3/pkgs' --exclude='holy/FlashRT/build*' \
    -czf ~/flashrt_bundle.tar.gz \
    miniforge3 holy/FlashRT holy/cuda_warmup.py holy/pytorch/dist
```

**② 目标机**：
```bash
cd ~ && tar -xzf flashrt_bundle.tar.gz
~/miniforge3/bin/conda init bash   # bashrc 已有则跳过
exec bash && conda activate flash_pyrt311
```
**目标机已有 Miniconda 时**：不要跑 Miniforge 的 conda init（双 initialize 块会让 which conda/env list 打架），改用绝对路径激活 `conda activate /home/galbot/miniforge3/envs/flash_pyrt311`（用原有 conda 激活即可），或干脆直接调 `~/miniforge3/envs/flash_pyrt311/bin/python`（env 自包含，运行不需要激活注入的变量）。cuBLAS 软链在 env 自己的 lib/ 里、靠 RUNPATH 生效，与哪个 conda 拥有它无关

**③ 验证**（期望与本机逐字一致）：
```bash
python ~/holy/cuda_warmup.py
python -c "
import flash_rt, torch
print('flash_rt:', flash_rt.__version__)
print('torch    :', torch.__version__, torch.cuda.get_device_capability())
from flash_rt import flash_rt_kernels
print('kernels  : ok')"
# 期望: flash_rt 0.2.0 / torch 2.4.0a0+gitee1b680 (8, 7) / kernels ok
```

**为什么软链能随包走**：conda 里的 cuBLAS 软链指向绝对路径 `/usr/local/cuda-11.4/lib64/libcublas.so.11.6.6.84`，同版本 JetPack 目标机必有；editable 安装的 .pth 记录指向 `/home/galbot/holy/FlashRT`，同路径即有效。

**例外路径**（目标机用户名/家目录不同）：tar 不适用，改干净配方——目标机重装 Miniforge → `conda create -n flash_pyrt311 python=3.11` → 装 `holy/pytorch/dist/torch-*.whl`（可移植条件见 [[galbot-pi05-env-setup]]：JP5.1.x+py3.11）→ `pip install safetensors sentencepiece` → 拷 FlashRT 源码树（含 flash_rt/ 下已编译 .so，勿重编）→ `pip install -e ".[torch]"` → 重打 cuBLAS 软链（配方同主记忆）。CUDA 11.8 工具链仅编译需要，运行不装。

**红线**：目标 JetPack 6/CUDA 12 不可用此包（无 .so.11 运行库）。

**2026-09-21 galbot-echo 机实测成功**：R35.6.0/cuda-11.4 完全同构，包解到错误子目录后 `mv` 到 ~ 即全部生效（cuBLAS 链是两跳 `libcublas.so.11 → libcublas.so.11.11.3.6 → /usr/local/cuda-11.4/...`，看链接要 `ls -lL` 穿透到底）。注意 echo 机交互 shell **无 conda 命令**（bashrc 无任何 init 块），运行一律走绝对路径 `~/miniforge3/envs/flash_pyrt311/bin/python`；验证输出与源机逐字一致。

**bundle v2 + 双轨管理（2026-09-21 定稿）**：**bundle 只装环境+资产**（miniforge3/FlashRT/cuda_warmup.py/pytorch/dist/models/.cache/flash_rt，**不含 scripts**，不压缩——safetensors 压不动）；**代码轨走 git**：`~/holy` 本身即仓库（.gitignore 圈住 FlashRT/models/pytorch，只跟踪 scripts+三份 MD+cuda_warmup.py），remote = `git@github.com:lovelyCat-Hn/flashrt-ops.git`（SSH 已通，2026-09-21 首推 master 完成）。仓库内 `agent_memory/` 是本记忆目录的 git 快照（2026-09-21 建，供新机 agent 读取获得上下文；记忆有重大更新时记得重新拷贝并 commit 刷新快照）。换机 SOP：scp bundle → tar -xf → `git clone <remote> /tmp/ops && cp -r /tmp/ops/{scripts,hotfix_flashrt}/. ~/holy/ && cp /tmp/ops/*.md ~/holy/`（fresh 机 ~/holy 无 .git，不能直接 pull）→ `bash ~/holy/hotfix_flashrt/apply_hotfix.sh` → verify_deploy.sh。首版 v2 带过 scripts，无害，git 覆盖。新增 `holy/models/pi05_lerobot_base`（权重+openpi norm_stats）、`.cache/flash_rt/paligemma_tokenizer.model`。一键验收 `bash ~/holy/scripts/verify_deploy.sh`（⑤ graph 探测必须真 predict——graph 崩在首推理不在 load；③ 段含本地补丁探针）。env 自包含含钉版 numpy==1.26.4+opencv-python-headless==4.10.0.84，目标机零 pip 零联网。踩坑处置表在 `~/holy/DEPLOY.md`（11 条）。

**⚠️ bundle 与 FlashRT 本地补丁失配（2026-09-22 发现并解决）**：v2 bundle 打包于 09-21 白天，FlashRT 源码停在 `6406a1c2`；之后两个本地纯 Python 提交不在包里——`6425fe8d`（action_dim 可配置，改 api.py+pi05_rtx.py）+ `bff59419`（norm_mode 语义+normalize_state，改 actions.py）。**环境本体（torch wheel/.so/tokenizer/权重）完全不受影响**，editable 安装只指路径不拷文件 → 新机补 3 个 py 文件即齐。机制：ops 仓库 `hotfix_flashrt/`（3 个验证过的文件 + `apply_hotfix.sh`）——**不用 git am**（bundle 内 FlashRT git 状态不完全可控），文件覆盖+grep 特征探针幂等，应用后顺手本地 git 快照。echo 机已验证：假 bundle 状态→热修→逐字节一致；verify ③ 段新探针"本地补丁: action_dim / norm_mode"自动发现缺失。**维护规矩：echo 机上 FlashRT 以后每次本地改动，必须同步刷新 `~/holy/hotfix_flashrt/` 里对应文件并 commit 推 git**（热修包以 ops 仓库为准，不跟 bundle 走；新机不重打 bundle 就永远拿得到最新补丁）。

相关：[[galbot-pi05-env-setup]]（环境本体与 cuBLAS 三层现象）、[[galbot-pi05-g1-finetune-data]]（补丁内容与 G1 三件套）、[[galbot-user-runs-install-commands]]（安装类命令由用户执行）
