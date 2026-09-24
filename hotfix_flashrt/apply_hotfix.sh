#!/usr/bin/env bash
# FlashRT 本地补丁热修（新机解压 bundle 后跑一次；幂等，重复跑无副作用）
#
# 背景：flashrt_bundle（2026-09-21 传 Windows）里的 FlashRT 源码停在 6406a1c2，
# 缺纯 Python 本地提交：
#   6425fe8d  pi05_rtx/api: 输出动作维可配置（G1 16 维，load_model(action_dim=N)）
#   bff59419  actions: norm_mode 语义分发（q01_q99|mean_std）+ normalize_state
#   4822d755  cuda_graph: 改走 cudaGraphInstantiateWithFlags（L4T r35.6 iGPU
#             驱动旧式 Instantiate 必段错误；CUDA graph 抓图由此恢复可用）
# 环境本体（torch wheel / .so / tokenizer / 权重）不受影响——只换这 4 个 py 文件。
# 本目录的文件 = echo 机 2026-09-24 验证过的版本（graph 修复经 bench+数值校验）。
#
# 用法: bash ~/holy/hotfix_flashrt/apply_hotfix.sh
#   （FLASHRT_DIR 环境变量可覆盖目标，默认 ~/holy/FlashRT）
set -u
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${FLASHRT_DIR:-$HOME/holy/FlashRT}"

probe() {  # 0=补丁已在, 1=缺
    grep -q "_out_action_dim" "$DST/flash_rt/frontends/torch/pi05_rtx.py" 2>/dev/null \
        && grep -q "action_dim" "$DST/flash_rt/api.py" 2>/dev/null \
        && grep -q "def normalize_state" "$DST/flash_rt/core/utils/actions.py" 2>/dev/null \
        && grep -q "cudaGraphInstantiateWithFlags" "$DST/flash_rt/core/cuda_graph.py" 2>/dev/null
}

if probe; then
    echo "✅ FlashRT 本地补丁已在（action_dim + norm_mode + graph WithFlags），无需处理"
    exit 0
fi

echo "→ 应用热修到 $DST ..."
for f in flash_rt/api.py flash_rt/frontends/torch/pi05_rtx.py flash_rt/core/utils/actions.py flash_rt/core/cuda_graph.py; do
    cp -v "$SRC/$f" "$DST/$f" || { echo "❌ 复制失败: $f"; exit 1; }
done

if probe; then
    echo "✅ 热修应用完成（4 文件，纯 Python，无需重装/重编）"
else
    echo "❌ 应用后探针仍失败——文件版本不匹配？对照 DEPLOY.md 处置表 #10"
    exit 1
fi

# 快照到本地 git（失败不致命，只是新机 FlashRT 仓库不带补丁提交记录）
if git -C "$DST" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git -C "$DST" add -A && git -C "$DST" \
        -c user.name=galbot -c user.email=galbot@echo \
        commit -q -m "hotfix: action_dim + norm_mode/normalize_state + cuda_graph WithFlags（echo 机 2026-09-24 同步）

Co-Authored-By: Claude Code <noreply@anthropic.com>" 2>/dev/null; then
        echo "✅ 已快照到 FlashRT 本地 git"
    else
        echo "  (git 快照跳过：无变化或仓库不可写——不影响使用)"
    fi
fi
