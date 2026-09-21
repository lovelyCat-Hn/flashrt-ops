#!/usr/bin/env bash
# FlashRT + pi0.5 一键验收（部署完跑这个；只读+GPU推理，不动任何系统配置）
# 用法: bash ~/holy/scripts/verify_deploy.sh
set -u
PY=~/miniforge3/envs/flash_pyrt311/bin/python
PASS=0; FAIL=0

ck() {  # ck <名称> <命令...>
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then echo "  ✅ $name"; PASS=$((PASS+1));
    else echo "  ❌ $name"; FAIL=$((FAIL+1)); fi
}

echo "=== ① 目标机前提 ==="
[ "$(uname -m)" = "aarch64" ] && echo "  ✅ aarch64" || echo "  ❌ 架构非 aarch64"
grep -q "R35" /etc/nv_tegra_release 2>/dev/null && echo "  ✅ L4T R35.x" \
    || echo "  ❌ 非 JetPack 5.x（R35），此包不可用！"
ck "cuda-11.4 cuBLAS 真身" ls -lL /usr/local/cuda-11.4/lib64/libcublas.so.11.6.6.84
[ "$HOME" = "/home/galbot" ] && [ "$(whoami)" = "galbot" ] && echo "  ✅ galbot@/home/galbot" \
    || echo "  ⚠️ 用户/家目录不同——同路径配方失效，需走干净配方（见 DEPLOY.md/迁移记忆）"
FREE=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 40 ] && echo "  ✅ 磁盘 ${FREE}G 足够" || echo "  ❌ 磁盘仅 ${FREE}G（需 ≥40G）"

echo "=== ② 部署位置 ==="
ck "miniforge3 env" test -x "$PY"
ck "FlashRT editable 目录" test -d ~/holy/FlashRT/flash_rt
ck "编译产物 fa2 .so" ls ~/holy/FlashRT/flash_rt/flash_rt_fa2*.so
ck "norm_stats（openpi q01/q99）" ls ~/holy/models/pi05_lerobot_base/assets/physical-intelligence/libero/norm_stats.json
ck "tokenizer" ls ~/.cache/flash_rt/paligemma_tokenizer.model
ck "权重字节数（14467165872）" test "$(stat -c %s ~/holy/models/pi05_lerobot_base/model.safetensors 2>/dev/null)" = 14467165872

echo "=== ③ 环境链 ==="
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
"$PY" - <<'EOF'
import sys
ok = True
def c(name, fn):
    global ok
    try:
        r = fn(); print(f"  ✅ {name}: {r}")
    except Exception as e:
        print(f"  ❌ {name}: {e}"); ok = False
import numpy as np
c("numpy==1.26.4", lambda: np.__version__ + ("" if np.__version__=="1.26.4" else "  ⚠️版本漂移!"))
import torch
c("torch sm_87", lambda: f"{torch.__version__} {torch.cuda.get_device_capability()}")
import flash_rt
c("flash_rt", lambda: flash_rt.__version__)
from flash_rt import flash_rt_kernels
c("kernels 可导入", lambda: "ok")
import cv2
c("cv2", lambda: cv2.__version__)
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
import pathlib
s = load_norm_stats(pi05_candidates(pathlib.Path("/home/galbot/holy/models/pi05_lerobot_base")),
                    checkpoint_dir=pathlib.Path("/home/galbot/holy/models/pi05_lerobot_base"))
c("norm_stats 加载", lambda: f"actions q01 {len(s['actions']['q01'])} 维")
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo "=== ④ 加载+推理（2视角全INT8，期望 (10,7)、~0.2s 级） ==="
PI05_NO_GRAPH=1 FVK_PI05_RTX_FORCE_INT8=1 "$PY" ~/holy/scripts/load_pi05.py 2>&1 | tail -3
[ ${PIPESTATUS[0]} -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo "=== ⑤ graph 抓图探测（信息项：r35.5 驱动预期崩，崩了不算失败） ==="
# graph 抓图发生在首次 predict（不是 load），必须真推理才探得到
PI05_NO_GRAPH=0 timeout 240 "$PY" - <<'EOF' >/dev/null 2>&1
import numpy as np, flash_rt
m = flash_rt.load_model("/home/galbot/holy/models/pi05_lerobot_base", config="pi05")
imgs = {"image": np.zeros((224,224,3), np.uint8),
        "wrist_image": np.zeros((224,224,3), np.uint8)}
a = m.predict(imgs, prompt="test", state=np.zeros(8, np.float32))
print("graph predict OK", a.shape)
EOF
RC=$?
[ $RC -ne 0 ] && echo "  ⚠️ graph 仍崩（exit $RC）——沿用 PI05_NO_GRAPH=1，与新机驱动无关的结论可更新" \
          || echo "  🎉 新机 graph 能用！bench 可开 PI05_NO_GRAPH=0 拿更快延迟"

echo ""
echo "===== 验收汇总: PASS=$PASS FAIL=$FAIL ====="
[ $FAIL -eq 0 ] && echo "🎉 部署完整，可进入基准/集成阶段" || echo "按 ❌ 项对照 DEPLOY.md 第 4 节处置表修"
exit $FAIL
