#!/usr/bin/env bash
# FlashRT + pi0.5 一键验收（部署完跑这个；只读+GPU推理，不动任何系统配置）
# 用法: bash ~/holy/scripts/verify_deploy.sh
set -u
PY=~/miniforge3/envs/flash_pyrt311/bin/python
PASS=0; FAIL=0

# 权重=自备项（不随 bundle 分发，见 DEPLOY.md §1/§4-10）；就位则 ②④⑤ 真冒烟
CKPT=~/holy/models/pi05_g1_onlypick_deploy
WS=$(stat -c %s -L "$CKPT/model.safetensors" 2>/dev/null || echo 0)   # -L 穿透软链
WOK=$([ "$WS" -ge 5000000000 ] && echo yes || echo no)

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
ck "tokenizer" ls ~/.cache/flash_rt/paligemma_tokenizer.model
if [ "$WOK" = yes ]; then
    ck "在役权重就位（$((WS / 2**30))GiB，软链可解析）" test -s "$CKPT/model.safetensors"
    ck "在役 norm_stats（ckpt 自带 q01/q99）" ls "$CKPT/norm_stats.json"
else
    echo "  ℹ️ 权重未就位（自备项，不随 bundle、不判失败）——就位方式："
    echo "     把 ckpt 目录备到 $CKPT/（model.safetensors ≥5GiB 可为软链 + norm_stats.json + config.json）"
    echo "     装配/来源见 DEPLOY.md §4-10 与 USAGE.md §3.5；就位后重跑本脚本，④⑤ 自动真冒烟"
fi

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
import inspect
def _need_hotfix():
    raise RuntimeError("缺本地补丁 → bash ~/holy/hotfix_flashrt/apply_hotfix.sh")
c("本地补丁: action_dim", lambda: "ok"
  if "action_dim" in inspect.signature(flash_rt.load_model).parameters else _need_hotfix())
try:
    from flash_rt.core.utils.actions import normalize_state  # noqa
    c("本地补丁: norm_mode/normalize_state", lambda: "ok")
except ImportError:
    c("本地补丁: norm_mode/normalize_state", _need_hotfix)
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
import pathlib
s = None
for _d in (pathlib.Path("/home/galbot/holy/models/pi05_g1_onlypick_deploy"),   # 在役优先
           pathlib.Path("/home/galbot/holy/models/pi05_lerobot_base")):        # base 兜底
    try:
        s = load_norm_stats(pi05_candidates(_d), checkpoint_dir=_d)
        break
    except Exception:
        pass
if s is None:
    raise RuntimeError("无 norm_stats 可载（权重自备项未就位，见 ② 提示）")
c("norm_stats 加载", lambda: f"actions q01 {len(s['actions']['q01'])} 维")
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo "=== ④ 加载冒烟（在役 ckpt：safetensors+norm_stats+前端链路） ==="
if [ "$WOK" = yes ]; then
    PI05_NO_GRAPH=1 FVK_PI05_RTX_FORCE_INT8=1 "$PY" ~/holy/scripts/inference/load_pi05.py 2>&1 | tail -3
    [ ${PIPESTATUS[0]} -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
else
    echo "  ⏭️ 跳过（权重未就位，见 ② 提示）"
fi

echo "=== ⑤ graph 抓图探测（信息项：旧驱动 r35.5 预期崩，崩了不算失败） ==="
# graph 抓图发生在首次 predict（不是 load），必须真推理才探得到
if [ "$WOK" = yes ]; then
PI05_NO_GRAPH=0 timeout 240 "$PY" - <<'EOF' >/dev/null 2>&1
import json, pathlib, numpy as np, flash_rt
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
ck = pathlib.Path("/home/galbot/holy/models/pi05_g1_onlypick_deploy")
mf = json.loads((ck / "flashrt_deploy.json").read_text()) if (ck / "flashrt_deploy.json").exists() else {}
st = load_norm_stats(pi05_candidates(ck), checkpoint_dir=ck)
m = flash_rt.load_model(str(ck), config="pi05", action_dim=int(mf.get("action_dim", 16)))
imgs = {k: np.zeros((224, 224, 3), np.uint8)
        for k in ("image", "wrist_image", "wrist_image_right")}   # 三视图，与在役闭环同键
a = m.predict(imgs, prompt="test", state=np.zeros(len(st["state"]["q01"]), np.float32))
print("graph predict OK", a.shape)
EOF
RC=$?
[ $RC -ne 0 ] && echo "  ⚠️ graph 崩（exit $RC）——沿用 PI05_NO_GRAPH=1（r35.5 驱动属预期）" \
          || echo "  🎉 graph 能用！bench 可开 PI05_NO_GRAPH=0 拿更快延迟"
else
    echo "  ⏭️ 跳过（权重未就位，见 ② 提示）"
fi

echo ""
echo "===== 验收汇总: PASS=$PASS FAIL=$FAIL ====="
[ $FAIL -eq 0 ] && echo "🎉 部署完整，可进入基准/集成阶段" || echo "按 ❌ 项对照 DEPLOY.md 第 4 节处置表修"
exit $FAIL
