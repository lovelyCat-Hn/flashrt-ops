#!/usr/bin/env python
"""G1 微调权重 → FlashRT 部署目录装配 + 预检（对齐程序）。

输入 lerobot/openpi 微调输出，产出一个 FlashRT 可直接 load_model 的目录：
  model.safetensors   —— 软链到源权重（不复制 14G）
  config.json         —— 复制
  norm_stats.json     —— openpi 形状 + "norm_mode" 语义标记（对齐核心：
                          q01_q99 = openpi 分位数；mean_std = lerobot MEAN_STD。
                          FlashRT unnormalize_actions / normalize_state 按此分发）
  flashrt_deploy.json —— 部署清单（action_dim/state_dim/视角/相机映射/夹爪标定），
                          run_g1_inference.py 读它实现零参数接入

预检项：
  ① safetensors header 校验：paligemma 骨干张量在、action_out_proj 仍 [32,1024]
    （pi0.5 原生 32 维隐空间，微调不应改动；16/7 维只是消费侧切片）
  ② config.json 的 action/state 特征维数与 norm_stats 维数三方一致
  ③ 语义模式决策打印（--mode 显式指定优先；否则 lerobot 源默认 mean_std、
    openpi 源默认 q01_q99——务必对照训练配置的 normalization_mapping 确认！）
  ④ tokenizer 在位检查

用法:
  python g1_ckpt_prep.py --src <微调输出目录> --out <部署目录> \
      [--stats <stats.json 路径>] [--mode mean_std|q01_q99] \
      [--grip-wmin 0.0 --grip-wmax 0.08]

注: 微调后 action[t] 的夹爪维是 0~100 百分比；SDK 读到的夹爪 position 是
  开口宽度（米）。--grip-wmin/--grip-wmax 标定满/零开度宽度，写入清单供
  运行时换算；不填则清单里为 null，运行时原样透传并警告。
"""
import argparse
import glob
import json
import os
import pathlib
import shutil
import struct
import sys

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True, help="微调输出目录（含 model.safetensors + config.json）")
ap.add_argument("--out", required=True, help="部署目录（将创建）")
ap.add_argument("--stats", help="norm/stats 来源 json；缺省自动探测")
ap.add_argument("--mode", choices=("mean_std", "q01_q99"),
                help="归一化语义（强烈建议显式指定，须与训练配置一致）")
ap.add_argument("--grip-wmin", type=float, help="夹爪零开度 SDK 宽度（米）")
ap.add_argument("--grip-wmax", type=float, help="夹爪满开度 SDK 宽度（米）")
ap.add_argument("--views", type=int, default=3, choices=(2, 3))
args = ap.parse_args()

src, out = pathlib.Path(args.src), pathlib.Path(args.out)
TOKENIZER = pathlib.Path("~/.cache/flash_rt/paligemma_tokenizer.model").expanduser()
fail = 0


def check(ok, msg):
    global fail
    print(f"  {'✓' if ok else '✗'} {msg}")
    if not ok:
        fail += 1
    return ok


# ── ① 源文件 ──
print("== ① 源权重 ==")
w = src / "model.safetensors"
if not check(w.exists(), f"model.safetensors: {w}"):
    sys.exit(1)
cfg_p = src / "config.json"
if not check(cfg_p.exists(), f"config.json: {cfg_p}"):
    sys.exit(1)
cfg = json.loads(cfg_p.read_text())

# safetensors header 校验（只读头部，不加载 14G）
with open(w, "rb") as f:
    (hlen,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(hlen))
tensors = {k: v["shape"] for k, v in hdr.items() if k != "__metadata__"}
n_tensors = len(tensors)
check(n_tensors > 700, f"张量数 {n_tensors}（预期 ~812）")


def tget(k):
    """键查找，兼容 lerobot 微调仓的 model. 前缀 wrap（FlashRT loader 同规则）。"""
    return tensors[k] if k in tensors else tensors.get("model." + k)


ao = tget("action_out_proj.weight")
ai = tget("action_in_proj.weight")
check(ao == [32, 1024],
      f"action_out_proj.weight {ao}（原生 32 维隐空间，应为 [32,1024]）")
check(ai == [1024, 32], f"action_in_proj.weight {ai}")
palig = sum(1 for k in tensors
            if k.startswith("paligemma_with_expert.")
            or ".paligemma_with_expert." in k)
check(palig > 500, f"paligemma 骨干张量 {palig} 个")

# ── ② config 特征维数 ──
print("== ② config 特征 ==")
of = cfg.get("output_features", {})
act_dim = of.get("action", {}).get("shape", [None])[0]
st_dim = cfg.get("input_features", {}).get(
    "observation.state", of.get("observation.state", {})).get("shape", [None])[0]
if st_dim is None:  # 某些 config 把 state 放 input_features 之外，兜底搜
    for feat in list(cfg.get("input_features", {}).values()) + list(of.values()):
        if "state" in json.dumps(feat.get("names", "")):
            st_dim = feat["shape"][0]
max_ad = cfg.get("max_action_dim")
check(act_dim is not None, f"output_features.action.shape = {act_dim}")
check(st_dim is not None, f"observation.state.shape = {st_dim}")
if max_ad is not None:
    check(act_dim <= max_ad, f"action_dim {act_dim} ≤ max_action_dim {max_ad}")

# ── ③ 定位并规整 norm_stats ──
print("== ③ 归一化统计 ==")


def openpi_shape(d):
    """lerobot dataset stats / openpi norm_stats → {actions, state} 统一形状。"""
    if "actions" in d or ("state" in d and "action" not in d):
        return d, "openpi"
    out = {}
    if "action" in d:
        out["actions"] = d["action"]
    if "observation.state" in d:
        out["state"] = d["observation.state"]
    return out, "lerobot"


stats_path = pathlib.Path(args.stats) if args.stats else None
if stats_path is None:
    cands = [src / "norm_stats.json", src / "meta" / "stats.json", src / "stats.json"]
    cands += sorted(glob.glob(str(src / "assets" / "**" / "norm_stats.json"),
                              recursive=True))
    cands = [pathlib.Path(c) for c in cands if pathlib.Path(c).exists()]
    check(bool(cands), "自动探测到 stats 文件" +
          (f": {cands[0]}" if cands else "（用 --stats 指定）"))
    stats_path = cands[0] if cands else None
if stats_path is None:
    sys.exit(1)

raw = json.loads(stats_path.read_text())
if "norm_stats" in raw and isinstance(raw["norm_stats"], dict):
    raw = raw["norm_stats"]
stats, schema = openpi_shape(raw)
if not check("actions" in stats, f"{stats_path} 含 action 统计"):
    sys.exit(1)

# 语义模式决策
mode = args.mode or raw.get("norm_mode")
auto = False
if mode is None:
    auto = True
    mode = "mean_std" if schema == "lerobot" else "q01_q99"
need = ("mean", "std") if mode == "mean_std" else ("q01", "q99")
have_block = all(k in stats["actions"] for k in need)
if not have_block:
    alt = "q01_q99" if mode == "mean_std" else "mean_std"
    print(f"  ! {stats_path} 缺 {need}，改用 {alt} 语义")
    mode = alt
print(f"  归一化语义: {mode}" + ("（自动推断——务必对照训练配置确认！)"
                                if auto else "（显式指定）"))
if schema == "lerobot" and auto:
    print("  >> lerobot 管线默认 MEAN_STD 归一化；若训练配置 normalization_mapping")
    print(">> 用了 QUANTILE，请 --mode q01_q99 重跑。语义错=动作系统性畸变。")

for name, want in (("actions", act_dim), ("state", st_dim)):
    blk = stats.get(name)
    if blk is None:
        check(False, f"{name} 统计缺失（{stats_path}）")
        continue
    key = "mean" if mode == "mean_std" else "q01"
    n = len(blk[key])
    check(n == want, f"{name} 统计维数 {n} == config {want}")

# ── ④ tokenizer ──
print("== ④ tokenizer ==")
check(TOKENIZER.exists() and TOKENIZER.stat().st_size > 4_000_000,
      f"{TOKENIZER}（{TOKENIZER.stat().st_size if TOKENIZER.exists() else 0} 字节，应 ~4.26MB）")

# ── ⑤ 写部署目录 ──
print("== ⑤ 写部署目录 ==")
out.mkdir(parents=True, exist_ok=True)
link = out / "model.safetensors"
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(w.resolve())
print(f"  软链 {link} -> {w.resolve()}")
shutil.copy2(cfg_p, out / "config.json")
stats_out = {"norm_mode": mode, **stats}
(out / "norm_stats.json").write_text(json.dumps(stats_out, indent=1))
print(f"  写   {out / 'norm_stats.json'}（norm_mode={mode}）")

manifest = {
    "action_dim": act_dim,
    "state_dim": st_dim,
    "views": args.views,
    "norm_mode": mode,
    "camera_map": {"image": "HEAD_RIGHT_CAMERA",
                   "wrist_image": "LEFT_ARM_CAMERA",
                   "wrist_image_right": "RIGHT_ARM_CAMERA"},
    "state_layout": {
        "0-6": "right_arm_joint1-7 (rad)【右臂在前，meta/info.json 权威】",
        "7": "right_arm_gripper (0~100 %，SDK 宽度米需标定换算)",
        "8-14": "left_arm_joint1-7 (rad)",
        "15": "left_arm_gripper 同上",
        "16-20": "leg_joint1-5（2026-09-22 实测与数据集逐维吻合）",
        "21-22": "head_joint1-2"},
    "gripper": {"sdk_unit": "width_m",
                "width_min": args.grip_wmin, "width_max": args.grip_wmax},
    "source_ckpt": str(src), "source_stats": str(stats_path),
}
(out / "flashrt_deploy.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
print(f"  写   {out / 'flashrt_deploy.json'}")

print()
if fail:
    print(f"✗ 预检 {fail} 项未过，先解决再部署")
    sys.exit(1)
print(f"✓ 预检通过。真机运行:\n"
      f"  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \\\n"
      f"  ~/miniforge3/envs/flash_pyrt311/bin/python \\\n"
      f"  ~/holy/scripts/inference/run_g1_inference.py --ckpt {out}")
