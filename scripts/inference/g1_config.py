"""G1 运行脚本共享配置：config/g1.toml，优先级 CLI 显式 > config > 内置默认。

约定（四个 run/warmup 脚本统一遵守）：
  - config 支持的 add_argument 一律 default=None（store_true 同样 default=None，
    未传时 argparse 给 None 而非 False），随后 apply() 按三态解析回填。
  - 安全开关（--exec 真实运动）不进配置，每次命令行显式给。
  - BUILTIN 是 config/g1.toml 的镜像兜底：文件缺失或缺项时逐键回落。
  - 布尔想显式关：脚本提供 --no-grip 这类覆盖旗标（config 写 false 也行）。
"""
import pathlib
import tomllib

DEFAULT_PATH = "/home/galbot/holy/config/g1.toml"

# 与 config/g1.toml 同源；改这里时同步改 toml（反之亦然）
BUILTIN = {
    "run": {
        "ckpt": "/home/galbot/holy/models/pi05_g1_ft",
        "prompt": "Left arm pick up A. Right arm pick up A.",
    },
    "warmup": {
        "speed": 0.15,        # 臂/头 rad/s（腿固定 0.2）
        "skip_zero": False,
        "skip_leg": False,
    },
    "execute": {
        "steps": 3,
        "delta_max": 0.05,
        "speed": 0.15,
    },
    "loop": {
        "rounds": 10,
        "steps_per_round": 3,
        "steps_per_cmd": 3,       # 合步：一条 SDK 指令跨 K 个 chunk 步
        "delta_max": 0.05,
        "speed": 0.25,            # 闭环实测最优（2026-09-23 5/5 全绿组合）
        "max_excursion": 0.25,
        "settle": False,
        "settle_frac": 0.5,
        "switch_dist": 0.06,
        "chunk_mode": "track",
        "traj_dt": 0.1,
    },
    "gripper": {
        "enabled": True,          # --grip 的 config 形态
        "speed": 0.05,            # m/s
        "effort": 30,             # N
        "chg": 2.0,               # 重发变化阈值 %
    },
    "inference": {
        "rounds": 10,
        "hold": 10,
        "tier": "int8_full",
        "ctrl_hz": 30.0,
    },
}


def load(path=DEFAULT_PATH):
    """读 toml；文件缺失返回 {}（apply 逐键回落 BUILTIN）。"""
    p = pathlib.Path(path)
    if not p.exists():
        return {}, p
    with open(p, "rb") as f:
        return tomllib.load(f), p


def apply(args, mapping, path=DEFAULT_PATH):
    """按 mapping 把 config 值回填进 args 中 CLI 未显式指定的项。

    mapping: {args 的 dest: (toml 小节, 键)}。
    返回从 config 文件实际覆盖的 (dest, 值) 列表（内置默认兜底的不算）。
    """
    cfg, p = load(path)
    from_file = []
    for dest, (sec, key) in mapping.items():
        if getattr(args, dest) is not None:
            continue                          # CLI 显式传参，最高优先级
        val = cfg.get(sec, {}).get(key, BUILTIN.get(sec, {}).get(key))
        if val is None:
            raise SystemExit(f"[config] {dest} 无 CLI 值、config 无该项、内置默认缺失")
        setattr(args, dest, val)
        if p.exists() and key in cfg.get(sec, {}):
            from_file.append((dest, val))
    if p.exists():
        cover = ", ".join(f"{d}={v!r}" if not isinstance(v, str) or len(v) < 40
                          else d for d, v in from_file)
        print(f"[config] {p}（优先级 CLI>config>内置）覆盖: {cover or '无，全内置默认'}")
    else:
        print(f"[config] {path} 不存在，全内置默认（--config 可指定别处）")
    return from_file
