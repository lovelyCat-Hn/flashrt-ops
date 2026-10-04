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
# 单位：角度 rad｜时长 s｜速度 rad/s（夹爪 m/s）｜力 N｜百分比 %｜频率 Hz
BUILTIN = {
    "run": {
        # 各机产物目录名不同：本机（第三台）=pi05_g1_place_deploy，echo 机=pi05_g1_ft，
        # 另一设备=pi05_g1_deploy；BUILTIN 是兜底镜像，运行时以各机 config/g1.toml 为准
        "ckpt": "/home/galbot/holy/models/pi05_g1_place_deploy",
        "prompt": "Left arm places A in the top-left corner. Right arm places A in the top-left corner.",
    },
    "warmup": {
        "speed": 0.15,        # 臂/头关节速度 rad/s（腿固定 0.2 rad/s）
        "skip_zero": False,
        "skip_leg": False,
    },
    "execute": {
        "steps": 3,           # 步（chunk 前 K 步，≤10）
        "delta_max": 0.05,    # rad/步
        "speed": 0.15,        # rad/s
    },
    "loop": {
        # 2026-10-04 工作点定案（RTC B 脚本当日 14 跑）：速度律 臂速÷原速=n÷(30×pace×div)
        "rounds": 200,            # 轮。spc=20 时代预算 52-60 轮（9/30）；spc=25
                                  # 实测落盒 34-46 轮；观察跑 60；q 停=R 落盒才停
        "steps_per_round": 25,    # 步/轮（25/0.38/1.65=1.33× 原速；chunk 整除 2 轮/块）
        "steps_per_cmd": 25,      # 步/条（合步不跨 chunk）
        "delta_max": 0.3,         # rad/步（典型步距 3-10 mrad，只兜快相位削顶）
        "speed": 1.0,             # rad/s；0.25 会削工作点快相位（9/30 实测）
        "pace": 0.38,             # s；低于水位→RTC hold 兜底（纯墙钟零垃圾）；
                                  # 0.35-0.43 墙钟相同（节奏钉在推理水位）=死 knob
        "pace_div": 1.65,         # spc=25 下 1.45-1.65 任务时间零差（div 饱和），
                                  # 取 1.65=臂最慢（1.33×）jerk 最小余量最大
        "near_div": 2,            # 近距阻尼：导程<near_gap 时改用此系数（单调收敛+
                                  # 滤放置犹豫摆幅；9/30 L 释放段真机验证）
        "near_gap": 0.06,         # 近距阻尼触发导程，rad（60 mrad）
        "cache_frames": 1,        # 1=每帧全量（恒定）。2 已判死（9/30）：state 进
                                  # prompt 每轮重置前缀计数=纯空转，勿再试
        "max_excursion": 3.0,     # rad；按数据集包络重标（合法抓取 max 2.785，
                                  # 旧 0.25 真任务 100% 误触）
        "settle": False,
        "settle_frac": 0.5,       # 比例（无量纲 0~1）
        "switch_dist": 0.06,      # rad
        "chunk_mode": "track",    # track / settle / traj（traj 已封存慎用）
        "traj_dt": 0.1,           # s/点（traj 模式）
        "catch_timeout": 8.0,     # s；回放步末追平门超时（run_g1_replay 用）
    },
    "replay": {
        # run_g1_replay 专用（与 loop 解耦）：速度按 ep0 增量分布定标——
        # 每控制步最忙关节位移 p95 0.147 rad ÷ 0.75s 自然窗口 ≈ 0.2，
        # 取 0.15：90% 步在窗口内自然完成，全程无满速冲刺（冲击∝速度）
        "speed": 0.15,
        "pace": 0.95,             # s；节拍窗口：每步位移摊满窗口单条指令，
                                  # 推理重叠在内（治衔接停走，2026-09-24）
        "catch_timeout": 8.0,     # 步末追平门超时
        "chunk_mode": "track",
        "traj_dt": 0.1,           # s/点
    },
    "gripper": {
        "enabled": True,          # --grip 的 config 形态
        "speed": 0.15,            # m/s；0.05 把 80mm 开行程拖成 1.6s（数据集是
                                  # 0.1s 快开），place 任务闭爪只在空爪无冲击风险
        "effort": 30,             # N
        "chg": 2.0,               # %（0~100% 语义）
    },
    "inference": {
        "rounds": 10,             # 轮
        "hold": 10,               # 次
        "tier": "int8_full",
        "ctrl_hz": 30.0,          # Hz（=数据集 fps）
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
