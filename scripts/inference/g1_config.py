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
        # 各机产物目录名不同：本机（第三台）当前=pick 任务 pi05_g1_pick_deploy
        # （2026-10-05 装配；place=pi05_g1_place_deploy，切回见 config [run] 注释），
        # echo 机=pi05_g1_ft，另一设备=pi05_g1_deploy；BUILTIN 是兜底镜像，
        # 运行时以各机 config/g1.toml 为准
        "ckpt": "/home/galbot/holy/models/pi05_g1_pick_deploy",
        "prompt": "Left arm pick up A. Right arm pick up A.",
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
        # 2026-10-07 工作点 v3 定案（前置 --nav-suspend；10-08 键名正规化，
        # 旧名见 docs/lerobot-alignment.md）：速度律 臂速÷原速=n÷(30×pace×div)
        "rounds": 200,            # 轮。spc=20 时代预算 52-60 轮（9/30）；spc=25
                                  # 实测落盒 34-46 轮；v3 落盒 ~27 轮（t+10.0s）；
                                  # 观察跑 60；q 停=R 落盒才停
        "n_action_steps": 25,     # 步/轮（旧名 steps_per_round；25/0.35/1.5=1.59×
                                  # 原速；chunk 整除 2 轮/块）
        "steps_per_command": 25,  # 步/条（旧名 steps_per_cmd；合步不跨 chunk）
        "delta_max": 0.3,         # rad/步（典型步距 3-10 mrad，只兜快相位削顶）
        "speed": 1.0,             # rad/s；0.25 会削工作点快相位（9/30 实测）
        "pace": 0.35,             # s；nav-suspend 下水位≈0.34-0.35（推理 329+取图
                                  # +开销），hold 窗=0（两跑 76 轮）。0.34 以下勿入
                                  # =贴水崖边，偶发抖动即回流 hold 窗
        "pace_div": 1.5,          # v2 时代 div 饱和三档零差是"轮时间钉在水位"下的
                                  # 结论；v3 pace 活了 div 才开始有意义。1.5=落盒
                                  # t+10.0s 纪录，代价=导程残差 p50 79/max 283 mrad
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
        "catch_timeout": 8.0,     # s；死配置（无消费方——replay 读 [replay].
                                  # catch_timeout），留作历史遗留
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

# 2026-10-08 命名正规化（对照表见 docs/lerobot-alignment.md）：[loop] 旧键名 → 新键名。
# 命中旧键直接 SystemExit——三态回填（CLI > config > BUILTIN）会把缺键静默回落
# 内置默认，配置类错误宁可启动报错不可静默换值。
_RENAMED_LOOP_KEYS = {
    "steps_per_round": "n_action_steps",
    "steps_per_cmd": "steps_per_command",
}
_KNOWN_LOOP_KEYS = {
    "rounds", "n_action_steps", "steps_per_command", "delta_max", "speed",
    "pace", "pace_div", "near_div", "near_gap", "cache_frames",
    "max_excursion", "settle", "settle_frac", "switch_dist", "chunk_mode",
    "traj_dt", "catch_timeout",
}


def load(path=DEFAULT_PATH):
    """读 toml；文件缺失返回 {}（apply 逐键回落 BUILTIN）。

    [loop] 内旧键名（steps_per_round/steps_per_cmd）命中即 SystemExit 并提示
    新键名；未认键打印告警（tomllib 本身静默忽略未知键，见 config/README §7）。
    """
    p = pathlib.Path(path)
    if not p.exists():
        return {}, p
    with open(p, "rb") as f:
        data = tomllib.load(f)
    loop = data.get("loop")
    if isinstance(loop, dict):
        for old, new in _RENAMED_LOOP_KEYS.items():
            if old in loop:
                raise SystemExit(
                    f"[config] {p} [loop].{old} 已改名 {new}"
                    f"（2026-10-08 命名正规化，见 docs/lerobot-alignment.md），"
                    f"请更新 toml；旧键硬报错防静默回落内置默认")
        unknown = set(loop) - _KNOWN_LOOP_KEYS
        if unknown:
            print(f"[config] ⚠ {p} [loop] 未认键（tomllib 静默忽略，请核对拼写）: "
                  + ", ".join(sorted(unknown)))
    return data, p


def apply(args, mapping, path=DEFAULT_PATH):
    """按 mapping 把 config 值回填进 args 中 CLI 未显式指定的项。

    mapping: {args 的 dest: (toml 小节, 键)}。
    返回从 config 文件实际覆盖的 (dest, 值) 列表（内置默认兜底的不算）。
    [loop] 旧键名在 load() 即硬报错（防静默回落），见 _RENAMED_LOOP_KEYS。
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
