"""S1 运行脚本共享配置：config/s1.toml，优先级 CLI 显式 > config > 内置默认。

结构与 scripts/g1/inference/g1_config.py 同构（apply 三态回填/未认键告警/
旧键硬报错），通用机制已收编 common/config/chain.py（common 整理步1，
2026-10-30）——本文件只剩 S1 政策件：
  - S1 档为【扁平顶层键 = CLI dest 名】（config/s1.toml 即此形态，
    run_s1_loop.py 内联预扫描同读这份文件），mapping 值是键名字符串。
  - BUILTIN 是 config/s1.toml 的镜像兜底 + traj 标准配方键（S1 toml
    未列的键回落 G1 traj [lerobot] 定档值，见各键注释）。

约定（与 g1_config 一致）：
  - config 支持的 add_argument 一律 default=None，apply() 按三态解析回填。
  - 安全开关（--exec 真实运动）不进配置，每次命令行显式给。
  - 布尔想显式关：--no-X 反关旗标（config 写 false 也行）。
"""
import pathlib
import sys
import tomllib

_HOLY_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
if _HOLY_ROOT not in sys.path:
    sys.path.append(_HOLY_ROOT)   # 尾部追加：勿遮蔽脚本目录同名包（holy/policies 前科 10-30）

from common.config import chain  # noqa: E402

DEFAULT_PATH = "/home/galbot/holy/config/s1.toml"

# 与 config/s1.toml 同源镜像（改一边同步另一边）；run_s1_loop.py（适配器版
# 执行器）只消费其子集。单位：角度 rad｜时长 s｜速度 rad/s（爪 m/s）｜
# 百分比 %｜频率 Hz
BUILTIN = {
    # ── s1.toml 已列键（镜像）──
    "ckpt": ("/home/galbot/model_weights/pi05/"
             "S1_run_20261028_121132_b3f637a6_010000/pretrained_model"),
    "prompt": "test",         # 训练 task 字符串（tasks.parquet 全库唯一）
    "rounds": 200,
    "ctrl_hz": 30.0,          # 行节拍（traj 脚本 dest=fps）
    "execute_actions_per_chunk": 0,   # 0=换块即预取（水位=预取触发阈值）
    "frames_per_row": 1,      # 10-30 五角闭环定案=1 现役：单点也带 tfs=T 时间轴
                              # 真机静（时间轴存在性=唯一必要条件）；fr2~8=纹理备档
    "rows_per_cmd": 1,        # 跨行攒批 R：1=每拍一包（最简）；R>1 攒 R 拍一包
                              # （R1/R2 真机均静，取最简）
    "filter_alpha": 0.1,      # S1 初档（用户定案 2026-10-28；G1 现役 0.3）
    "max_joint_delta_per_step": 0.05,  # rad/步（对原始增量 0.00% 截断=封版）
    "grip_path": "joint",     # traj 标准（关节流；cmd=逃生门）
    "grip_velocity": 0.05,    # m/s 爪速度帽（关节流 velocity 字段被尊重）
    "grip_effort": 10,        # 爪 effort（同事同款；pyi：爪维生效）
    # ── traj 标准配方键（S1 toml 未列时回落；值=G1 traj [lerobot] 定档）──
    "skip_frames": True,      # 换块跳帧安装（水位 0 的官配对，见 traj --help）
    "aggregate_fn": "weighted_average",  # 换块融合 α新 0.7（真机 A/B 10-30：双层 crossfade+EMA 胜 latest_only 单层 EMA，后者判退）
    "cmd_every": 1,           # 纯流式直发
    "grip_state_cmd": False,  # S1 判读：state 爪维喂实时回读（SDK 爪回读滞后
                              # 未测，真机首跑专项核查；G1 定档 true 不照抄）
    "burst_reserve_ms": 0.0,  # 纯流式无突发预算
    "nav_suspend": False,     # S1 无导航栈（旗标保留兼容档键，生效 no-op）
    "execution_horizon": 0,   # trained-RTC 专属；S1 pi05 非 RTC 权重=0
    "head_crop": False,       # S1 训练口径=4:3 letterbox 全幅（s1.toml 判读）
    "cache_frames": 1,        # 1=每帧全量（2 已判死，G1 9/30 实录）
    "max_excursion": 3.0,     # rad 漂移护栏（同 G1）
    "grip_enabled": True,     # S1 数据集带爪维，爪下发默认开（--no-grip 反关）
    "grip_chg": 2.0,          # cmd 逃生门变化阈值 %（joint 路旁路）
}

# 命名正规化（对照 docs/lerobot-alignment.md）：命中旧键直接 SystemExit——
# 三态回填会把缺键静默回落内置默认，配置类错误宁可启动报错不可静默换值。
_RENAMED_KEYS = {
    "n_action_steps": "execute_actions_per_chunk",
    "skip_install": "skip_frames",
    "seam_overlap_alpha": "aggregate_fn",
    "fusion_alpha": "aggregate_fn",
    "ema_alpha": "filter_alpha",
    "delta_max": "max_joint_delta_per_step",
    "rtc_horizon": "execution_horizon",
    "grip_speed": "grip_velocity",
}
# 未认键告警白名单（run_s1_loop.py 消费、traj 移植版不消费的键也算认识）
_KNOWN_EXTRA = {"policy", "subtask_every", "live_frames"}


def load(path=DEFAULT_PATH):
    """读 toml；文件缺失返回 ({}, p)（apply 逐键回落 BUILTIN）。"""
    p = pathlib.Path(path)
    if not p.exists():
        return {}, p
    with open(p, "rb") as f:
        data = tomllib.load(f)
    for old, new in _RENAMED_KEYS.items():
        if old in data:
            raise SystemExit(
                f"[config] {p}.{old} 已改名 {new}（键名对齐同事栈词汇/"
                f"g1.toml [lerobot] 体系，见 docs/lerobot-alignment.md），"
                f"请更新 toml；旧键硬报错防静默回落内置默认")
    unknown = set(data) - set(BUILTIN) - _KNOWN_EXTRA
    if unknown:
        print(f"[config] ⚠ {p} 未认键（tomllib 静默忽略，请核对拼写）: "
              + ", ".join(sorted(unknown)))
    return data, p


def apply(args, mapping, path=DEFAULT_PATH):
    """按 mapping 把 config 值回填进 args 中 CLI 未显式指定的项。

    mapping: {args 的 dest: toml 顶层键}（扁平档）。通用机制见
    common/config/chain.py；S1 扁平档无 include 继承（裸 tomllib 原语义）。
    返回从 config 文件实际覆盖的 (dest, 值) 列表（内置默认兜底的不算）。
    """
    load(path)   # 改名键硬报错/未认键告警（政策件；缺失文件在此 no-op）
    return chain.apply(args, mapping, path, BUILTIN)
