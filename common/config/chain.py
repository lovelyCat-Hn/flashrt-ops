"""三态配置链通用核心（common 收编版，2026-10-30）。

优先级 CLI 显式 > config 文件 > 各机 BUILTIN（镜像兜底）。本模块只含与
机型无关的机制；键名正规化表（改名硬报错/未认键白名单）与 BUILTIN 档案
是各机政策，留在 scripts/{g1,s1}/inference/{g1,s1}_config.py。

mapping 两种形态同吃（apply 内按类型分流，行为与收编前逐字一致）：
  - S1 扁平：{dest: "toml顶层键"}
  - G1 分节：{dest: ("toml小节", "键")}

约定（原 g1_config/s1_config 文件头逐条保留）：
  - config 支持的 add_argument 一律 default=None（store_true 同样 default=None），
    随后 apply() 按三态解析回填。
  - 安全开关（--exec 真实运动）不进配置，每次命令行显式给。
  - 布尔想显式关：脚本提供 --no-X 反关旗标（config 写 false 也行）。
"""
import pathlib
import tomllib


def load_toml_resolved(p):
    """读 toml 并递归解析 include 继承（机型文件覆盖 base 同名键，按节深合并）。

    include 相对当前文件目录解析；基座文件缺失 = 硬报错（红线/语义默认静默
    丢失不可接受，同旧键硬报错哲学）。机型文件未写 include = 无继承、原样读。
    （原 g1_config._load_toml_resolved 逐行收编；S1 扁平档不用继承，勿迁移）
    """
    with open(p, "rb") as f:
        data = tomllib.load(f)
    include = data.pop("include", None)
    if include:
        base_p = (p.parent / include).resolve()
        if not base_p.exists():
            raise FileNotFoundError(
                f"[config] {p} include 的基座文件不存在: {base_p}"
                f"——继承链断了（cache_frames/tier 等红线会静默丢失），先补文件")
        merged = load_toml_resolved(base_p)
        for key, val in data.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **val}
            else:
                merged[key] = val
        data = merged
    return data


def _lookup(cfg, builtin, spec):
    """按 mapping spec 取值：str=扁平顶层键，tuple=(小节, 键)。"""
    if isinstance(spec, tuple):
        sec, key = spec
        return cfg.get(sec, {}).get(key, builtin.get(sec, {}).get(key))
    return cfg.get(spec, builtin.get(spec))


def _in_file(cfg, spec):
    """该键是否真来自 config 文件（内置兜底不算）。"""
    if isinstance(spec, tuple):
        return spec[1] in cfg.get(spec[0], {})
    return spec in cfg


def apply(args, mapping, path, builtin, loader=None):
    """按 mapping 把 config 值回填进 args 中 CLI 未显式指定的项（通用核心）。

    loader: 读档函数 path→dict；缺省=裸 tomllib.load（S1 扁平档原语义）。
    G1 小节档传 load_toml_resolved 获得 include 继承。改名键硬报错/未认键
    告警是各机 load() 的职责，本函数不感知。
    返回从 config 文件实际覆盖的 (dest, 值) 列表（内置默认兜底的不算）。
    """
    from_file = []
    exists = pathlib.Path(path).exists()
    cfg = {}
    if exists:
        cfg = loader(path) if loader is not None else _read_plain(path)
    for dest, spec in mapping.items():
        if getattr(args, dest) is not None:
            continue                          # CLI 显式传参，最高优先级
        val = _lookup(cfg, builtin, spec)
        if val is None:
            raise SystemExit(f"[config] {dest} 无 CLI 值、config 无该项、内置默认缺失")
        setattr(args, dest, val)
        if exists and _in_file(cfg, spec):
            from_file.append((dest, val))
    if exists:
        cover = ", ".join(f"{d}={v!r}" if not isinstance(v, str) or len(v) < 40
                          else d for d, v in from_file)
        print(f"[config] {pathlib.Path(path)}（优先级 CLI>config>内置）覆盖: "
              f"{cover or '无，全内置默认'}")
    else:
        print(f"[config] {path} 不存在，全内置默认（--config 可指定别处）")
    return from_file


def _read_plain(path):
    with open(path, "rb") as f:
        return tomllib.load(f)
