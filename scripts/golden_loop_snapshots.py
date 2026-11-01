#!/usr/bin/env python
"""G1/S1 traj 执行器重构金样回归（common/ 收编工程闸，2026-10-30）。

common/ 整理工程的红线=「只搬不改、行为零变」——本 harness 是它的机械判据：

  capture：从现行 traj 脚本 ast 提取纯函数（脚本模块级过程式、import 即执行，
           只能 ast 提取），桩掉 SDK/全局，喂固定输入 → 输出存金样 JSON。
  check  ：同一批 case 用【当前仓库】重跑（函数还在旧脚本=legacy 提取；
           已搬进 common/=直接 import；两边都找不到=FAIL，case 丢失），
           与金样逐位对比（float 全等，不设容差——纯搬运不容许位漂）。

覆盖面（Phase 0 定档）：
  interp（g1/s1_traj_interp 双份互证）、config 三态链（双机 apply/改名硬报错/
  BUILTIN 快照）、traj S1：build_rows（EMA/链钳/接缝窗/boost 五态）、
  build_block_stream（fr1/fr8/包络钳/prev0 零缝步）、_rp_flush tfs 数学
  （R1fr1/R2fr8/reserve/tfs_ms/空队）、letterbox224/head_training_crop、
  包络护栏两件、pstats/LatencyTracker；traj G1 同名件独立金样（G1 现状口径闸）。

用法：
  python golden_loop_snapshots.py --capture   # 重建金样（仅封版/刻意重录时）
  python golden_loop_snapshots.py --check     # 每个迁移 phase 收尾必跑
金样件：scripts/golden_baseline/golden_<date>.json（入库，随重构 commit 走）。
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import pathlib
import platform
import sys
import tempfile
import time
import types

import numpy as np

HOLY = pathlib.Path(__file__).resolve().parents[1]
INF = HOLY / "scripts" / "s1" / "inference"
G1F = HOLY / "scripts" / "g1" / "inference"
GOLDEN_DIR = HOLY / "scripts" / "golden_baseline"
SEED = 20261030

S1_TRAJ = INF / "run_s1_loop_traj.py"
G1_TRAJ = G1F / "run_g1_loop_traj.py"

# ── ast 提取 ────────────────────────────────────────────────────────────────

def extract(path, names):
    """模块级 def/class 源码段提取；返回 {name: src}，缺失列在 missing。"""
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    return out, sorted(set(names) - set(out))


def exec_into(ns, srcs):
    for name, src in srcs.items():
        exec(compile(src, f"<golden:{name}>", "exec"), ns)


class _StubCmd:
    """JointCommand 桩：记录全部属性。"""

class _StubPoint:
    """TrajectoryPoint 桩。"""

class _StubTraj:
    """Trajectory 桩：joint_names/points 整体赋值语义。"""

class _StubRobot:
    """robot 桩：记录 set_joint_commands_batch 收到的包。"""
    def __init__(self):
        self.sent = []
        self.status = "SUCCESS"

    def set_joint_commands_batch(self, traj):
        self.sent.append({
            "joint_names": list(traj.joint_names),
            "points": [
                {"tfs": p.time_from_start_second,
                 "pos": [float(c.position) for c in p.joint_command_vec],
                 "vel": [float(c.velocity) for c in p.joint_command_vec],
                 "acc": [float(c.acceleration) for c in p.joint_command_vec],
                 "eff": [float(c.effort) for c in p.joint_command_vec]}
                for p in traj.points],
        })
        return self.status


def import_mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_interp():
    """s1_traj_interp（import 安全纯件）→ create_interpolator。"""
    return import_mod("s1_traj_interp", INF / "s1_traj_interp.py").create_interpolator


# ── 公共命名空间 ─────────────────────────────────────────────────────────────

ARM14 = [f"r{i}" for i in range(1, 8)] + [f"l{i}" for i in range(1, 8)]


def base_args(**kw):
    """build_rows/build_block_stream/_rp_flush 的 args 依赖面（现役配方值）。"""
    d = dict(delta_max=0.05, ema_alpha=0.1, seam_alpha=0.05, seam_beats=4,
             seam_boost_alpha=0.0, seam_boost_beats=4, fps=30,
             frames_per_row=1, merge=1, rows_per_cmd=1, burst_reserve_ms=0.0,
             tfs_ms=0.0, interp="cubic", grip_rawrows=False,
             grip_ema_alpha=0.0)
    d.update(kw)
    return types.SimpleNamespace(**d)


class _LiveNS(dict):
    """部分键动态透传模块属性（common 路径的活状态）。

    脚本提取路径里 exec 后的 ns 就是函数 globals——`global _EMA_STATE` 直写
    ns，case 读 ns["_EMA_STATE"] 天然新鲜；common 路径活状态住在 FU 模块，
    读键时透传 getattr 才等价。
    """

    def __init__(self, live):
        super().__init__()
        self._live = live   # {键: 模块}；读 ns[k] ≡ getattr(mod, k)

    def __getitem__(self, k):
        if k in self._live:
            return getattr(self._live[k], k)
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if k in self._live:
            setattr(self._live[k], k, v)
        super().__setitem__(k, v)


def common_namespace(func_names, args, **extra):
    """函数已搬进 common/（步2b-1 起脚本不再含 def）：import + 模块属性回填。

    返回与 traj_namespace 等形的 ns；找不到映射的函数名返回 None。
    """
    from common.control import fusion as FU
    from common.control import guards as GD
    from common.loop import traj_engine as TE
    home = {"build_rows": FU, "build_block_stream": FU,
            "chunk_abs_block": GD, "env_first_violation": GD,
            "_rp_flush": TE}
    try:
        mods = {n: home[n] for n in func_names}
    except KeyError:
        return None
    ns = _LiveNS({"_EMA_STATE": FU, "_rp_q": TE, "_rp_n": TE})
    robot = _StubRobot()
    ns.update({"np": np, "args": args, "ARM_NAMES": ARM14,
               "TICK": 1.0 / args.fps, "robot": robot,
               "TrajectoryPoint": _StubPoint, "JointCommand": _StubCmd,
               "Trajectory": _StubTraj})
    ns.update(extra)
    for n, m in mods.items():
        ns[n] = getattr(m, n)
    FU.args = args
    for k in ("chunk", "BASE_ARM", "INTERP", "GRIP", "GRIP_INTERP",
              "ENV_LO", "ENV_LO14", "ENV_HI14"):
        setattr(FU, k, extra.get(k))
    FU._EMA_STATE = extra.get("_EMA_STATE")
    GD.BASE_ARM = extra.get("BASE_ARM")
    GD.ENV_LO = extra.get("ENV_LO")
    GD.ENV_HI = extra.get("ENV_HI")
    TE.args = args                       # _rp_flush（TE 模块属性）回填
    TE.TICK = 1.0 / args.fps
    TE.ARM_NAMES = ARM14
    TE.robot = robot                     # 与 ns["robot"] 同实例（sent 读数）
    TE.TrajectoryPoint, TE.JointCommand, TE.Trajectory = (
        _StubPoint, _StubCmd, _StubTraj)
    TE._rp_q = list(extra.get("_rp_q", []))
    TE._rp_n = extra.get("_rp_n", 0)
    return ns


def traj_namespace(func_names, args, **extra):
    """双后端取名：脚本仍含 def=ast 提取（legacy 仓库态）；已搬 common/=import。

    返回 (ns, missing)——两边都取不到才 missing 非空（case 丢失=FAIL）。
    """
    srcs, missing = extract(S1_TRAJ, func_names)
    if not missing:
        ns = {"np": np, "args": args, "ARM_NAMES": ARM14, "TICK": 1.0 / args.fps,
              "BASE_ARM": None, "chunk": None, "_EMA_STATE": None,
              "ENV_LO": None, "ENV_HI": None, "ENV_LO14": None, "ENV_HI14": None,
              "GRIP": None, "_rp_q": [], "_rp_n": 0,
              "TrajectoryPoint": _StubPoint, "JointCommand": _StubCmd,
              "Trajectory": _StubTraj, "robot": _StubRobot(),
              "create_interpolator": load_interp()}
        ns.update(extra)
        exec_into(ns, srcs)
        return ns, []
    ns = common_namespace(func_names, args, **extra)
    if ns is None:
        return None, missing
    return ns, []


# ── G1 机体函体命名空间（步2c：G1 分歧函体现状口径闸）─────────────────────

class _G1CmdStub:
    """galbot_sdk.g1 JointCommand 桩（仅 position/velocity/effort 载荷面）。"""

    def __init__(self):
        self.position = self.velocity = self.effort = 0.0


class _G1GroupStub:
    """G1JointGroup 桩：getattr → 同名字符串（爪名即 token，进记录可比）。"""

    right_gripper = "right_gripper"
    left_gripper = "left_gripper"


class _G1SensorStub:
    """SensorType 桩：getattr → 同名字符串。"""

    HEAD_RIGHT_CAMERA = "HEAD_RIGHT_CAMERA"
    HEAD_LEFT_CAMERA = "HEAD_LEFT_CAMERA"


class _StubGS:
    """get_gripper_state 快照桩（width 米/velocity/effort 三字段）。"""

    def __init__(self, width, velocity, effort):
        self.width, self.velocity, self.effort = width, velocity, effort


class _StubRobotG1:
    """G1 SDK 面桩：读关节/取图/爪下发/爪回读。grip_script=回读弹射队列
    （空→恒 (0.05, 0, 100)）；rgb={相机名: jpeg 字节}；joint_fail=读失败分支。
    joint_vals=关节读数 fixture；grip_cmds/cmds=下发记录（金样可比）。"""

    def __init__(self):
        self.joint_vals = [round(0.01 * i, 3) for i in range(16)]
        self.joint_fail = False
        self.rgb = {}
        self.grip_script = []
        self.grip_cmds = []      # set_gripper_command (组, w, spd, effort, block)
        self.cmds = []           # set_joint_commands ([(p,v,e)], 组, tfs)
        self.grip_width = None   # 最近下发的爪宽（米）

    def get_joint_positions(self, groups, names):
        if self.joint_fail:
            return []
        return list(self.joint_vals[:len(names)])

    def get_rgb_data(self, cam, *a, **k):
        d = self.rgb.get(cam)
        return {"data": d} if d is not None else None

    def _gs(self):
        if self.grip_script:
            return _StubGS(*self.grip_script.pop(0))
        return _StubGS(0.05, 0.0, 100.0)

    def get_gripper_state(self, group):
        return self._gs()

    def set_gripper_command(self, group, w, spd, effort, block):
        self.grip_cmds.append((group, round(w, 9), spd, effort, block))
        self.grip_width = w
        return "ControlStatus.SUCCESS"

    def set_joint_commands(self, cmds, joint_names=None, joint_groups=None,
                           time_from_start_s=0.0):
        self.cmds.append(([(c.position, c.velocity, c.effort) for c in cmds],
                          list(joint_groups or []), time_from_start_s))
        if cmds:
            self.grip_width = cmds[0].position
        return "ControlStatus.SUCCESS"


# G1 机体的模块级活状态（legacy=ns 键 / common=bodies.g1 模块属性）
_G1_STATE_KEYS = ("_GRIP_EMA", "_GRIP_BEAT", "_GRIP_SS_STATE", "_GRIP_SS_SENT",
                  "_GRIP_SENT_AT", "_GRIP_PRE_HIST", "_GRIPJ_EMA",
                  "_GRIPJ_HOLD", "_GRIPJ_TX", "_LAST_FRAMES", "_FREEZE_STREAK",
                  "GRIP", "GRIP_SS", "GRIP_WMIN", "GRIP_WMAX", "CAM_MAP")
_G1_STATE_INIT = {"_GRIP_EMA": {}, "_GRIP_BEAT": 0, "_GRIP_SS_STATE": {},
                  "_GRIP_SS_SENT": {}, "_GRIP_SENT_AT": {}, "_GRIP_PRE_HIST": {},
                  "_GRIPJ_EMA": {}, "_GRIPJ_HOLD": {}, "_GRIPJ_TX": 0,
                  "_LAST_FRAMES": {}, "_FREEZE_STREAK": {}}


def g1_args(**kw):
    """G1 机体函体的 args 依赖面（traj 现役旗标名）。"""
    d = dict(do_exec=True, grip_state_cmd=True, head_crop=False,
             grip_ema_alpha=0.0, grip_send_every=0, grip_min_interval=0,
             grip_preclose=0, gripj_every=1, grip_hold_on_stall=False,
             grip_hold_gap_mm=5.0, grip_hold_window=3, grip_hold_plateau_mm=1.0,
             grip_hold_exit_mm=8.0, grip_hold_margin_mm=1.0,
             grip_hold_relax_mm=0.0, grip_hold_relax_eff=90.0,
             grip_hold_vel=0.0, grip_effort_hold=0.0, grip_hold_quiet_n=0)
    d.update(kw)
    return types.SimpleNamespace(**d)


def g1_namespace(func_names, args, **extra):
    """G1 机体函体双后端取名：脚本仍含 def=ast 提取（G1 现状闸，capture 口）；
    已搬 common/bodies/g1.py=import+模块属性回填（check 口）。SDK 面全桩化
    （两侧同桩=位可比，不依赖本机是否装 g1 SDK）。"""
    import time as _time
    import cv2
    srcs, missing = extract(G1_TRAJ, func_names)
    if not missing:
        from common.bodies import vision as V
        ns = {"np": np, "time": _time, "os": os, "cv2": cv2,
              "sys": sys, "signal": __import__("signal"),
              "subprocess": __import__("subprocess"),
              "G1JointGroup": _G1GroupStub, "JointCommand": _G1CmdStub,
              "SensorType": _G1SensorStub,
              "letterbox224": V.letterbox224,
              "head_training_crop": V.head_training_crop,
              "GRIP_IDX": (7, 15),
              "ARM_NAMES": (G1ARM14 := [f"right_arm_joint{i}" for i in range(1, 8)]
                            + [f"left_arm_joint{i}" for i in range(1, 8)]),
              "STATE_NAMES": None, "robot": _StubRobotG1(), "args": args,
              "VIEWS": 3}
        ns.update(_G1_STATE_INIT)
        ns.update({k: _G1_STATE_INIT.get(k) for k in
                   ("GRIP", "GRIP_SS", "GRIP_WMIN", "GRIP_WMAX", "CAM_MAP")})
        ns.update(extra)
        exec_into(ns, srcs)
        return ns, []
    from common.bodies import g1 as B
    ns = _LiveNS({k: B for k in _G1_STATE_KEYS})
    robot = _StubRobotG1()
    ns.update({"np": np, "time": _time, "robot": robot, "args": args,
               "JointCommand": _G1CmdStub, "G1JointGroup": _G1GroupStub,
               "SensorType": _G1SensorStub, "VIEWS": 3})
    # 机体模块的 SDK 类/状态对齐 legacy 桩位（每案重置=与 fresh ns 同语义）
    B.G1JointGroup, B.JointCommand, B.SensorType = (
        _G1GroupStub, _G1CmdStub, _G1SensorStub)
    for k, v in _G1_STATE_INIT.items():
        setattr(B, k, v)
    for k in ("GRIP", "GRIP_SS", "GRIP_WMIN", "GRIP_WMAX", "CAM_MAP"):
        setattr(B, k, extra.get(k))
    B.VIEWS = extra.get("VIEWS", 3)
    B.args = args
    for n in func_names:
        ns[n] = getattr(B, n)
    return ns, []


def _jpeg(arr):
    """ndarray → JPEG 字节（grab_views 桩载荷；同库编解码=逐位确定）。"""
    import cv2
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    return buf.tobytes()


def fixed_inputs(n_rows=6):
    """chunk=(n,16)〔臂7+爪+臂7+爪，pi05 G1 语义〕；base_arm=14 维臂位。"""
    rng = np.random.default_rng(SEED)
    chunk = (rng.standard_normal((n_rows, 16)) * 0.02).astype(np.float32)
    base_arm = (rng.standard_normal(14) * 0.5).astype(np.float32)
    return chunk, base_arm


def env16():
    rng = np.random.default_rng(SEED + 1)
    lo = np.concatenate([rng.standard_normal(14) * 0.5 - 2.0,
                         [-np.inf, -np.inf]]).astype(np.float32)
    hi = np.concatenate([rng.standard_normal(14) * 0.5 + 2.0,
                         [np.inf, np.inf]]).astype(np.float32)
    return lo, hi


# ── case 定义 ───────────────────────────────────────────────────────────────

def _j(x):
    """JSON 安全化（数组递归列表化，float 保全精度 repr）。"""
    if isinstance(x, np.ndarray):
        return _j(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, float) or isinstance(x, int) or isinstance(x, str) \
            or x is None or isinstance(x, bool):
        return x
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    return repr(x)


def run_case(cid):
    """单 case → JSON 可比结果。case 注册表（新阶段函数搬家后在此补 import 路径）。"""
    chunk, base_arm = fixed_inputs()
    last_tgt = (base_arm + 0.01).astype(np.float32)
    cur = (base_arm + 0.003).astype(np.float32)   # build_*(cur)=当前臂位读数 14 维

    # traj S1：build_rows 五态
    if cid == "build_rows/first_block":
        ns, miss = traj_namespace(["build_rows"], base_args(),
                                  chunk=chunk, BASE_ARM=base_arm)
        return {"missing": miss, "rows": _j(ns["build_rows"](cur))}
    if cid == "build_rows/ema_continuation":
        ns, _ = traj_namespace(["build_rows"], base_args(), chunk=chunk,
                               BASE_ARM=base_arm, _EMA_STATE=last_tgt.copy())
        return {"rows": _j(ns["build_rows"](cur, prev0=last_tgt)),
                "ema_state": _j(ns["_EMA_STATE"])}
    if cid == "build_rows/seam_ramp":
        ns, _ = traj_namespace(["build_rows"], base_args(seam_alpha=0.05),
                               chunk=chunk, BASE_ARM=base_arm,
                               _EMA_STATE=last_tgt.copy())
        return {"rows": _j(ns["build_rows"](cur, prev0=last_tgt, seam=True))}
    if cid == "build_rows/seam_boost":
        ns, _ = traj_namespace(["build_rows"],
                               base_args(seam_boost_alpha=0.3,
                                         seam_boost_beats=4),
                               chunk=chunk, BASE_ARM=base_arm,
                               _EMA_STATE=last_tgt.copy())
        return {"rows": _j(ns["build_rows"](cur, prev0=last_tgt, seam=True))}
    if cid == "build_rows/ema_off":
        ns, _ = traj_namespace(["build_rows"], base_args(ema_alpha=0.0),
                               chunk=chunk, BASE_ARM=base_arm)
        return {"rows": _j(ns["build_rows"](cur))}
    if cid == "build_rows/offset":
        rng = np.random.default_rng(SEED + 2)
        off = (rng.standard_normal(14) * 0.01).astype(np.float32)
        ns, _ = traj_namespace(["build_rows"], base_args(), chunk=chunk,
                               BASE_ARM=base_arm)
        return {"rows": _j(ns["build_rows"](cur, offset=off))}

    # traj S1：build_block_stream
    if cid in ("build_stream/fr1", "build_stream/fr8_env",
               "build_stream/fr8_prev0", "build_stream/fr8_offset"):
        kw = {"frames_per_row": 8} if cid != "build_stream/fr1" else {}
        env = env16() if cid.endswith("env") else (None, None)
        prev0 = last_tgt if cid.endswith("prev0") else None
        rng = np.random.default_rng(SEED + 2)
        off = (rng.standard_normal(14) * 0.01).astype(np.float32) \
            if cid.endswith("offset") else None
        lo14 = np.concatenate([env[0][:7], env[0][8:15]]) if env[0] is not None \
            else None
        hi14 = np.concatenate([env[1][:7], env[1][8:15]]) if env[1] is not None \
            else None
        args = base_args(**kw)
        ns, _ = traj_namespace(
            ["build_rows", "build_block_stream"], args, chunk=chunk,
            BASE_ARM=base_arm, ENV_LO=env[0], ENV_HI=env[1],
            ENV_LO14=lo14, ENV_HI14=hi14,
            INTERP=load_interp()("cubic", dim=14, input_hz=30,
                                 output_hz=30 * args.frames_per_row))
        rows, frames, bursts, grip = ns["build_block_stream"](
            cur, prev0=prev0, offset=off)
        return {"rows": _j(rows), "frames": _j(frames), "bursts": bursts,
                "grip_none": grip is None,
                "first_frame_eq_prev0": (None if prev0 is None else
                                         bool(np.array_equal(
                                             frames[0], np.asarray(prev0))))}

    # traj S1：_rp_flush tfs 数学
    if cid in ("rp/R1fr1", "rp/R2fr8", "rp/reserve", "rp/tfs_ms", "rp/empty"):
        rng = np.random.default_rng(SEED + 3)
        R, fr = (2, 8) if cid == "rp/R2fr8" else (1, 1)
        reserve = 5.0 if cid == "rp/reserve" else 0.0
        tfs_ms = 2.5 if cid == "rp/tfs_ms" else 0.0
        args = base_args(rows_per_cmd=R, frames_per_row=fr,
                         burst_reserve_ms=reserve, tfs_ms=tfs_ms)
        ns, _ = traj_namespace(["_rp_flush"], args)
        frames = [(rng.standard_normal(14) * 0.01).astype(np.float32)
                  for _ in range(R * fr)]
        if cid != "rp/empty":
            ns["_rp_q"].extend(frames)
            ns["_rp_n"] = R
        st, n = ns["_rp_flush"]()
        sent = ns["robot"].sent
        tfs = [p["tfs"] for p in sent[0]["points"]] if sent else []
        return {"status": st, "n": n, "tfs": tfs,
                "pos0": sent[0]["points"][0]["pos"] if sent else None,
                "names": sent[0]["joint_names"] if sent else None,
                "q_left": len(ns["_rp_q"]), "rp_n_left": ns["_rp_n"],
                "vel0": sent[0]["points"][0]["vel"] if sent else None}

    # 图像件（步2a 起住 common/bodies/vision；仍回退脚本提取以兼容未迁移仓库态）
    if cid in ("img/letterbox224", "img/head_crop"):
        try:
            import cv2
        except ImportError:
            return {"skipped": "no cv2"}
        rng = np.random.default_rng(SEED + 4)
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        which = "letterbox224" if cid == "img/letterbox224" \
            else "head_training_crop"
        srcs, missing = extract(S1_TRAJ, ["letterbox224", which])
        if len(srcs) == 2:
            ns = {"np": np, "cv2": cv2,
                  "HEAD_TRAINING_CROP_YX": (126, 214, 52, 172),
                  "HEAD_TRAINING_CONTENT_HW": (164, 224),
                  "HEAD_TRAINING_PAD_TBLR": (30, 30, 0, 0)}
            exec_into(ns, srcs)
        else:
            from common.bodies import vision as V
            ns = {"np": np, "cv2": cv2,
                  "letterbox224": V.letterbox224,
                  "head_training_crop": V.head_training_crop,
                  "HEAD_TRAINING_CROP_YX": V.HEAD_TRAINING_CROP_YX,
                  "HEAD_TRAINING_CONTENT_HW": V.HEAD_TRAINING_CONTENT_HW,
                  "HEAD_TRAINING_PAD_TBLR": V.HEAD_TRAINING_PAD_TBLR}
        if cid == "img/head_crop":
            img = ns["letterbox224"](img)   # crop 期望 224² letterbox 输出
        out = ns[which](img)
        return {"shape": list(out.shape), "sum": float(out.astype(np.int64).sum()),
                "corner": _j(out[0, 0].astype(np.float32)),
                "center_px": _j(out[112, 112].astype(np.float32))}

    # 护栏
    if cid in ("guard/env_first", "guard/chunk_abs"):
        lo, hi = env16()
        ns, _ = traj_namespace(["chunk_abs_block", "env_first_violation"],
                               base_args(), chunk=chunk, BASE_ARM=base_arm,
                               ENV_LO=lo, ENV_HI=hi)
        absb = ns["chunk_abs_block"](chunk)
        viol = ns["env_first_violation"](absb)
        return {"abs_head": _j(absb[0]), "viol": _j(viol)}

    # 遥测（步2a 起住 common/loop/telemetry；回退脚本提取兼容未迁移仓库态）
    if cid == "tele/pstats":
        srcs, _missing = extract(S1_TRAJ, ["LatencyTracker", "pstats"])
        if len(srcs) == 2:
            from collections import deque
            ns = {"np": np, "deque": deque}
            exec_into(ns, srcs)
        else:
            from common.loop import telemetry as T
            ns = {"LatencyTracker": T.LatencyTracker, "pstats": T.pstats}
        tr = ns["LatencyTracker"]()
        for v in (10.0, 12.0, 11.5, 40.0, 9.0, 13.0, 12.5, 11.0):
            tr.add(v)
        return {"pstats": ns["pstats"](tr)}

    # interp 双份互证 + 金样
    if cid == "interp/frames_cubic":
        mk1 = load_interp()
        mk2 = import_mod("g1_traj_interp",
                         G1F / "g1_traj_interp.py").create_interpolator
        rng = np.random.default_rng(SEED + 5)
        wp = np.cumsum(rng.standard_normal((7, 14)) * 0.05, axis=0) \
            .astype(np.float32)
        i1 = mk1("cubic", dim=14, input_hz=30, output_hz=30)
        i2 = mk2("cubic", dim=14, input_hz=30, output_hz=30)
        f1 = i1.interpolate(wp)
        f2 = i2.interpolate(wp)
        i3 = mk1("linear", dim=14, input_hz=30, output_hz=240)
        f3 = i3.interpolate(wp)
        return {"cubic": _j(f1), "s1_eq_g1": bool(np.array_equal(f1, f2)),
                "linear_240": _j(f3)}

    # G1 机体函体（步2c：G1 分歧函体现状口径闸；SDK 全桩，两侧同桩位可比）
    if cid == "g1body/state_from_joints":
        vals = [0.0] * 16
        vals[7], vals[15] = 0.0585, 0.011          # 行程 50% / 3.88%
        ns, miss = g1_namespace(["state_from_joints"], g1_args(),
                                GRIP_WMIN=0.007, GRIP_WMAX=0.11)
        r1 = ns["state_from_joints"](vals)
        ns2, _ = g1_namespace(["state_from_joints"], g1_args(),
                              GRIP_WMIN=None, GRIP_WMAX=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r2 = ns2["state_from_joints"](vals)    # 未标定分支：原样+⚠
        return {"missing": miss, "st": _j(r1), "st_nocal": _j(r2),
                "warn": buf.getvalue()}

    if cid == "g1body/grip_state_cmd_override":
        st = [0.0] * 16
        GR = {"names": [("right_gripper", 7), ("left_gripper", 15)],
              "sent": {"right_gripper": 0.0, "left_gripper": 60.0},
              "sent_peak": {"right_gripper": 80.0, "left_gripper": 0.0},
              "chg": 2.0}
        ns, _ = g1_namespace(["grip_state_cmd_override"], g1_args(), GRIP=GR)
        r1 = ns["grip_state_cmd_override"](st)     # R 持物签名 33.5 / L 喂指令
        ns2, _ = g1_namespace(["grip_state_cmd_override"], g1_args(), GRIP=None)
        r2 = ns2["grip_state_cmd_override"](st)    # 无爪恒等
        ns3, _ = g1_namespace(["grip_state_cmd_override"],
                              g1_args(grip_state_cmd=False), GRIP=GR)
        r3 = ns3["grip_state_cmd_override"](st)    # 旗标关恒等
        return {"holding": _j(r1), "no_grip": _j(r2), "flag_off": _j(r3)}

    if cid == "g1body/read_joints":
        ns, _ = g1_namespace(["read_joints"], g1_args())
        rb = ns["robot"]
        ok = ns["read_joints"](rb, [f"j{i}" for i in range(14)])
        rb.joint_fail = True
        try:
            ns["read_joints"](rb, [f"j{i}" for i in range(14)])
            fail = None
        except SystemExit as e:
            fail = str(e)
        return {"vals": _j(ok), "fail": fail}

    if cid == "g1body/send_grip_chg":
        GR = {"names": [("right_gripper", 7), ("left_gripper", 15)],
              "sent": {}, "sent_peak": {}, "chg": 2.0,
              "wmin": 0.007, "wmax": 0.11, "speed": 0.15, "effort": 30}
        ns, _ = g1_namespace(["send_grip"], g1_args(), GRIP=GR)
        rb = ns["robot"]
        rb.grip_script = [(0.02, 0.0, 100.0)] * 4
        row = [0.0] * 16
        row[7], row[15] = 50.0, 20.0
        c1 = ns["send_grip"](rb, row)              # 首拍双爪发
        c2 = ns["send_grip"](rb, row)              # 变化 <chg → hold
        row2 = list(row)
        row2[7] = 60.0
        c3 = ns["send_grip"](rb, row2)             # 变化超阈 → 发
        return {"c1": c1, "c2": c2, "c3": c3, "sent": _j(GR["sent"]),
                "cmds": _j(rb.grip_cmds)}

    if cid == "g1body/send_grip_ss":
        GR = {"names": [("right_gripper", 7)], "sent": {}, "sent_peak": {},
              "chg": 2.0, "wmin": 0.007, "wmax": 0.11, "speed": 0.15,
              "effort": 30}
        ns, _ = g1_namespace(["send_grip"], g1_args(), GRIP=GR,
                             GRIP_SS=(50.0, 20.0))
        rb = ns["robot"]
        rb.grip_script = [(0.02, 0.0, 100.0)] * 3
        row = [0.0] * 16
        row[7] = 60.0
        c1 = ns["send_grip"](rb, row)              # 首拍定态 hi 发
        c2 = ns["send_grip"](rb, row)              # 无翻转 hold
        row[7] = 30.0
        c3 = ns["send_grip"](rb, row)              # 中间带未到合阈 hold
        row[7] = 15.0
        c4 = ns["send_grip"](rb, row)              # 合阈 lo 发
        ns["_GRIP_SS_SENT"]["right_gripper"] = (time.monotonic() - 3.0, 15.0)
        row[7] = 31.0
        c5 = ns["send_grip"](rb, row)              # 漂移补发（0.5Hz 限频窗过）
        return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
                "ss_state": _j(ns["_GRIP_SS_STATE"]), "cmds": _j(rb.grip_cmds)}

    if cid == "g1body/send_grip_joint":
        GR = {"names": [("right_gripper", 7), ("left_gripper", 15)],
              "sent": {}, "sent_peak": {}, "chg": 2.0,
              "wmin": 0.007, "wmax": 0.11, "speed": 0.15, "effort": 30}
        ns, _ = g1_namespace(["send_grip_joint"], g1_args(
            grip_ema_alpha=0.3, grip_hold_on_stall=True), GRIP=GR)
        rb = ns["robot"]
        rb.grip_script = [(0.02, 0.0, 100.0)] * 8
        row = [0.0] * 16
        row[7], row[15] = 0.0, 100.0               # R 深闭合 / L 全开
        h1 = ns["send_grip_joint"](rb, row, 0.033)
        h2 = ns["send_grip_joint"](rb, row, 0.033)
        h3 = ns["send_grip_joint"](rb, row, 0.033)  # 实宽平台+深指令满窗 → 进入
        row2 = list(row)
        row2[7] = 80.0                              # 上穿 exit → HOLD退出
        h4 = ns["send_grip_joint"](rb, row2, 0.033)
        return {"h1": h1, "h2": h2, "h3": h3, "h4": h4,
                "hold": _j(ns["_GRIPJ_HOLD"]), "ema": _j(ns["_GRIPJ_EMA"]),
                "cmds": _j(rb.cmds)}

    if cid == "g1body/grab_views":
        img1 = (np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
                % 251)
        img2 = np.roll(img1, 7, axis=1)
        CAM = {"image": "HEAD_RIGHT_CAMERA", "left": "HEAD_LEFT_CAMERA"}
        ns, _ = g1_namespace(["grab_views"], g1_args(), CAM_MAP=CAM, VIEWS=2)
        rb = ns["robot"]
        rb.rgb = {"HEAD_RIGHT_CAMERA": _jpeg(img1),
                  "HEAD_LEFT_CAMERA": _jpeg(img2)}
        out1 = ns["grab_views"](rb)
        out2 = ns["grab_views"](rb)                # 同帧 → 冻结计数 1
        freeze = None
        try:
            for _ in range(2):                     # 计数 2、3 → 第 2 圈触发停车
                ns["grab_views"](rb)
        except SystemExit as e:
            freeze = str(e)
        return {"keys": sorted(out1), "shape": list(out1["image"].shape),
                "sum": int(out1["image"].astype(np.int64).sum()),
                "sum2": int(out2["left"].astype(np.int64).sum()),
                "freeze": freeze}

    if cid == "g1body/nav_resume":
        ns, _ = g1_namespace(["NavSuspend"], g1_args())
        nav = ns["NavSuspend"]()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            nav.resume()                           # 空 stopped=no-op
            nav.stopped = {2147483000: "galbot_vtn"}
            nav.resume()                           # 假 pid OSError 吞掉 → 全 CONT
        return {"out": buf.getvalue(), "stopped": _j(nav.stopped)}

    # config 链
    if cid == "config/s1":
        import s1_config as C
        return {"builtin": _j(C.BUILTIN),
                "renamed": _apply_renamed_case(C, ("ema_alpha", "filter_alpha")),
                "apply": _apply_flat_case(C)}
    if cid == "config/g1":
        sys.path.insert(0, str(G1F))
        import g1_config as C
        return {"builtin": _j({repr(k): v for k, v in C.BUILTIN.items()}),
                "renamed": _apply_renamed_case(C, None),
                "apply": _apply_g1_case(C)}

    raise KeyError(cid)


def _apply_flat_case(C):
    """S1 扁平 mapping 三态：toml 覆盖/CLI 优先/缺键回落内置。"""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "s1.toml"
        p.write_text('prompt = "hi"\nframes_per_row = 3\n')
        mapping = {"prompt": "prompt", "frames_per_row": "frames_per_row",
                   "rows_per_cmd": "rows_per_cmd"}
        a = types.SimpleNamespace(prompt=None, frames_per_row=None,
                                  rows_per_cmd=None)
        from_file = C.apply(a, mapping, path=str(p))
        r1 = {"prompt": a.prompt, "fpr": a.frames_per_row,
              "rpc": a.rows_per_cmd, "from_file": [(d, _j(v)) for d, v in from_file]}
        a2 = types.SimpleNamespace(prompt="CLI", frames_per_row=None,
                                   rows_per_cmd=None)
        C.apply(a2, mapping, path=str(p))
        r1["cli_wins"] = a2.prompt
        return r1


def _apply_g1_case(C):
    """G1 (小节,键) mapping：toml 小节读回。"""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g1.toml"
        p.write_text('[run]\nprompt = "pick"\n[lerobot]\nfilter_alpha = 0.2\n')
        mapping = {"prompt": ("run", "prompt"),
                   "filter_alpha": ("lerobot", "filter_alpha")}
        a = types.SimpleNamespace(prompt=None, filter_alpha=None)
        from_file = C.apply(a, mapping, path=str(p))
        return {"prompt": a.prompt, "fa": a.filter_alpha,
                "from_file": [(d, _j(v)) for d, v in from_file]}


def _apply_renamed_case(C, pair):
    """改名键硬报错（SystemExit 文本快照）。"""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.toml"
        if pair:
            p.write_text(f"{pair[0]} = 0.3\n")
        else:
            p.write_text('[run]\ndelta_max = 0.1\n')  # G1 旧键样本
        try:
            C.apply(types.SimpleNamespace(), {}, path=str(p))
            return "NO_EXIT"
        except SystemExit as e:
            return str(e).replace(str(p.parent), "<tmp>")


# ── 注册表 / capture / check ────────────────────────────────────────────────

CASES = [
    "build_rows/first_block", "build_rows/ema_continuation",
    "build_rows/seam_ramp", "build_rows/seam_boost", "build_rows/ema_off",
    "build_rows/offset",
    "build_stream/fr1", "build_stream/fr8_env", "build_stream/fr8_prev0",
    "build_stream/fr8_offset",
    "rp/R1fr1", "rp/R2fr8", "rp/reserve", "rp/tfs_ms", "rp/empty",
    "img/letterbox224", "img/head_crop",
    "guard/env_first", "guard/chunk_abs",
    "tele/pstats", "interp/frames_cubic",
    "g1body/state_from_joints", "g1body/grip_state_cmd_override",
    "g1body/read_joints", "g1body/send_grip_chg", "g1body/send_grip_ss",
    "g1body/send_grip_joint", "g1body/grab_views", "g1body/nav_resume",
    "config/s1", "config/g1",
]


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def capture(out):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(INF))
    if str(HOLY) not in sys.path:
        sys.path.insert(1, str(HOLY))   # common/ 回退路径可达（2a 埋病此处补）
    cases = {}
    for cid in CASES:
        cases[cid] = run_case(cid)
        print(f"  ✓ {cid}")
    meta = {
        "date": "2026-10-30",
        "git_head": os.popen("git -C %s rev-parse --short HEAD"
                             % HOLY).read().strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "files": {str(p.relative_to(HOLY)): _sha(p)
                  for p in (S1_TRAJ, G1_TRAJ, INF / "s1_traj_interp.py",
                            G1F / "g1_traj_interp.py", INF / "s1_config.py",
                            G1F / "g1_config.py")},
    }
    doc = {"meta": meta, "cases": cases}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"金样已写 {out}（{len(cases)} case）")


def _eq(a, b, path=""):
    """逐位相等；返回差异描述列表。"""
    if isinstance(a, float) and isinstance(b, float):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a is b else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _eq(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            return [f"{path}: keys differ +{sorted(kb-ka)[:5]} "
                    f"-{sorted(ka-kb)[:5]}"]
        out = []
        for k in sorted(ka, key=repr):
            out += _eq(a[k], b[k], f"{path}.{k}")
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def check(golden):
    sys.path.insert(0, str(INF))
    if str(HOLY) not in sys.path:
        sys.path.insert(1, str(HOLY))   # common/ 回退路径可达（2a 埋病此处补）
    doc = json.loads(golden.read_text())
    base = doc["cases"]
    bad = 0
    for cid in CASES:
        if cid not in base:
            print(f"  ✗ {cid}: 金样缺此 case")
            bad += 1
            continue
        try:
            now = run_case(cid)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {cid}: 重跑异常 {type(e).__name__}: {e}")
            bad += 1
            continue
        diffs = _eq(base[cid], now, cid)
        if diffs:
            bad += 1
            print(f"  ✗ {cid}: {len(diffs)} 处位漂，首 3 条：")
            for d in diffs[:3]:
                print(f"      {d}")
        else:
            print(f"  ✓ {cid}")
    print(f"\n{'PASS' if bad == 0 else 'FAIL'}：{len(CASES)-bad}/{len(CASES)} "
          f"case 与金样 {golden.name}（git {doc['meta']['git_head']}）逐位一致")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true", help="重建金样基线")
    g.add_argument("--check", action="store_true", help="当前仓库 vs 金样")
    a = ap.parse_args()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden = GOLDEN_DIR / "golden_20261030.json"
    if a.capture:
        capture(golden)
    else:
        if not golden.exists():
            raise SystemExit(f"金样不存在：{golden}（先 --capture）")
        sys.exit(check(golden))


if __name__ == "__main__":
    main()
