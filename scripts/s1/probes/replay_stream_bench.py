#!/usr/bin/env python
"""S1 流式回放台——把轨迹数据喂回 run_s1_loop 同一条下发路径，隔离模型变量
（G1 probes/replay_stream_bench.py 的 S1 镜像，2026-10-28）。

目的：①真机首验 tfs×singorix WBC 语义的最安全入口（轨迹冻结、无模型方差、
delta-max 链钳自带慢速就位效应——首行离当前读数远时被钳成小步渐进）；
②量化执行链（节拍抖动/块缝/回读残差）；③遥测为闭环对比预留同格式。

同构保证：**直接 import run_s1_loop 的代码对象**——stream_row / plan_row /
ema_row / send_grippers / sleep_to 原封复用，回放只换"动作来源"（文件/合成
vs 推理）。刻意比主执行器多一层：每 tick 重读臂关节刷新链钳基准（G1 traj
同款 per-tick re-read；主执行器现为每块读一次，真机 A/B 后再回移）。

数据源（win 上传轨迹）：
  --data file.npz ：chunks [C,L,D] 绝对目标 + joint_names [D]（S1 运行时实名）
                    + 可选 grip [C,L,2] 夹爪宽度 m〔left,right〕+ 可选 hz
  --data file.json：同构 {"chunks": ..., "joint_names": [...],
                          "grip": null | [...], "hz": 30}
  --synthetic N  ：本地合成平滑正弦块（离线冒烟/真机节拍测试，零依赖）
  支持关节族 = left_arm_joint1-7 / right_arm_joint1-7 / torso_lift_joint1
  （**米**） / head_joint1-2；未知关节名直接拒绝。lerobot parquet 转换器
  等样例到了再加（flash_pyrt311 无 pyarrow/pandas）。

A/B 维度（v1 精简，G1 台的 stream/direct/stepwait/tfs 谱系按需再移植）：
  --frames-per-row（1=纯 30Hz 直发，同执行器默认；>1=行内插值微流）
  --rows-per-cmd（跨行打包，窗口=R×行周期；10-28 窗口律：66.7ms 窗全安静）
  --filter-alpha（回放忠实默认 **0**，显式传才滤波——与执行器 toml 档无关）
  --max-joint-delta-per-step（臂链钳，默认 0.05；0=关仅限位）
  --swap-gap-ms（块间 hold 模拟推理落地窗；G1 定案忠实节奏=0）
  --rounds（整个序列重复遍数）

安全：dry-run 默认零 SDK（任何机器可跑）；--exec 需交互确认+start_controller
（急停手不离，模式切换失力矩 10-27 教训）；包络 = S1_LIMITS 钳位 + delta-max
臂链钳（每 tick 重读基准）；Ctrl-C 即退并落盘部分遥测。

用法：
  干跑： python replay_stream_bench.py --synthetic 3
  真机： PYTHONPATH=/data/galbot/lib ~/miniforge3/envs/flash_pyrt311/bin/python \
         scripts/s1/probes/replay_stream_bench.py --data traj.npz --exec
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_INF = _HERE.parents[1] / "inference"
sys.path.insert(0, str(_INF))

# ── import run_s1_loop（其 argparse 在模块层 parse → 临时换空 argv）──
# 目的：stream_row/plan_row/ema_row/send_grippers/sleep_to 用同一代码对象，
# 回放即与主执行器逐指令同路径（回放台立意，G1 同构原则）。
_argv = sys.argv
sys.argv = ["run_s1_loop.py"]
_spec = importlib.util.spec_from_file_location("run_s1_loop", _INF / "run_s1_loop.py")
rsl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rsl)
sys.argv = _argv

from policies.base import (  # noqa: E402
    HEAD_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, S1Targets, TORSO_JOINTS)

AP = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--data", default=None, help="轨迹文件 .npz/.json（与 --synthetic 二选一）")
AP.add_argument("--synthetic", type=int, default=0,
                help="本地合成 N 块平滑正弦轨迹（50 行/块，17 关节族）")
AP.add_argument("--seed", type=int, default=8, help="合成轨迹随机种子")
AP.add_argument("--rounds", type=int, default=1, help="序列重复遍数")
AP.add_argument("--ctrl-hz", type=float, default=30.0, help="行节拍 Hz")
AP.add_argument("--execute-actions-per-chunk", "--n-action-steps",
                dest="execute_actions_per_chunk", type=int, default=0,
                help="每块消费行数：0=整块；N=只放前 N 行〔旧名 --n-action-steps〕")
AP.add_argument("--frames-per-row", type=int, default=1,
                help="每行流式帧数（1=不插值直发，执行器默认；>1=行内插值微流）")
AP.add_argument("--rows-per-cmd", type=int, default=1,
                help="每次 set_joint_commands 打包行数 R（窗口=R×行周期；"
                     "1=逐行。30Hz 实时窗口律配方=2×fr12，见 run_s1_loop）")
AP.add_argument("--filter-alpha", "--ema-alpha", dest="filter_alpha",
                type=float, default=0.0,
                help="行目标 EMA α=新样本权重；回放忠实默认 0〔旧名 --ema-alpha〕")
AP.add_argument("--max-joint-delta-per-step", "--delta-max",
                dest="max_joint_delta_per_step", type=float, default=0.05,
                help="臂链钳 rad/tick（0=关仅限位）〔旧名 --delta-max〕")
AP.add_argument("--swap-gap-ms", type=float, default=0.0,
                help="块间 hold ms（模拟推理落地窗；忠实回放=0，G1 定案）")
AP.add_argument("--seam-cont", dest="seam_cont", action="store_true",
                help="块界从上一块末目标续锚（数据连续口径，消接缝顿挫）；"
                     "默认关=从当前读数重锚（执行器同款安全行为，含 LP 尾反向）")
AP.add_argument("--log-hz", type=float, default=0.0,
                help="tick 内高频状态突发采样（pos/vel/acc/cur + 末发目标），"
                     "落 kaka_*.npz；0=关。根因排查用（电流=力矩代理）")
AP.add_argument("--marks", action="store_true",
                help="回放中按回车打异响标记（时间戳进 npz，与日志同钟）")
AP.add_argument("--no-read-back", action="store_true",
                help="块尾不做关节回读残差（exec 模式默认回读）")
AP.add_argument("--exec", dest="do_exec", action="store_true",
                help="真机下发（默认 dry-run 零 SDK）")
AP.add_argument("--out", default=os.path.expanduser("~/holy/evidence/s1_replay"))
args = AP.parse_args()

# 对齐下发参数到 run_s1_loop 的全局（复用代码对象读取的就是这份）
for _k, _v in dict(ctrl_hz=args.ctrl_hz,
                   execute_actions_per_chunk=args.execute_actions_per_chunk,
                   frames_per_row=args.frames_per_row,
                   rows_per_cmd=args.rows_per_cmd,
                   filter_alpha=args.filter_alpha,
                   max_joint_delta_per_step=args.max_joint_delta_per_step,
                   grip_velocity=0.05, grip_effort=10).items():
    setattr(rsl.args, _k, _v)
rsl._args[0] = rsl.args

ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
_ALLOWED = set(ARM_JOINTS + TORSO_JOINTS + HEAD_JOINTS)


def load_traj(path: str | None, synthetic: int):
    """→ (joint_names, chunks [C,L,D], grip [C,L,2]|None)。npz/json/合成。"""
    if path:
        p = pathlib.Path(os.path.expanduser(path))
        if p.suffix == ".npz":
            z = np.load(p, allow_pickle=False)
            chunks = z["chunks"].astype(np.float64)
            names = [str(s) for s in z["joint_names"]]
            grip = (z["grip"].astype(np.float64) if "grip" in z.files else None)
        elif p.suffix == ".json":
            d = json.loads(p.read_text())
            chunks = np.asarray(d["chunks"], dtype=np.float64)
            names = [str(s) for s in d["joint_names"]]
            grip = (np.asarray(d["grip"], dtype=np.float64)
                    if d.get("grip") is not None else None)
        else:
            raise SystemExit(f"仅支持 .npz/.json（parquet 转换器等样例到了再加）：{p}")
        if chunks.ndim != 3:
            raise SystemExit(f"chunks 须 [C,L,D]，got {chunks.shape}")
        if len(names) != chunks.shape[2]:
            raise SystemExit(f"joint_names({len(names)}) ≠ D({chunks.shape[2]})")
        if grip is not None and grip.shape != (*chunks.shape[:2], 2):
            raise SystemExit(f"grip 须 [C,L,2]（宽 m，left/right），got {grip.shape}")
        bad = sorted(set(names) - _ALLOWED)
        if bad:
            raise SystemExit(f"未知关节名 {bad}；允许={sorted(_ALLOWED)}")
        return names, chunks, grip
    if synthetic > 0:
        rng = np.random.default_rng(args.seed)
        names = ARM_JOINTS + TORSO_JOINTS + HEAD_JOINTS
        amp = np.array([0.15] * 14 + [0.05] + [0.10] * 2)
        ph = rng.uniform(0, 2 * np.pi, len(names))
        cyc = rng.uniform(1.0, 2.0, len(names))
        k = np.arange(50)[:, None]
        chunk = amp * np.sin(2 * np.pi * cyc * k / 50 + ph)      # (50, 17)
        chunks = np.repeat(chunk[None], synthetic, axis=0)       # 块间零缝（离线口径）
        return names, chunks, None
    raise SystemExit("须 --data PATH 或 --synthetic N")


def row_to_targets(names: list[str], row: np.ndarray,
                   grip_row: np.ndarray | None) -> S1Targets:
    """一行 (D,) → S1Targets（按关节族归组；爪维单走 grip 通道）。"""
    tg = S1Targets()
    la, ra, hd = [], [], []
    for nm, v in zip(names, row):
        if nm in LEFT_ARM_JOINTS:
            la.append(float(v))
        elif nm in RIGHT_ARM_JOINTS:
            ra.append(float(v))
        elif nm in TORSO_JOINTS:
            tg.torso = float(v)
        elif nm in HEAD_JOINTS:
            hd.append(float(v))
    tg.left_arm = np.array(la) if la else None
    tg.right_arm = np.array(ra) if ra else None
    tg.head = np.array(hd) if hd else None
    if grip_row is not None:
        tg.left_grip_m, tg.right_grip_m = float(grip_row[0]), float(grip_row[1])
    return tg


def synthetic_state() -> np.ndarray:
    """19 维读数占位（dry-run 无 SDK；臂/头 0、torso 0.5 m、爪 50%）。"""
    return np.array([0.0] * 7 + [50.0] + [0.0] * 7 + [50.0, 0.5, 0.0, 0.0],
                    dtype=np.float32)


def advance_arm_state(state_raw: np.ndarray, names: list[str],
                      tgt: list[float], tgt_names: list[str]) -> None:
    """dry-run 虚拟跟随：状态按计划目标推进（模拟 perfect tracking）。
    否则链钳对冻结零态每行全咬，离线钳位计数无信息量。"""
    cur = dict(zip(tgt_names, tgt))
    for nm in names:
        if nm in ARM_JOINTS and nm in cur:
            if nm in LEFT_ARM_JOINTS:
                state_raw[LEFT_ARM_JOINTS.index(nm)] = cur[nm]
            else:
                state_raw[8 + RIGHT_ARM_JOINTS.index(nm)] = cur[nm]


def refresh_arm_state(robot, state_raw: np.ndarray, names: list[str]) -> None:
    """每 tick 重读臂关节刷新链钳基准（G1 traj 同款 per-tick re-read；
    主执行器现为每块读一次——真机 A/B 后再回移）。只刷数据里出现的臂维。"""
    want = [nm for nm in names if nm in ARM_JOINTS]
    if not want:
        return
    vals = robot.get_joint_positions(joint_names=want)
    for nm, v in zip(want, vals):
        if nm in LEFT_ARM_JOINTS:
            state_raw[LEFT_ARM_JOINTS.index(nm)] = float(v)
        else:
            state_raw[8 + RIGHT_ARM_JOINTS.index(nm)] = float(v)


def read_back(robot, names: list[str], last_tgt: dict[str, float]) -> dict:
    """块尾回读残差（臂 mrad / torso mm / head mrad）。"""
    want = [nm for nm in names if nm in ARM_JOINTS + TORSO_JOINTS + HEAD_JOINTS]
    vals = robot.get_joint_positions(joint_names=want)
    res = {}
    for nm, v in zip(want, vals):
        tgt = last_tgt.get(nm)
        if tgt is None:
            continue
        res[nm] = (float(v) - tgt) * 1000.0   # 臂/头 rad→mrad；torso m→mm（数值同 ×1000）
    mx = max((abs(v) for v in res.values()), default=0.0)
    return {"residual_max": mx, "residual": res}


def _new_tel(executing: bool) -> dict:
    return {"config": {**{k: getattr(args, k) for k in (
        "data", "synthetic", "seed", "rounds", "ctrl_hz",
        "execute_actions_per_chunk", "frames_per_row", "rows_per_cmd",
        "filter_alpha", "max_joint_delta_per_step", "swap_gap_ms",
        "seam_cont")},
        "executing": executing},
        "chunks": []}


MARKS = []      # perf_counter 时刻（耳听异响打标）


def _mark_reader() -> None:
    import threading
    def _loop():
        for _line in sys.stdin:   # 裸回车也要记（按回车一次一记；勿 strip 过滤空行）
            MARKS.append(time.perf_counter())
            print(f"  ✓ 标记 #{len(MARKS)}", flush=True)
    threading.Thread(target=_loop, daemon=True).start()


def run_replay(robot, body_names, names, chunks, grip, executing: bool,
               tel: dict, log=None) -> None:
    """主回放循环——逐行下发路径与 run_s1_loop 逐指令同构。
    遥测累积进传入的 tel（Ctrl-C 时 main 侧可落盘部分遥测）。
    log=None 或 dict（--log-hz>0 时传 dict）：tick 内突发采样高频状态。"""
    period = 1.0 / args.ctrl_hz
    log_hz = float(getattr(args, "log_hz", 0.0) or 0.0)
    if log is not None:
        log.update(t=[], pos=[], vel=[], acc=[], cur=[], tick=[],
                   ticks_t=[], ticks_tgt=[], marks=MARKS, names=names)
    state_raw = (rsl.read_s1_state(robot, body_names)[0] if executing
                 else synthetic_state())
    last_tgt: dict[str, float] = {}          # 跨块缝计量：上一块最后行目标

    for rep in range(args.rounds):
        for ci, chunk in enumerate(chunks):
            tr = time.perf_counter()
            prev = (dict(last_tgt) if (args.seam_cont and last_tgt)
                    else rsl.state_to_prev(state_raw))  # 续锚上块末目标 / 重锚读数
            ema = dict(prev)
            n = (len(chunk) if args.execute_actions_per_chunk <= 0
                 else max(0, min(args.execute_actions_per_chunk, len(chunk))))
            clamped = frames = 0
            seam_max = None
            late_ms = []
            R = max(1, rsl.args.rows_per_cmd)
            pending = []      # 已规划待打包的行 (names, tgt)——窗口=R×行周期
            for k in range(n):
                if executing:
                    refresh_arm_state(robot, state_raw, names)
                tg = row_to_targets(names, chunk[k],
                                    grip[ci, k] if grip is not None else None)
                if args.filter_alpha > 0:
                    tg = rsl.ema_row(tg, ema, args.filter_alpha)
                rnames, tgt, cl = rsl.plan_row(tg, state_raw)
                clamped += cl
                if not executing:
                    advance_arm_state(state_raw, names, tgt, rnames)
                if executing:
                    rsl.send_grippers(robot, tg, state_raw)   # 爪逐行（矩阵实证爪无罪）
                pending.append((rnames, tgt))
                if k == 0 and last_tgt:
                    d = [abs(prev[nm] - last_tgt[nm]) * 1000
                         for nm in rnames if nm in last_tgt
                         and nm in ARM_JOINTS]
                    seam_max = max(d) if d else None
                if k == n - 1:
                    last_tgt = dict(zip(rnames, tgt))
                if len(pending) >= R or k == n - 1:
                    if executing:
                        frames += rsl.stream_rows(robot, pending[0][0],
                                                  [t for _, t in pending], prev)
                    for nm_i, tgt_i in pending:
                        for nm, v in zip(nm_i, tgt_i):
                            prev[nm] = v      # 批末行 = 新插值链基准
                    pending.clear()
                if executing and log is not None:
                    dl = tr + period * (k + 1)
                    k0 = time.perf_counter()
                    log["ticks_t"].append(k0)
                    log["ticks_tgt"].append([float(v) for v in tgt])
                    i = 0
                    while time.perf_counter() + 0.002 < dl:
                        st_j = robot.get_joint_states(joint_names=names)
                        # ⚠ SDK 返回序≠请求序（10-29 实测请求 [右臂,左臂] 返回
                        # [左臂,右臂]，同 group 乱序坑族）——必须按 joint_name 对号
                        by_nm = {s.joint_name: s for s in st_j}
                        if any(n not in by_nm for n in names):
                            raise RuntimeError(
                                f"get_joint_states 返回对不上号（请求 {len(names)} "
                                f"返回 {len(st_j)}，joint_name 空？）——拒绝采错标数据")
                        log["t"].append(time.perf_counter())
                        log["pos"].append([float(by_nm[n].position) for n in names])
                        log["vel"].append([float(by_nm[n].velocity) for n in names])
                        log["acc"].append([float(by_nm[n].acceleration) for n in names])
                        log["cur"].append([float(by_nm[n].current) for n in names])
                        log["tick"].append(len(log["ticks_t"]) - 1)
                        i += 1
                        # 网格点钳到 dl-2ms：不钳则末格越过后 nxt<dl 恒假，
                        # tick 尾部 ~3ms 全速空转（10-29 采样密度 bug 根因二）
                        nxt = min(k0 + (i + 1) / log_hz, dl - 0.002)
                        if time.perf_counter() < nxt:
                            rsl.sleep_to(nxt)
                late = (time.perf_counter() - (tr + period * (k + 1))) * 1000
                late_ms.append(max(0.0, late))
                rsl.sleep_to(tr + period * (k + 1))   # 绝对截止节拍；欠账不追发
            item = {"chunk": ci, "rep": rep, "rows": n, "clamped": clamped,
                    "frames": frames,
                    "late_ms_p50": float(np.percentile(late_ms, 50)),
                    "late_ms_max": float(np.percentile(late_ms, 100)),
                    "seam_max_mrad": seam_max}
            if executing and not args.no_read_back:
                item.update(read_back(robot, names, last_tgt))
            tel["chunks"].append(item)
            print(f"  chunk{ci} rep{rep}: rows {n} | 帧流 {frames} | 钳位 {clamped}"
                  f" | 缝 {seam_max if seam_max is None else f'{seam_max:.1f} mrad'}"
                  f" | 迟到 p50 {item['late_ms_p50']:.2f}/max {item['late_ms_max']:.2f} ms"
                  + (f" | 残差 max {item['residual_max']:.1f}"
                     if "residual_max" in item else ""))
            if args.swap_gap_ms > 0:
                time.sleep(args.swap_gap_ms / 1000.0)
    return tel


def _dump(tel: dict, log: dict | None = None) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"replay_{int(time.time())}.json"
    path.write_text(json.dumps(tel, ensure_ascii=False, indent=1))
    print(f"落盘 {path}")
    if log is not None and log.get("t"):
        import numpy as np
        kp = out / f"kaka_{int(time.time())}.npz"
        np.savez(kp, t=np.array(log["t"]),
                 pos=np.array(log["pos"]), vel=np.array(log["vel"]),
                 acc=np.array(log["acc"]), cur=np.array(log["cur"]),
                 tick=np.array(log["tick"]),
                 ticks_t=np.array(log["ticks_t"]),
                 ticks_tgt=np.array(log["ticks_tgt"]),
                 marks=np.array(MARKS),
                 names=np.array(log["names"]),
                 config=json.dumps(tel["config"]))
        print(f"落盘 {kp}（{len(log['t'])} 样本 × {len(log['names'])} 关节 | "
              f"标记 {len(MARKS)}）")


def main() -> int:
    names, chunks, grip = load_traj(args.data, args.synthetic)
    print(f"✓ 轨迹载入：{len(chunks)} 块 × {chunks.shape[1]} 行 × {chunks.shape[2]} 维"
          f"（{len(names)} 关节名；grip={'有' if grip is not None else '无'}）")

    if not args.do_exec:
        print("（dry-run：不发指令，零 SDK，只走节拍与护栏路径）")
        tel = _new_tel(executing=False)
        run_replay(None, None, names, chunks, grip, False, tel)
        _dump(tel)
        return 0

    from galbot_sdk.s1 import GalbotRobot
    robot = GalbotRobot()
    if not robot.init(set()):
        raise SystemExit("robot.init 失败（服务在跑吗？）")
    body_names = robot.get_joint_names(only_active_joint=True,
                                       joint_groups=["left_arm", "right_arm",
                                                     "torso", "head"])
    print(f"✓ SDK init；body 关节 {len(body_names)}")
    time.sleep(3)
    try:
        print("\n⚠⚠ --exec 将回放真实关节指令（无推理护栏之外的模型语义——"
              "轨迹即命令）。急停已释放？手臂旁无人/无障碍？手不离急停。")
        if input("确认继续？(y/N) ").strip().lower() != "y":
            raise SystemExit("未确认，退出。")
        robot.start_controller("all")
        print("✓ start_controller(all)——执行模式")
        if args.marks:
            _mark_reader()
            print("✓ 打标线程挂载：回放中听到异响按回车一次一记")
        tel = _new_tel(executing=True)
        log = ({"marks": MARKS} if args.log_hz > 0 else None)
        try:
            run_replay(robot, body_names, names, chunks, grip, True, tel, log)
            _dump(tel, log)
        except KeyboardInterrupt:
            _dump(tel, log)   # 中断也落盘部分遥测（v1 缺口：卡顿首跑日志全丢）
            raise
    finally:
        # SDK 优雅关停（执行器 main 尾同款三连；缺它=退出段 C++ 析构段错误，
        # 10-28 回放台拒答路径实证）
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n^C 即退（已发指令停止在最后目标；遥测未落盘）")
        sys.exit(130)
