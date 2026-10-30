"""S1 traj 闭环引擎（common 收编版 10-30 步2b-2b）。

原文=run_s1_loop_traj.py 逐行搬运不改逻辑：取图/块落盘件 + 推理 worker
（_PredictJob/fresh_obs/_launch）+ 换块融合主循环（②落地/护栏/干跑预览/
③执行确认/循环体/④汇总/归位/SDK 优雅三连）整体住 run()。跨模块状态走
模块属性：入口 init 后一次回填（见入口收编块）；run() 顶部 `global` 名单
由切片模块级绑定 ast 机器导出——chunk/BASE_ARM/job/k/last_tgt/_rp_q/_rp_n
等活状态保持为【本引擎模块属性】（非函数局部），嵌套函数既有 `global` 语句
照原样指向这里，worker 线程跨函数可见性与搬前逐位一致。名单含 for/if 体内
临时名（超声明无害），缺名才是 UnboundLocal/静默分叉。

    import common.loop.traj_engine as TE
    TE.args, TE.robot, TE.model, TE.ns = args, robot, model, ns
    ...
    TE.run()   # 不返回（内部归位 + SDK 三连 + os._exit(0)）
"""
import json
import os
import threading
import time

import cv2
import numpy as np

try:    # 离线纯分析（golden/回放台）零 SDK：类占位 None，用前由回填方注入
    from galbot_sdk.s1 import Trajectory, TrajectoryPoint, JointCommand
except ImportError:
    Trajectory = TrajectoryPoint = JointCommand = None

from flash_rt.core.utils.actions import normalize_state
from common.loop.telemetry import LatencyTracker, pstats
import common.control.fusion as FU
from common.control.fusion import build_block_stream
import common.control.guards as GD
from common.control.guards import chunk_abs_block, env_first_violation
from common.bodies.s1 import (
    ARM_NAMES,
    ARM_SLICE,
    STATE_NAMES,
    _GRIPJ_HOLD,   # 汇总区 hold 统计读（同 dict 对象原地改写）
    grab_views,
    grip_state_cmd_override,
    read_joints,
    s1_warm,
    send_grip,
    send_grip_joint,
    state_from_joints,
)

# ── 入口回填（一次，TE.run() 前）：init 只读面 ──
args = None          # argparse Namespace（CLI/config 合成；全键）
robot = None         # GalbotRobot 实例
model = None         # flash_rt.load_model 产物
ns = None            # model._pipe.norm_stats（state 归一化 stats）
TICK = None          # 1/args.fps 节拍（s）
WATCH = None         # QuitWatcher（q/Ctrl-C 后台监听）
HOME = None          # 起始位（漂移护栏基准）
WMARK = 0            # 预取水位阈值（0=换块即预取）
GAP_EPS = None       # 行导程免发门（与入口同值回填）
GRIP = None          # 爪标定/状态 dict（None=无爪）
ENV_LO = None        # 关节包络盒下界 (16,)（None=护栏关）
ENV_LO14 = None      # 包络盒下界臂 14 维（帧级钳位用）
ENV_HI14 = None      # 包络盒上界臂 14 维（帧级钳位用）
INTERP = None        # 臂轨迹插值器（create_interpolator 产物）
NAV = None           # S1 恒 None（G1 兼容位；尾部 resume 用）
_frames_dir = None   # --dump-frames 目录（None=关）
_dump_dir = None     # --dump-chunks 目录（None=关）
CKPT = None          # checkpoint 路径（落盘元数据用）
_env_file = None     # 包络盒 toml 路径（落盘元数据用）

_frame_seq = 0
env_hits = 0                         # 包络护栏：连续越界块数（干净块清零）
env_trunc = 0                        # 包络护栏：累计截断块数（汇总用）


def _dump_frames(obs):
    """三路模型输入图落盘（letterbox224 后 RGB）。在推理 worker 里调，
    done.set() 之后写=不推迟结果交接、不进 dur_ms；写失败静默（诊断用不阻塞）。"""
    global _frame_seq
    _frame_seq += 1
    for k, im in obs.items():
        try:
            cv2.imwrite(str(_frames_dir / f"{_frame_seq:04d}_{k}.png"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        except Exception:
            pass


def _dump_chunk(chunk_arr, base_arm, idx, infer_ms_val):
    """原始模型输出落盘（包络截断前）：chunk(50,16)+BASE_ARM(14)+元数据。"""
    if _dump_dir is None:
        return
    # 任务指纹（每目录一次）：npz 里没有 ckpt 信息，离线回放台按它配对同任务
    # 包络盒（pick/place 盒高度重叠，轨迹反推判别力为零——2026-10-12 实测）
    _meta = _dump_dir / "task.json"
    if not _meta.exists():
        _meta.write_text(json.dumps(
            {"ckpt": str(CKPT), "prompt": args.prompt,
             "envelope": str(_env_file) if _env_file.exists() else None,
             "delta_max": args.delta_max, "fps": args.fps},
            ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(str(_dump_dir / f"chunk_{idx:03d}.npz"),
             chunk=np.asarray(chunk_arr, dtype=np.float32),
             base_arm=np.asarray(base_arm, dtype=np.float32),
             block=idx, infer_ms=np.float64(infer_ms_val),
             fps=np.int64(args.fps))


def _rp_flush():
    """攒帧队列 → 一包带时间轴批；返回 (ControlStatus, 发送帧数)，空队=(None, 0)。

    窗口=R×tick（预算 burst_reserve），tfs=fdt·(i+1) 全批累计；冲账时队列短于
    满窗仍按满窗铺——臂走到段末目标后自然保持=缓停（勿缩短窗口造假速度）。"""
    global _rp_n
    if not _rp_q:
        return None, 0
    fdt = ((args.rows_per_cmd * TICK if args.rows_per_cmd > 1 else TICK)
           - args.burst_reserve_ms / 1000.0) / len(_rp_q)
    _pts = []
    for _bi, _fr in enumerate(_rp_q):
        _p = TrajectoryPoint()
        _p.time_from_start_second = round(
            fdt * (_bi + 1) + args.tfs_ms / 1000.0, 6)
        _vec = []
        for _ji in range(14):
            _c = JointCommand()
            _c.position = float(_fr[_ji])   # 标准关节只消费 position
            _c.velocity = _c.acceleration = _c.effort = 0.0
            _vec.append(_c)
        _p.joint_command_vec = _vec   # ⚠ getter 返副本：整体赋值
        _pts.append(_p)
    _traj = Trajectory()
    _traj.joint_names = ARM_NAMES
    _traj.points = _pts
    _st = robot.set_joint_commands_batch(_traj)
    _n = len(_rp_q)
    _rp_q.clear()
    _rp_n = 0
    return _st, _n


def run():
    # ── 主流程原文缩进整体搬入（步2b-2b）：global 名单=切片模块级绑定
    # （ast 机器导出，见文件头）；超声明无害，缺名才是分叉 ──
    global BASE_ARM, _, _PredictJob, _a, _abs, _back, _back_cmd, _behind, _bstep
    global _bursts, _bw, _bw_s, _cand, _cur0, _cur_sw, _d, _d_all, _d_old, _dir
    global _dom, _ds, _dur, _dur0, _e, _entry, _fd, _fi, _fmax, _frames, _ftol, _g
    global _g0, _g0_raw, _gaps, _gb, _gframes, _grp_idx, _grp_k0, _grp_n, _hs, _i
    global _intra, _is_merged, _j, _jcap, _k, _launch, _lvl, _m, _mag, _ndip, _nf
    global _nfr, _nrow, _obj, _obs_k, _old_chunk, _old_k, _old_rows_abs, _ovp
    global _ovp_off, _plan, _plan0, _prev0, _ramp_on, _rec, _ref_new, _rel, _rise
    global _roff, _rows, _rp_flush, _rp_n, _rp_q, _rtc_prefix, _send_this, _seq
    global _skip_n, _st, _starve_t0, _stf, _v, _viol, _w, _why, _wp, _wp0, _x
    global aborted, ach, b, blend_hits, bursts, chunk, cmd_delta, cur, d_mrad, desc
    global dim, drift, dropped_ticks, entries, entry_gap_hist, env_hits, env_trunc
    global err, exc, exhausted, fin, fr, fr_s, frames_all, fresh_obs, gap0, gp
    global grab_ms, grip_frames_all, gs, infer_ms, job, k, kk, last_tgt
    global merged_skips, n, n_fr, n_swapped, name, next_t, now, overlap_hits, p
    global prev_block_ticks, round_ms, rows_abs, seam_bw_all, seam_first
    global seam_first_all, seam_flush, seam_intra_all, seam_rest, skip_sends, st
    global starve_ticks, starve_total, starve_windows, starve_worst
    global swap_envelope_check
    global t_block0, t_loop0, t_tick, tgi, tgt, tick_ms, track_err, v, w
    def _rtc_prefix(new_base):
        """旧块未消费尾段 → 新块 delta 坐标系（RTC 前缀，推理侧引导输入）。

        臂维重锚：绝对构型 (OLD_BASE+delta) − NEW_BASE（新块自己的 delta 基准）
        夹爪维（7/15）数据集语义即绝对 0-100，原值携带。
        尾段不足 horizon 时按 lerobot rollout 语义 hold-pad（重复末行——零填充
        会解码到数据集均值）；块耗尽/RTC 关时返回 None（引擎侧 enable=0 短路）。
        须在 fresh_obs 之后、predict 之前调用（chunk/BASE_ARM 还属旧块，
        new_base=本次取关节读数，与 job 存下的 state_arm 同源）——10-29 审查①
        起随 fresh_obs 在推理工作线程内取（预取到下一换块的窗口里主线程只
        推进 last_tgt/k，无换块写重叠）。

        ⚠ 10-09 修正：前缀窗口右移 inference_delay 拍。前缀快照在抓帧时刻（拍 25），
        而引导的新块要到换块（拍 ~35=推理 330ms/33ms≈10 拍后）才落地执行；期间旧块
        尾段照常下发，臂已物理走过前缀描述的区域——不右移则每次换块都引导新块
        "倒退"回已走过的 ~10 拍（141706 实证：接缝 251 mrad≈方向反转，与 RTC-off
        无异）。此即 lerobot rtc 的 inference_delay 语义（补偿快照→落地延迟），
        此前误读为换块时机而设 0。延迟拍数=推理 p50÷tick，动态估计。
        """
        if not args.rtc_horizon or chunk is None:
            return None
        _p50 = infer_ms.percentile(0.5) if len(infer_ms) else 340.0
        # 右移帽=旧块剩余行数（勿用 WMARK）：水位 0 模式下 WMARK=0 会把补偿钳没，
        # 前缀窗口退回快照时刻=每块引导新块倒退（141706 同款）。水位 ≥1 时两帽
        # 等价（delay≈10 < 剩余行数），在役配方行为不变。
        delay = max(0, min(int(round(_p50 / (TICK * 1000.0))), len(chunk) - k - 1))
        j0 = k + delay
        if j0 >= len(chunk):
            return None                      # 延迟把旧块尾段吃尽 → 该块免引导
        left = chunk[j0:]
        pf = np.empty_like(left)
        pf[:, :7] = left[:, :7] + BASE_ARM[None, :7] - new_base[None, :7]
        pf[:, 7] = left[:, 7]
        pf[:, 8:15] = left[:, 8:15] + BASE_ARM[None, 7:14] - new_base[None, 7:14]
        pf[:, 15] = left[:, 15]
        h = args.rtc_horizon
        if len(pf) < h:
            pf = np.concatenate([pf, np.repeat(pf[-1:], h - len(pf), axis=0)], axis=0)
        return pf


    class _PredictJob:
        """后台推理作业：GPU 推理与固定节拍执行重叠。

        S1 版（10-29 审查①）：取图/读关节（fresh_obs，S1 三相机 ~86ms p50）也
        搬进工作线程——主线程同步取图≈2.6 拍，水位 0 下每块预取拍必丢拍重锚
        =块频 ~2.5Hz 卡顿节奏（与真机「卡卡」特征频率同区间）。主线程只保留下发
        和节拍。SDK 双线程并发（worker 读 vs 主线程下发/回读）未真机验证，
        --no-obs-worker 一键回退主线程取图（G1 traj 旧行为）对照。
        同一时刻只有一个 predict 在飞（到水位才启动下一个）。
        """

        def __init__(self, model, prompt):
            self._model, self._prompt = model, prompt
            self._done = threading.Event()
            self._res = None
            self._err = None
            self.dur_ms = 0.0

        def start(self, obs=None, state_n=None, state_arm=None):
            """obs 三件套=None（默认）→ 工作线程内 fresh_obs() 自取。

            rtc_prefix 不再由调用方预计算（它依赖 st_arm）：随 fresh_obs 在线程内
            取。_rtc_prefix 读 chunk/BASE_ARM/last_tgt 发生在「上一块已换入、下一
            换块未到」窗口（主线程此刻只在拍间睡眠推进 last_tgt/k，GIL 下逐项
            原子），与换块写窗口不重叠。"""
            self._done.clear()
            self._err = None
            self.state_arm = state_arm   # 本块 delta 基准（预测时刻臂位）

            def _run(obs=obs, state_n=state_n, state_arm=state_arm):
                # 默认参绑定（def 时求值）：_run 内对 obs 赋值会把闭包名变局部，
                # 前置的 `if obs is None` 必 UnboundLocal——别改回闭包引用
                t = time.perf_counter()
                try:
                    if obs is None:
                        obs, state_n, state_arm = fresh_obs()
                        self.state_arm = state_arm
                    self._res = np.asarray(self._model.predict(
                        obs, prompt=self._prompt, state=state_n,
                        rtc_prefix=_rtc_prefix(state_arm)))
                except Exception as e:      # 主线程 result() 时再抛
                    self._err = e
                self.dur_ms = (time.perf_counter() - t) * 1000
                self._done.set()
                if _frames_dir is not None and isinstance(obs, dict):
                    _dump_frames(obs)   # 推理 worker 里写盘，tick 零代价

            threading.Thread(target=_run, daemon=True).start()

        def done(self):
            return self._done.is_set()

        def result(self):
            self._done.wait()
            if self._err is not None:
                raise self._err
            return self._res


    def fresh_obs():
        """取图+读关节+夹爪换算+归一化（默认在推理工作线程跑，10-29 审查①；
        S1 三相机 ~86ms p50≈2.6 拍），耗时计入 grab_ms。

        返回 (obs, 归一化 state, 臂位原始值)——臂位原始值作该块 delta 基准随块走。
        """
        t_g = time.perf_counter()
        obs = grab_views(robot)
        st_raw = grip_state_cmd_override(
            state_from_joints(read_joints(robot, STATE_NAMES)))
        st_n = normalize_state(st_raw, ns)
        grab_ms.add((time.perf_counter() - t_g) * 1000)
        return obs, st_n, st_raw[ARM_SLICE]


    def _launch(job):
        """起推理作业：obs-worker 开（默认）=线程内自取图，主线程零阻塞；
        --no-obs-worker=主线程先取好再交线程（SDK 并发疑虑时的对照门）。"""
        if args.obs_worker:
            job.start()
        else:
            job.start(*fresh_obs())


    # ── ② 干跑：只打印首块前 5 拍计划，绝不下发 ──
    infer_ms = LatencyTracker()
    grab_ms = LatencyTracker()
    job = _PredictJob(model, args.prompt)
    _launch(job)                         # 首块（取图在工作线程，干跑同样口径）
    chunk = job.result()
    _dur0 = job.dur_ms
    infer_ms.add(_dur0)
    BASE_ARM = job.state_arm
    job = None
    FU.chunk, FU.BASE_ARM = chunk, BASE_ARM   # 活状态回填（重绑后 FU 侧换新对象）
    GD.BASE_ARM = BASE_ARM                    # 护栏比对基准同回填（chunk_abs_block 读 GD）
    _dump_chunk(chunk, BASE_ARM, 1, _dur0)


    def swap_envelope_check():
        """换块包络检查（v3 轮头同款）：越界截断，行 0 越界/连续越界→停。返回是否中止。

        非有限值（NaN/Inf）先于包络查：nan 与任何界的比较恒 False，会静默穿透
        包络检查与 np.clip，最终以 position=nan 下发（infer_policy 对非 finite
        动作显式拒收，2026-10-12 对照补齐）。截到首个污染行前缀；行 0 即污染→停。
        """
        global chunk, env_hits, env_trunc, aborted
        _bad = ~np.isfinite(chunk)
        if _bad.any():
            _r = int(np.flatnonzero(_bad.any(axis=1))[0])
            if _r == 0:
                print("⛔ 非有限值护栏：chunk 行 0 含 NaN/Inf——无处可截，停止循环"
                      "（机械臂留在原地）")
                aborted = True
                return True
            print(f"⛔ 非有限值护栏：chunk 行{_r} 含 NaN/Inf——截到行{_r}前缀"
                  "（模型输出异常，关注是否复发）")
            chunk = chunk[:_r]
            FU.chunk = chunk
        if ENV_LO is None:
            return False
        _v = env_first_violation(chunk_abs_block(chunk))
        if _v is None:
            env_hits = 0
            return False
        _r, _j, _val, _bnd = _v
        env_hits += 1
        env_trunc += 1
        _side = "上" if _val > _bnd else "下"
        print(f"⛔ 包络护栏：指令行{_r} 维{_j}={_val:.3f} 越{_side}界 "
              f"{_bnd:.3f}——截到行{_r}前缀（连续 {env_hits}/"
              f"{args.envelope_abort_n} 块）")
        if _r == 0:
            print("⛔ 行 0 即越界，无处可截——停止循环（机械臂留在原地）")
            aborted = True
            return True
        if env_hits >= args.envelope_abort_n:
            print("⛔ 连续越界=感知漂移（场景认错了），继续跑只会反复撞桌"
                  "——停止循环（机械臂留在原地）")
            aborted = True
            return True
        chunk = chunk[:_r]
        FU.chunk = chunk
        return False


    if not args.do_exec:
        print(f"\n== [干跑] 首块轨迹流预览（tick {TICK * 1000:.1f} ms，"
              f"{args.frames_per_row} 帧/行，批窗 {args.rows_per_cmd}×{TICK * 1000:.0f} ms，"
              f"链式限幅 ±{args.delta_max} rad）"
              "未下发任何命令 ==")
        _cur0 = read_joints(robot, ARM_NAMES)
        _rows, _frames, _bursts, _gframes = build_block_stream(_cur0)
        print(f"  行 {len(_rows)} → 帧 {len(_frames)}（{args.interp}）；"
              f"全块最大帧间步长 {float(np.max(np.abs(np.diff(_frames, axis=0)))) * 1000:.3f} mrad")
        for kk in range(3):
            fr = _frames[_bursts[kk]]
            _fd = float(np.max(np.abs(np.diff(fr, axis=0)))) * 1000 if len(fr) > 1 else 0.0
            print(f"  行{kk}: 帧 {len(fr)} | 行目标 {np.round(_rows[kk], 3).tolist()}"
                  f"\n        帧首→帧末位移 {float(np.max(np.abs(fr[-1] - fr[0]))) * 1000:.2f} mrad"
                  f" | 帧间最大步长 {_fd:.3f} mrad")
        if ENV_LO is not None:
            _v = env_first_violation(chunk_abs_block(chunk))
            _m = ("全块通过" if _v is None else
                  f"⛔ 行{_v[0]} 维{_v[1]}={_v[2]:.3f} 越界 {_v[3]:.3f}")
            print(f"  包络预检: {_m}（帧级另钳臂包络盒）")
        if GRIP:
            desc = []
            for name, dim in GRIP["names"]:
                p = float(np.clip(chunk[0][dim], 0.0, 100.0))
                w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
                desc.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm")
            print(f"  夹爪目标: {'  '.join(desc)}")
        print("\n[干跑] 未下发。继续无下发彩排至 --rounds 块"
              "（覆盖水位预取→换块弃尾→出口全路径；逐拍节奏见循环打印）\n")

    # ── ③ 固定节拍循环（绝对截止时序；水位预取；换块弃尾）──
    WATCH.pause()
    if args.do_exec:
        input(f"\n⚠ 将连续驱动双臂 {args.rounds} 块 × 至多 {len(chunk)} 拍"
              f"（tick {TICK * 1000:.1f} ms，轨迹流 {args.frames_per_row} 帧/行"
              f"≈{args.fps * args.frames_per_row}Hz {args.interp}，"
              + ((f"指令沿 合步 1/{args.cmd_every}={args.fps / args.cmd_every:.1f}Hz"
                  f"（每 {args.cmd_every} 拍 1 条）" if args.cmd_every > 1 else
                  (f"指令沿 合步×{args.merge}={args.fps * args.frames_per_row / args.merge:.0f}Hz"
                   f"（目标 {args.fps // args.merge}Hz 宏步、每宏步 {args.merge} 拍、"
                   f"拍线|Δcmd|=帧流口径"
                   + ("、爪随臂宏步插帧" if (GRIP is not None
                                             and not args.grip_rawrows) else "")
                   + "）" if args.merge > 1 else
                   f"指令沿 {args.fps:g}Hz 逐拍 1 条")))
              + (f"，tfs {args.tfs_ms:g} ms" if args.tfs_ms > 0 else "")
              + f"，链式限幅 ±{args.delta_max} rad/行，"
              f"漂移护栏 ±{args.max_excursion} rad"
              + ("，夹爪下发开启）。\n" if GRIP else ")。\n")
              + "急停就绪后回车开始，循环期间随时按 q 退出...")
    WATCH.resume()

    round_ms = LatencyTracker()          # 块耗时（换块到换块，对齐 v3 轮耗时口径）
    tick_ms = LatencyTracker()           # tick 服务时长（不含节拍睡眠）
    track_err = []                       # 每拍回读跟踪误差 mrad
    aborted = False
    k = 0                                # 本块已消费步数
    _obs_k = 0                           # 预取观测拍摄时的本块拍号（跳帧安装用）
    n_swapped = 1                        # 已换入块数（首块同步换入计 1）
    blend_hits = 0                       # 前向搭接触发块数（--blend）
    entry_gap_hist = []                  # 实发拍 |Δcmd|（rad）滚动窗，g₀ 中位用（免发不计）
    entries = []                         # 各块入口行 j（首块不计；汇总用）
    last_tgt = None                      # 上一拍下发目标（|Δcmd|/接缝基准）
    FU._EMA_STATE = None                 # 循环起点重置：干跑预览那次 build 不作块 1 的跨块历史（活状态在 FU）
    t_loop0 = time.perf_counter()        # 任务秒表
    t_block0 = t_loop0                   # 块秒表（换块节拍）
    seam_first = None                    # 换块后首拍 |Δcmd|（接缝尖峰，主指标）
    seam_rest = []                       # 换块后其余拍 |Δcmd|（块内均值基准）
    seam_first_all = []                  # 跨块累计：每块首拍 |Δcmd|
    seam_intra_all = []                  # 跨块累计：每块块内均值 |Δcmd|
    seam_bw_all = []                     # 跨块累计：每块首拍最负分量（缝口回走，mrad）


    def seam_flush():
        """把当前块的接缝样本并入跨块累计并复位（换块/收尾两处调用）。"""
        global seam_first, seam_rest
        if seam_first is not None:
            seam_first_all.append(seam_first)
        if seam_rest:
            seam_intra_all.append(float(np.mean(seam_rest)))
        seam_first = None
        seam_rest = []
    starve_windows = 0                   # 星饿段数（块耗尽等推理的连续段）
    starve_ticks = 0                     # 星饿拍总数
    starve_total = 0.0                   # 星饿总时长 s
    starve_worst = 0.0                   # 最长星饿段 s
    _starve_t0 = None                    # 当前星饿段起点（None=不在星饿）
    skip_sends = 0                       # 导程 <GAP_EPS 免发拍数
    merged_skips = 0                     # 合步静默拍数（cmd-every>1 只消费不发送）
    _grp_k0 = 0                          # ramp 当前组起点拍号（块内，--cmd-every-ramp）
    _grp_idx = 0                         # 已耗 ramp 组数（耗尽后恒 cmd_every）
    dropped_ticks = 0                    # 落后超 1 拍的重锚次数（丢拍）
    overlap_hits = 0                     # 重叠混接生效块数（--seam-overlap-alpha）
    _old_rows_abs = None                 # 旧块行流快照（换块捕获，混接惯性项）
    _old_chunk = None                    # 旧块原始 chunk 快照（爪维混接用）
    _old_k = 0                           # 换块时旧流下一行号

    # ── 跨行攒批（--rows-per-cmd R，10-29 审查②）：R=1=每拍一包（行内 fr 帧，
    # 33.3ms 窗）；R>1=攒 R 拍帧并一包 set_joint_commands_batch，tfs 铺满 R×tick
    # 窗（R2=66.7ms）、每 R 拍一发——命令窗口与行周期解耦，与适配器执行器/回放台
    # 真机验证过的 R×fr 形状同款。换块=半窗攒帧随旧块弃尾；星饿进入/自然收尾=
    # 先冲账（放完攒窗）再停/收。_rp_flush 下发形状与拍内批逐字同款（tfs 累计
    # =墙钟节拍，绝不追发）。
    _rp_q = []                           # 攒帧队列（帧数组引用；换块即弃）
    _rp_n = 0                            # 已攒拍数


    # ── _rp_flush 已提至本模块级（步2b-2b：金样/回放台取名面；
    #    全部状态=模块定向 global/属性，提级语义逐位等价）──
    next_t = time.perf_counter()
    swap_envelope_check()
    _cur0 = read_joints(robot, ARM_NAMES)
    rows_abs, frames_all, bursts, grip_frames_all = build_block_stream(_cur0)
    print(f"\n── 块 1 | t+0.0s | 推理 {infer_ms.percentile(1.0) if len(infer_ms) else float('nan'):.0f} ms ──")
    print(f"[轨迹流] 行 {len(rows_abs)} → 帧 {len(frames_all)}（{args.interp} "
          f"{args.frames_per_row} 帧/行，≈{args.fps * args.frames_per_row}Hz；"
          f"批窗 {args.rows_per_cmd}×{TICK * 1000:.0f}ms"
          + ("=跨行攒批" if args.rows_per_cmd > 1 else "=每拍一包")
          + f"，取图{'在worker' if args.obs_worker else '在主线程'}）")

    while not aborted:
        # ── 节拍头：睡到绝对截止；落后超 1 拍即重锚（绝不连发追赶）──
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        elif now - next_t > TICK:
            dropped_ticks += 1
            next_t = now
        next_t += TICK
        t_tick = time.perf_counter()

        # ── 换块：推理落地 → 整块换入弃尾（lerobot 水位语义）──
        if job is not None and job.done():
            # 重叠混接前置捕获（--seam-overlap-alpha）：旧块行流/chunk 在覆盖前
            # 存快照——混接的惯性项=「旧计划未执行段」，弃尾模式直接丢、混接模式
            # 留作 0.3 权重参考流（同事 _aggregate_action_queues 的 old 侧）。
            _old_rows_abs = rows_abs.copy() if rows_abs is not None else None
            _old_chunk = chunk.copy() if chunk is not None else None
            _old_k = k
            chunk = job.result()
            _dur = job.dur_ms
            infer_ms.add(_dur)
            BASE_ARM = job.state_arm
            job = None
            FU.chunk, FU.BASE_ARM = chunk, BASE_ARM   # 活状态回填（重绑后 FU 侧换新对象）
            GD.BASE_ARM = BASE_ARM                    # 护栏比对基准同回填（chunk_abs_block 读 GD）
            n_swapped += 1
            _dump_chunk(chunk, BASE_ARM, n_swapped, _dur)   # n_swapped 初值=1（首块计入）→ 块2 起 idx 连号不覆盖
            prev_block_ticks = k          # 上块消费拍数（重置前捕获，表头用）
            k = 0
            _grp_k0 = _grp_idx = 0        # ramp 组游标随块复位
            if _rp_n:                     # 换块弃尾：半窗攒帧随旧块丢（勿把旧块
                _rp_q.clear(); _rp_n = 0  # 参考放进新块的执行窗口）
            seam_flush()
            # 重叠混接门与偏移（须在跳帧安装消费 _obs_k 前捕获）：偏移=取图→落地
            # 流逝拍数=跳帧安装的 skip 数，同一把墙钟对齐键（同事栈 timestep 语义）
            # 门=开区间 0<α<1（10-30 用户定案「换块过渡=EMA 单层」：S1 在役
            # aggregate_fn=latest_only（α=1.0=旧流 0 权重=恒等）——恒等混接免重建
            # 帧免打印；缝平滑只留 EMA 跨块连续（α=filter_alpha，从 last_tgt 重播）。
            # 召回 crossfade：--aggregate-fn weighted_average 或 --fusion-alpha 0.7
            _ovp = (0 < args.seam_overlap_alpha < 1 and args.merge <= 1
                    and n_swapped > 1
                    and last_tgt is not None and _old_rows_abs is not None)
            # ⚠ merge 模式必须旁路：_ovp 块内按行网格 INTERP.interpolate 重建
            # frames_all（n_macro×merge×per 帧），与宏步突发下标表不匹配（突发
            # 只覆盖前 1/merge）——latest_only(α=1.0) 行混合虽是恒等，帧重建
            # 仍是破坏性的，故 merge>1 直接关门。
            _ovp_off = max(0, prev_block_ticks - _obs_k) if _ovp else 0
            if _starve_t0 is not None:
                _d = t_tick - _starve_t0
                starve_windows += 1
                starve_total += _d
                starve_worst = max(starve_worst, _d)
                _starve_t0 = None
            round_ms.add((t_tick - t_block0) * 1000)
            t_block0 = t_tick
            print(f"\n── 块 {n_swapped} | t+{t_tick - t_loop0:6.1f}s | "
                  f"推理 {infer_ms.percentile(1.0):.0f} ms（上块 {round_ms.percentile(1.0) / 1000:.1f}s，"
                  f"{prev_block_ticks} 拍）──")
            # ── 跳帧安装（--skip-install，infer_policy add_chunk skip_frames 同款）：
            # 新 chunk 行 j 对应墙钟 t_obs + j·tick；换块时已流逝
            # prev_block_ticks−_obs_k 拍，模型把这拍的 plan 重绘在行 0..skip——行 0
            # 接续=先消费历史（块界「减速→冲刺」脉动的构造性根源）。跳掉后时间槽
            # 对齐，首行天然在臂前方 ~lag 处（残差 7-30 mrad 级），回溯段不进链。
            # 星饿期 k 冻结在块长=旧帧全部计流逝，同语义（hold 帧不占 plan 时隙）。
            if args.skip_install and prev_block_ticks > _obs_k:
                _skip_n = min(prev_block_ticks - _obs_k, len(chunk) - 1)
                if _skip_n > 0:
                    chunk = chunk[_skip_n:].copy()
                    FU.chunk = chunk
                    _obs_k = 0
                    print(f"  [跳帧安装] 丢前 {_skip_n} 行（观测→落地流逝拍，"
                          f"墙钟槽位对齐接续；余 {len(chunk)} 行）")
            if swap_envelope_check():
                break
            # 换块重建轨迹流：锚=当前实际位（链式限幅从实际位重新出发，块内开环
            # 执行、块间重锚校正——轨迹路线的固有节奏，同 infer_policy）
            _cur_sw = read_joints(robot, ARM_NAMES)
            # ── 入口动态匹配（--entry-match，10-17 回放台移植；10-17 晚真机
            # 174448 破案改锚）：闭环模型每块从观测臂位重锚定 → 新 chunk 前段天然
            # 落在 last_tgt 之后（臂恒追着指令流跑），gap 若锚臂位则可行行只剩
            # 1-13/50、每块深跳 gap 30-117 mrad 或超上限→行 0 回拖——一冲一收
            # =摆幅（α 阶梯把深跳摊成拍1-6 的 12-13 mrad 反弹，块中 |Δcmd| 放大
            # 3×）。锚改 last_tgt（续流语义）：在计划自然跨回参考的位置接入，首拍
            # 步长≈g₀。①主导关节不退到臂后方 ②命令流单调（不把参考往回带；
            # 追账段反向阶跃=伺服急停反转=摆动，回放台 171925 实证）③距
            # last_tgt≈g₀ ④深跳超上限=行 0（EMA 阻尼小步，防棘轮）。
            # 门均无符号绝对化+显著运动地板（180940 二破案，见门内注）。
            # 匹配只含 14 臂关节维（chunk 第 7 列=夹爪，独立通道不参与）。
            if args.entry_match and last_tgt is not None and n_swapped > 1:
                _plan = BASE_ARM[None, :] + np.concatenate(
                    [chunk[:, :7], chunk[:, 8:15]], axis=1)
                if not entry_gap_hist:
                    entries.append(0)
                    print("  [入口匹配] 缺 g₀ 历史（上块全免发）→行 0")
                else:
                    _g0_raw = float(np.median(
                        entry_gap_hist[-args.entry_gap_win:]))
                    _g0 = float(np.clip(_g0_raw, args.entry_gap_min,
                                        args.entry_gap_max))
                    _dir = last_tgt - _cur_sw
                    _mag = np.abs(_dir)
                    # 显著运动地板（180940 二破案）：平静段参考-臂差全是几 mrad
                    # 噪声，主导集被噪声关节占满，方向门把微抖行全拒→30 行内凑不
                    # 出可行行→假回退→行0 落回计划锚点后阶梯展开进冲刺段=块内
                    # 呼吸（导程 10→42→11/块）。低于地板的关节不设门；全臂低于
                    # 地板=全部不设门（平静段只按 gap 匹配）。
                    _dom = ((_mag >= args.entry_dir_floor)
                            & (_mag >= args.entry_dir_frac
                               * max(float(_mag.max()), 1e-9)))
                    _rel = _plan - _cur_sw[None]
                    # 方向门三代教训（174448/180940/181946）：符号门锁过冲帧
                    # （_ds 反转把前进续流误判违约）、绝对门锁拖曳帧（计划爬不过
                    # last_tgt 时参考被钉在臂前 40-98 mrad，每块行0+阶梯展开=脉冲，
                    # 拖曳恒存）——两代各有一类「全窗不可行」死帧。--entry-no-dir-gate
                    # 全关后选择恒有解：argmin|gap−g₀| 自然落跨回点/最近逼近点，
                    # backward 步长被 α 阶梯阻尼到 ≤0.12×（软着陆是 EMA 本体职能，
                    # 门只是回放台开环内容下的校验器，闭环里越位）。
                    if args.entry_no_dir_gate:
                        _cand = np.arange(len(_plan))
                    else:
                        _ds = np.sign(_dir)
                        _back = (_dom[None, :]
                                 & ((_rel * _ds[None, :]) < -args.entry_dir_tol))
                        _back_cmd = (_dom[None, :]
                                     & (((_plan - last_tgt[None]) * _ds[None, :])
                                        < -args.entry_dir_tol))
                        _viol = (_back | _back_cmd).any(axis=1)
                        _cand = np.where(~_viol)[0]
                    # gap 锚=last_tgt（174448 改锚，见上注）：闭环臂恒落后参考，
                    # 锚臂位=每块深跳追参考；锚参考=在计划跨回指令流处接入。
                    _gaps = np.max(np.abs(_plan - last_tgt[None]), axis=1)
                    # 窗口完整余量 14 行（水位 0 块长≈13 拍）∩ 跳深上限
                    _jcap = min(max(0, len(chunk) - 14),
                                max(0, int(args.entry_j_cap)))
                    if len(_cand) and (_cand <= _jcap).any():
                        _cand = _cand[_cand <= _jcap]
                        # 前向偏好软代价（10-18 用户提案）：|gap−g₀| 绝对值匹配在参考
                        # 两侧同距行间分不出方向，选中后方侧=首拍倒退（EMA 阻尼≤0.12×
                        # 但非零）。目标加 ε×后方越界量——软偏好非硬门，无死帧，纯
                        # 选择层不加内容。后方越界只在主导关节上计（151241 二破案）：
                        # 臂恒落后参考一段导程，用 last_tgt 全关节计罚会把落后关节的
                        # ≈导程常数抬进所有浅行——ε 名义挑前向、实为挑深跳（j 被推
                        # 深、参考更热），且同罚不改排序只加热。限主导关节=纯「模型
                        # 自身是否在倒退」的度量；全臂平静段 _dom 全假=罚项自动退场。
                        _obj = np.abs(_gaps[_cand] - _g0)
                        if args.entry_fwd_bias > 0:
                            _behind = np.maximum(
                                (last_tgt[None, :] - _plan) * _dom[None, :],
                                0.0).max(axis=1)
                            _obj = _obj + args.entry_fwd_bias * _behind[_cand]
                        _entry = int(_cand[np.argmin(_obj)])
                        chunk = chunk[_entry:].copy()
                        FU.chunk = chunk
                        entries.append(_entry)
                        print(f"  [入口匹配] j={_entry}（g₀={_g0 * 1000:.1f}"
                              + (f"←{_g0_raw * 1000:.1f}夹紧"
                                 if _g0 != _g0_raw else "")
                              + f" gap={_gaps[_entry] * 1000:.1f} mrad"
                              + (f"，后越={_behind[_entry] * 1000:.1f}"
                                 if args.entry_fwd_bias > 0 else "")
                              + f"，可行行 {len(_cand)}/{len(_plan)}；"
                              f"余 {len(chunk)} 行）")
                    else:
                        entries.append(0)
                        _nf = np.where(~_viol)[0]
                        _why = (f"可行最早 j={int(_nf[0])} 仍在窗外"
                                if len(_nf) else "全窗无可行行")
                        print(f"  [入口匹配] 前跳超上限{_jcap}→行 0"
                              f"（{_why}，EMA 阻尼小步）")
                # ── 凹陷填补（--entry-fill-dip，10-18 用户提案）：门全关后
                # argmin|gap−g₀| 是绝对值匹配，参考两侧同距的行都可选——选中后方侧
                # =首拍倒退（残摆源）；块内模型抖动的短促倒退段也照样入流。修补=
                # 按关节把「回补窗内的短倒退凹陷」从坑前水平线性桥到回补行（用户
                # 例：5→−7→−4→…→12 填成 5→桥→12，"补到能顺利衔接上的那个拍"；
                # 取 0 持位/镜像取正侧都是桥的特例）。超窗不回补=持续后撤（真倒退）
                # 放行——真后撤是平滑趋势非方向翻转，伺服追得上，不吃模型合法修正。
                # 在 EMA 阶梯之前执行，EMA 再低通一次。
                if args.entry_fill_dip:
                    _abs = BASE_ARM[None, :] + np.concatenate(
                        [chunk[:, :7], chunk[:, 8:15]], axis=1)
                    _ftol = float(args.entry_fill_tol)
                    _fmax = max(2, int(args.entry_fill_max))
                    _seq = np.vstack([last_tgt[None, :], _abs])   # 锚=指令流尾
                    _ndip = _nrow = 0
                    for _j in range(_seq.shape[1]):
                        _x = _seq[:, _j]
                        _i = 1
                        while _i < len(_x):
                            if _x[_i] < _x[_i - 1] - _ftol:       # 凹陷起点
                                _lvl = _x[_i - 1]
                                _rec = -1
                                for _k in range(_i, min(len(_x), _i + _fmax)):
                                    if _x[_k] >= _lvl - _ftol:
                                        _rec = _k
                                        break
                                if _rec > 0:                       # 窗内回补→桥
                                    _x[_i:_rec] = np.linspace(
                                        _lvl, _x[_rec], _rec - _i + 2)[1:-1]
                                    _ndip += 1
                                    _nrow += _rec - _i
                                    _i = _rec
                                else:
                                    _i += 1                        # 真倒退放行
                            else:
                                _i += 1
                    chunk[:, :7] = _seq[1:, :7] - BASE_ARM[None, :7]
                    chunk[:, 8:15] = _seq[1:, 7:14] - BASE_ARM[None, 7:]
                    if _ndip:
                        print(f"  [凹陷填补] 桥接 {_ndip} 处 / {_nrow} 行")
            # ── EMA 重播（10-12 真机否决修正，161933 实测拍0 p90 294.6/max 559.3）：
            # _EMA_STATE 原存整条 50 行链滤波完的链尾，而水位 10 弃尾每块只发 ~40
            # 行——状态记在旧计划未发送的尾巴上（超前已发送点 ~10 行 ≈ 数百 mrad），
            # 下一块 rows[0]=0.3×新首行+0.7×旧链尾=块首整体瞬移、伺服逐块冲刺。
            # 改为从「最后实际下发目标」重播滤波：rows[0] 距臂 ≤ 滞后量级，且与
            # blend 搭接基点（last_tgt）同基准，EMA/blend 状态差随之消失。
            if args.ema_alpha > 0 and last_tgt is not None:
                FU._EMA_STATE = last_tgt.copy()   # 重播状态写 FU（活状态唯一份）
            # ── 参考链连续（skip-install 配套，10-13 真机 110411 判明）：链式限幅
            # 从当前臂位重建链会把参考钳回 cur+delta_max（拍0 导程 50），换块锯齿
            # 50→330→50 = 脉动的残留源（跳帧安装只对了时间槽，没动链锚）。
            # 跳帧安装后新块行 0 本就 ≈ 墙钟正确位（≈last_tgt±模型修正），链改从
            # last_tgt 接续：首帧恰=last_tgt（零缝步），参考全程前向不重锚——同
            # 同事方案「执行原始计划行、无重锚」的语义。entry-match 同需参考
            # 连续（首行=last_tgt+clip 修正，与入口判据同基准）。非二者口径不变。
            _prev0 = last_tgt if ((args.skip_install or args.entry_match
                                   or args.continuous_ref)
                                  and last_tgt is not None) else None
            _roff = None
            if _prev0 is not None and args.ref_offset:
                # 偏移平移（10-13）：新计划行 0 锚在观测滞后臂上，与参考链的失配量
                # 就是块首回走刹车的来源。整条计划平移 offset 后 rows[0]=last_tgt
                # （失配 ≤delta-max 时零缝步），臂速无块界刹车、导程呼吸趋平。
                # offset 逐维钳 ±delta-max：平移后目标距计划绝对位 ≤50 mrad，
                # 吃不满包络余量 ±0.15；残余失配仍由链式限幅回走收敛兜底。
                _plan0 = BASE_ARM + np.concatenate([chunk[0][:7], chunk[0][8:15]])
                _roff = np.clip(last_tgt - _plan0, -args.delta_max, args.delta_max)
            rows_abs, frames_all, bursts, grip_frames_all = build_block_stream(
                _cur_sw, prev0=_prev0, offset=_roff,
                seam=(args.entry_match and n_swapped > 1))
            # ── 重叠混接（--seam-overlap-alpha，2026-10-20 同事 lerobot 栈
            # weighted_average 移植）：跳帧安装已把新块行 i 对齐墙钟槽「旧流
            # 自拍 _old_k+i」，此处把混接窗内每行替换为 (1−α)·旧流+α·新链——
            # 旧计划从「被弃尾」变为「惯性项」，换块跳变被凸组合衰减并摊在
            # 整个重叠区（同事 _aggregate_action_queues：0.3·old+0.7·new，
            # 重叠区=新块与旧队列的 timestep 交集；水位 0 节奏下稳态=每行
            # 都是两代计划混合，接缝不再是事件而是永续惯性）。爪维 dim7/15
            # 数据集语义即绝对 0-100，同窗同权凸组合仍在值域。链式限幅/样条
            # /包络在混接前已作用于新链（凸组合的两个分量都近 last_tgt，
            # 结果不引入超限步长）；帧流随行目标重建（同 --blend 手法）。
            if _ovp:
                _a = float(args.seam_overlap_alpha)
                _m = min(len(_old_rows_abs) - _old_k, len(rows_abs))
                if _m > 0:
                    rows_abs[:_m] = ((1.0 - _a) * _old_rows_abs[_old_k:_old_k + _m]
                                     + _a * rows_abs[:_m])
                    chunk[:_m, 7] = ((1.0 - _a) * _old_chunk[_old_k:_old_k + _m, 7]
                                     + _a * chunk[:_m, 7])
                    chunk[:_m, 15] = ((1.0 - _a) * _old_chunk[_old_k:_old_k + _m, 15]
                                      + _a * chunk[:_m, 15])
                    _wp0 = (np.asarray(_prev0, dtype=np.float32)
                            if _prev0 is not None else np.asarray(_cur_sw,
                                                                  dtype=np.float32))
                    _wp = np.vstack([_wp0[None, :], rows_abs])
                    frames_all = INTERP.interpolate(_wp)
                    if ENV_LO is not None:
                        np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :],
                                out=frames_all)
                    overlap_hits += 1
                    print(f"  [重叠混接] 旧{1.0 - _a:g}/新{_a:g}｜偏移 {_ovp_off} 行"
                          f"｜混接窗 {_m} 行（旧流自拍 {_old_k}，块内实际消费至"
                          f"水位为止）")
                else:
                    print(f"  [重叠混接] 旧流已耗尽（自拍 {_old_k} ≥ 流长 "
                          f"{len(_old_rows_abs)}）→ 纯新计划接管")
            # ── 前向搭接（--blend，10-12 回放台定案移植）：新块前 M 行目标替换为
            # 旧链末目标→rows[M−1] 直连——换块时新块行 0 落在臂后方 ~lag（预取到
            # 落地 ~10 拍臂又走了），原链会俯冲进回溯段=块首往复；搭接让参考链
            # 全程前向、臂不停车不回摆。1:1 替换不加拍；末行恰=原 rows[M−1]，
            # EMA 状态链/包络凸性/消费配额均不受影响。门与回放台同款（滞后>60）。
            if args.blend and last_tgt is not None:
                _d_all = np.max(np.abs(rows_abs - _cur_sw[None, :]), axis=1)
                _d_old = float(np.max(np.abs(last_tgt - _cur_sw)))
                _rise = np.where(_d_all >= _d_old - 0.01)[0]
                _m = (int(_rise[0]) + 1) if len(_rise) else len(rows_abs) - 1
                _m = max(2, min(_m, len(rows_abs) - 2))
                if (_d_old > args.blend_gate
                        and _d_old - float(_d_all.min()) > args.blend_gate
                        and _m < len(rows_abs) - 1):
                    _bstep = (rows_abs[_m - 1] - last_tgt) / _m
                    for _g in range(_m):
                        rows_abs[_g] = last_tgt + _bstep * (_g + 1)
                    _wp = np.vstack([np.asarray(_cur_sw, dtype=np.float32)[None, :],
                                     rows_abs])
                    frames_all = INTERP.interpolate(_wp)   # 帧流随行目标重建（消俯冲）
                    if ENV_LO is not None:
                        np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :],
                                out=frames_all)
                    blend_hits += 1
                    print(f"  [搭接] 前 {_m} 行前向直连旧链末目标"
                          f"（滞后 {_d_old * 1000:.0f} mrad → 行{_m - 1}，"
                          f"每行 ≤ {float(np.max(np.abs(_bstep))) * 1000:.1f} mrad）")

        exhausted = k >= len(chunk)
        if exhausted:
            # ── 星饿：不下发（刚性伺服保持最后目标原地等待），只等推理 ──
            if _starve_t0 is None:
                _starve_t0 = t_tick
                if _rp_n:                 # 冲账：攒窗内剩余帧先放完（臂走到段末
                    _stf, _ = _rp_flush() # 目标自然缓停，勿带账进星饿保持）
                    if _stf is not None and not str(_stf).startswith("ControlStatus.SUCCESS"):
                        print(f"⛔ 星饿冲账下发非 SUCCESS（{_stf}），停止循环")
                        aborted = True
                        break
            starve_ticks += 1
            if t_tick - _starve_t0 > args.starve_limit:
                print(f"⛔ 星饿超 {args.starve_limit}s（推理未归）——判挂死，停止循环"
                      "（机械臂留在原地）")
                aborted = True
        else:
            # ── 合步节流（--cmd-every N>1，默认 1=零变）：非发送拍只消费不发送，
            # k 照常推进（参考时间线/水位/星饿/预取语义全部不动）；发送拍整组并
            # 1 条，目标=组末行绝对位（帧/爪同取组末行，口径一致）。
            # --cmd-every-ramp：缝侧组宽序列（1/1/2→N），组游标 _grp_k0/_grp_idx
            # 随块复位；组宽=1 的发送拍 tgi=k=锚行 j 本身（entry-match 匹配精度
            # 不再被 +N−1 吃掉）。g₀ 窗只记满宽组（单位=4 行组差，细笔拍不入窗）。
            _ramp_on = bool(args.cmd_every_ramp) and args.cmd_every > 1
            if _ramp_on:
                _grp_n = (args.cmd_every_ramp[_grp_idx]
                          if _grp_idx < len(args.cmd_every_ramp) else args.cmd_every)
                _send_this = k >= _grp_k0 + _grp_n - 1
                _is_merged = args.cmd_every > 1 and _send_this
                if _send_this:
                    _grp_k0, _grp_idx = k + 1, _grp_idx + 1
            else:
                _is_merged = args.cmd_every > 1 and (
                    k == 0 if args.cmd_first_only else k % args.cmd_every == 0)
                # 发送拍 = 合步拍，或（first-only 口径）组后恢复的逐拍流式拍
                _send_this = _is_merged or (args.cmd_first_only and k >= args.cmd_every)
                _grp_n = args.cmd_every
            if args.cmd_every > 1 and not _send_this:
                k += 1
                merged_skips += 1
            else:
                # ── 发送拍：合步拍发组末行、流式拍发本行（全部走 set_joint_commands
                # burst 接口；接口铁律见文件头——本脚本不碰 set_joint_positions）──
                tgi = (k if _ramp_on else
                       (min(k + args.cmd_every - 1, len(rows_abs) - 1) if _is_merged else k))
                cur = read_joints(robot, ARM_NAMES)
                tgt = rows_abs[tgi]
                gap0 = float(np.max(np.abs(tgt - cur)))
                # 拍线 |Δcmd| 口径（replay 台 cd4c1a3 同构）：merge 模式记真实下发
                # 帧流——发送拍=本拍最后发送帧 vs 上次发送帧，免发拍持有上次
                # （帧+0.0=真原地、免发=静默）；merge=1 行目标口径不变
                #（接缝尖峰 seam_first 历史可比）。
                if args.merge > 1:
                    _ref_new = (frames_all[bursts[tgi][-1]]
                                if (gap0 >= GAP_EPS and bursts[tgi]) else last_tgt)
                else:
                    _ref_new = tgt
                cmd_delta = ((_ref_new - last_tgt) if last_tgt is not None
                             else np.zeros_like(tgt))
                d_mrad = float(np.max(np.abs(cmd_delta))) * 1000
                if seam_first is None:
                    seam_first = d_mrad          # 接缝尖峰：换块首发 |Δcmd|
                    # 缝口回走：首拍有符号最负分量（>0=目标倒退）。|Δcmd| 无符号
                    # 看不见它——量级正常但方向倒退=伺服减速/反打，停顿主嫌疑。
                    _bw = float(np.max(-cmd_delta)) * 1000
                    seam_bw_all.append(_bw)
                    _bw_s = f" | ⚠回走{_bw:.1f}mrad" if _bw >= 0.5 else ""
                else:
                    seam_rest.append(d_mrad)
                    _bw_s = ""
                if grip_frames_all is not None:
                    # 爪随臂宏步帧流：发本拍末帧（突发完成时刻的插值目标，
                    # 与拍线 |Δcmd| 帧流口径同帧）；空突发=静默持有
                    _gb = bursts[tgi]
                    gp = (send_grip_joint(robot, None, args.tfs_ms / 1000.0,
                                          frame=grip_frames_all[_gb[-1]])
                          if _gb else None)
                else:
                    gp = (send_grip_joint(robot, chunk[tgi], args.tfs_ms / 1000.0)
                          if args.grip_path == "joint"
                          else send_grip(robot, chunk[tgi]))
                if (args.entry_match and gap0 >= GAP_EPS
                        and _grp_n == args.cmd_every):
                    # g₀ 滚动窗：只记实发拍（免发拍污染中位），同回放台口径
                    entry_gap_hist.append(d_mrad / 1000.0)
                    if len(entry_gap_hist) > 32:
                        del entry_gap_hist[:-32]
                drift = float(np.max(np.abs(tgt - HOME)))
                if drift > args.max_excursion:
                    print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                          f"{args.max_excursion * 1000:.0f}，停止循环（机械臂留在原地）")
                    aborted = True
                    break
                if gap0 >= GAP_EPS:
                    b = bursts[tgi]
                    if args.do_exec:
                        # 帧突发 → 带时间轴批【S1 移植版改形，文件头④】：G1 traj
                        # 此处=逐帧单点 tfs=0 流（G1 伺服在役形状）；S1 singorix
                        # WBC 按 tfs 时间轴细插，tfs=0=参考跳变→LP 削顶→力矩跳变
                        # =卡卡（S1 ③定案 10-29 晚真机实锤，「时间轴是解药非病因」）。
                        # 故并一包 set_joint_commands_batch：tfs 全批累计=墙钟节拍
                        # （fdt·(i+1)，绝不追发；--tfs-ms>0=每点追加到达时限余量）。
                        # --rows-per-cmd R>1 再跨拍攒窗（窗口=R×tick、每 R 拍一发；
                        # 10-29 审查②，适配器执行器/回放台 R×fr 真机形状同款）。
                        _rp_q.extend(frames_all[_fi] for _fi in b)
                        _rp_n += 1
                        st = None
                        if _rp_n >= args.rows_per_cmd:
                            _st, n_fr = _rp_flush()
                            if not str(_st).startswith("ControlStatus.SUCCESS"):
                                st = _st
                        else:
                            n_fr = len(b)
                            st = f"攒批 {_rp_n}/{args.rows_per_cmd}"
                        if st is not None and not str(st).startswith("攒批"):
                            print(f"⛔ 下发非 SUCCESS（{st}），停止循环")
                            aborted = True
                            break
                    else:
                        st = "dry"   # 干跑彩排：打印节奏与遥测，绝不下发
                        n_fr = len(b)
                else:
                    skip_sends += 1
                    n_fr = 0
                    st = "held（导程≈0 免发）"
                last_tgt = _ref_new.copy() if _ref_new is not None else None
                k += 1
                ach = read_joints(robot, ARM_NAMES)
                err = float(np.max(np.abs(ach - tgt))) * 1000
                track_err.append(err)
                # 合步拍打实际下发沿频率（沿=fps/组宽，ramp 变宽逐拍如实）；
                # 流式拍维持 @参考流密度 旧口径（fps×frames_per_row）
                if n_fr:
                    fr_s = (f"{n_fr}帧·沿{args.fps / _grp_n:.0f}Hz" if _is_merged
                            else f"{n_fr}帧@{args.fps * args.frames_per_row / args.merge:.0f}Hz")
                else:
                    fr_s = ("· 无帧 ·" if (args.merge > 1 and not bursts[tgi])
                            else "· 免发 ·")
                print(f"  拍{k - 1}: |Δcmd| {d_mrad:5.1f} mrad"
                      f" | 导程 {gap0 * 1000:5.1f} → {fr_s}"
                      f" | 残差 {err:4.1f} mrad"
                      + (f" | 合步[{tgi}]×{_grp_n}" if _is_merged else "")
                      + _bw_s
                      + f" | {gp if gp else st}"
                      + (f" 〈攒批 {_rp_n}/{args.rows_per_cmd}〉"
                         if args.do_exec and 0 < _rp_n < args.rows_per_cmd else ""))

        # ── 出口：最后一块消费到水位（或包络截短后耗尽）──
        _w = min(WMARK, len(chunk))
        if n_swapped >= args.rounds and (k >= _w or k >= len(chunk)):
            break

        # ── 预取：到水位且无在飞推理 → 启动下块（取图+推理全在工作线程，
        # 10-29 审查①：主线程同步取图 ~86ms≈2.6 拍=每块必丢拍）──
        if job is None and n_swapped < args.rounds and k >= _w:
            _obs_k = k               # 观测拍摄时点（跳帧安装的流逝基准）
            job = _PredictJob(model, args.prompt)
            _launch(job)

        tick_ms.add((time.perf_counter() - t_tick) * 1000)

    # 收尾冲账（自然出口：放完最后半窗攒帧再进汇总/归位；aborted 出口不冲——
    # 护栏/星饿/非 SUCCESS 停机让臂留在原地，攒帧一并作废）
    if args.do_exec and not aborted and _rp_n:
        _stf, _nfr = _rp_flush()
        if _stf is not None and not str(_stf).startswith("ControlStatus.SUCCESS"):
            print(f"⚠ 收尾冲账下发非 SUCCESS（{_stf}；共 {_nfr} 帧）")

    # ── ④ 汇总 ──
    seam_flush()
    if _starve_t0 is not None:
        _d = time.perf_counter() - _starve_t0
        starve_windows += 1
        starve_total += _d
        starve_worst = max(starve_worst, _d)
    if GRIP and args.do_exec:
        time.sleep(8.0)   # 等爪走完最后一程再读终态（死区 0.9s+满程 0.8s，含余量）
        for name, _ in GRIP["names"]:
            gs = robot.get_gripper_state(name)
            if gs is not None:
                print(f"夹爪终态 {name}: {gs.width * 1000:.1f} mm "
                      f"(moving={gs.is_moving} eff={gs.effort:.1f})")
        if args.grip_hold_on_stall and _GRIPJ_HOLD:
            _hs = "  ".join(f"{n.split('_')[0].upper()} 进{v['enters']}"
                            f"{'/保持中' if v['on'] else ''}"
                            f"{'/松' + str(v['relax_n']) if v['relax_n'] else ''}"
                            f"{'/静' + str(v['silent_n']) if v['silent_n'] else ''}"
                            for n, v in _GRIPJ_HOLD.items())
            print(f"爪憋死保持统计: {_hs}")
    fin = read_joints(robot, ARM_NAMES)
    exc = float(np.max(np.abs(fin - HOME))) * 1000
    print(f"\n== 汇总（{n_swapped} 块 / {len(tick_ms)} 拍 / tick {TICK * 1000:.1f} ms / "
          f"轨迹流 {args.frames_per_row} 帧/行 {args.interp}"
          f" / 批窗 {args.rows_per_cmd}×{TICK * 1000:.0f}ms）==")
    print(f"总用时: {time.perf_counter() - t_loop0:.1f} s"
          f"（数据集单条任务 13s / 390 步参照）")
    if len(infer_ms):
        print(f"推理: {pstats(infer_ms)}")
    if len(grab_ms):
        print(f"取图+读关节: {pstats(grab_ms)}")
    if len(tick_ms):
        print(f"tick 服务时长: {pstats(tick_ms)} | 丢拍重锚 {dropped_ticks} | 免发 {skip_sends}"
              + (f" | 合步静默 {merged_skips} 拍（cmd-every {args.cmd_every}）"
                 if args.cmd_every > 1 else ""))
    if len(round_ms):
        print(f"块耗时: {pstats(round_ms)}")
    if track_err:
        print(f"导程残差: mean {np.mean(track_err):.1f} | max {max(track_err):.1f} mrad"
              + (f"；包络护栏截断 {env_trunc} 块" if ENV_LO is not None else ""))
    if seam_first_all:
        _intra = float(np.mean(seam_intra_all)) if seam_intra_all else float("nan")
        print(f"接缝尖峰（换块首拍 |Δcmd|）: 均值 {np.mean(seam_first_all):.1f} | 最大 "
              f"{max(seam_first_all):.1f} mrad | 块内均值 {_intra:.1f} mrad"
              f"（RTC A/B 主指标：--rtc-horizon on 应显著压低首拍/块内比）"
              + (f"（合步口径：首发含 {args.cmd_every} 行组差，勿与直发跑直比）"
                 if args.cmd_every > 1 else "")
              + (f" | 搭接触发 {blend_hits} 块" if args.blend else ""))
    if seam_bw_all:
        _bw = np.asarray(seam_bw_all, float)
        print(f"缝口回走（换块首拍最负分量，>0=目标倒退）: 块 {len(_bw)} | "
              f"p50/p90/max {np.percentile(_bw, 50):.1f}/{np.percentile(_bw, 90):.1f}/"
              f"{_bw.max():.1f} mrad | ≥2mrad {int((_bw >= 2).sum())} 块"
              f"（回走大=伺服减速/反打，停顿嫌疑；对比 154843 好感基线）")
    if args.entry_match and entries:
        _e = np.asarray(entries, float)              # 仅块 2+ 追加（首块无入口）
        print(f"入口匹配: 块 {len(_e)} | j min/p50/max "
              f"{_e.min():.0f}/{np.median(_e):.0f}/{_e.max():.0f}"
              f"（行0 {int((_e == 0).sum())} 块；j 是内容/位姿失配的深读数，"
              "闭环应显著浅于回放台）")
    if overlap_hits:
        print(f"重叠混接: {overlap_hits} 块（α新={args.seam_overlap_alpha:g}，"
              "偏移=取图→落地流逝拍；首拍 |Δcmd| 即混接后缝口径，与 A 组直比）")
    if starve_windows or starve_ticks:
        print(f"星饿: {starve_windows} 段 / {starve_ticks} 拍 / 共 {starve_total:.1f} s"
              f"（最长 {starve_worst:.1f}s，上限 {args.starve_limit}s 未触发即健康）")
    if dropped_ticks:
        print(f"⚠ 丢拍重锚 {dropped_ticks} 次（tick 服务超 {TICK * 1000:.1f} ms——"
              "SDK 下发/回读超拍等"
              + ("；取图已在 worker，仍持续丢拍=节拍/SDK 侧问题" if args.obs_worker else "；取图在主线程（--no-obs-worker），每块预取拍必丢")
              + "）")
    print(f"最终偏离起始位: {exc:.0f} mrad（护栏 {args.max_excursion * 1000:.0f} mrad）"
          + (" ⛔ 护栏触发过" if aborted else ""))

    # 自然跑完（rounds 消费尽）→ warm 慢速归位到 S1_HOME_POSE（G1 无此步=操作员
    # 手动收尾；S1 内嵌，10-29「一瞬间复位」病的治愈口径）。q/Ctrl-C 走
    # QuitWatcher 立即 os._exit 不经过这里（traj 忠实语义：已下发目标可能仍在
    # 限速执行）；护栏/星饿/非 SUCCESS 停机（aborted）也留在原地供排查。
    if args.do_exec and not aborted:
        print("[warm] 归位 S1_HOME_POSE（@0.15 rad/s 慢速）…")
        s1_warm(robot, "归位")

    WATCH.restore()
    if NAV is not None:
        NAV.resume()
    print("SDK 关闭中...")
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)   # SDK 残留线程，干净退出
