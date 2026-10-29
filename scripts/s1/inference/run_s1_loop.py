#!/usr/bin/env python
"""S1 通用推理执行路线（policy 无关）——run_g1_loop_traj.py 的 S1 镜像 + 适配器解耦。

架构（用户定案：上层配置与代码隔离）：
  run_s1_loop.py           本文件 = S1 SDK 交互 + 循环节拍 + 安全栏 + 遥测（policy 无关）
  policies/<name>.py       policy 专属（引擎加载/观测预处理/动作语义/System 2）全部住适配器
  policies/base.py         PolicyAdapter 接口 + S1Targets（执行器唯一认识的动作表达）
新增 policy = policies/ 加一个文件注册进 ADAPTERS，执行器零改动。

执行路线（用户定案 2026-10-28：对齐 run_g1_loop_traj 的流式哲学，勿退回
set_joint_positions——官方明示后者不适合流式）：
  lerobot 动作块 → 逐行动作 → 逐帧 set_joint_commands 直发（默认 fr=1 =
  tfs0 立即执行，行节拍=脚本绝对截止墙钟 tick，G1 traj 同款；fr>1=A/B
  插值微流，行内小 batch 累计 tfs）。水位（execute_actions_per_chunk）：
  0=整块行流完（50 行≈1.67s）立即重推理；N=消费 N 行即重推理。
  ⚠ G1 的异步预取（换块即预取/星饿 hold）未带——同步循环下重推理 232ms
  ≪ 块长 1.67s，吞吐损失 ~12% 挂账；夹爪仍走独立 API（G1 --grip-state-cmd
  位）。⚠ tfs×singorix WBC 真机语义未验证——G1 血泪史（tfs0 每点全停
  滞后 2.1×、tfs33.3 stall 判死）上机前先小步复验。

安全设计（三重闸 + 零运动默认）：
  ① 默认 dry-run：全链照跑（取图/推理/映射/打印），不发任何指令
  ② --exec 需 adapter.needs_mapping=True（S1 语义映射存在）——现有两 policy
    （fineart=ALOHA 14 维 / pi05=G1 16 维）语义映射均未定义，--exec 一律拒绝；
    映射随微调线落地后在其适配器里打开
  ③ --exec 交互确认（官方 example5 同款）+ start_controller 显式开启 +
    包络护栏（限位表逐关节钳位 + delta-max 相对当前读数 + 连续违规熔断）

S1 事实源：docs/s1-joint-api-quickref.md；⚠ 2026-10-27 手臂掉落教训——模式切换/
接管瞬间手臂可能失力矩，--exec 前手不离急停。

用法（本机 flash_pyrt311 已含 SDK 3.11 绑定）:
  PYTHONPATH=/data/galbot/lib LD_LIBRARY_PATH=/data/galbot/lib \\
    ~/miniforge3/envs/flash_pyrt311/bin/python run_s1_loop.py --list-policies
  ... run_s1_loop.py --policy fineart_vla --task "pick up the object" --rounds 3
  ... run_s1_loop.py --policy fineart_vla --live-frames --subtask-every 5 --rounds 10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

os.environ.setdefault("FLASH_RT_PI05_ACTION_CHUNK_SIZE", "50")   # 须在 import flash_rt 前定死

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import ADAPTERS, create  # noqa: E402
from policies.base import (  # noqa: E402
    HEAD_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, S1_HOME_POSE, S1_LIMITS,
    S1Targets, TORSO_JOINTS, gripper_pct_to_width_m, gripper_width_m_to_pct,
    letterbox224)

try:   # SDK 绑定（C 扩展，首次 import ~70ms）顶层预热——懒 import 放热路径会把
       # 首调成本记进 tick0 预算顶爆节拍（10-29 突发采样日志 bug 根因，见
       # probes/replay_stream_bench.py）。无 SDK 环境（--list-policies/合成回放）仍可加载。
    from galbot_sdk.s1 import JointCommand, Trajectory, TrajectoryPoint  # noqa: E402
except ImportError:
    JointCommand = Trajectory = TrajectoryPoint = None

AP = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--policy", default="fineart_vla", choices=sorted(ADAPTERS))
AP.add_argument("--ckpt", default=None, help="默认用适配器类属性 CKPT_DIR")
AP.add_argument("--task", "--prompt", dest="prompt", default="pick up the object",
                help="任务提示（System 2 prompt；toml 键=prompt，G1 [lerobot] 同名）")
AP.add_argument("--rounds", type=int, default=3, help="换块轮数")
AP.add_argument("--ctrl-hz", type=float, default=30.0,
                help="行节拍 Hz（S1 扩展键；G1 定盘 30Hz 口径）")
AP.add_argument("--execute-actions-per-chunk", "--n-action-steps",
                dest="execute_actions_per_chunk", type=int, default=0,
                help="水位（同事/lerobot 同名键，G1 toml 同款）：0=整块行流完"
                     "（50 行≈1.67s）立即重推理（新鲜度第一杠杆）；N=消费 N 行"
                     "即重推理。两种都逐帧墙钟节拍直发〔旧名 --n-action-steps〕")
AP.add_argument("--frames-per-row", type=int, default=1,
                help="每行动作的流式帧数（默认 1=不插值纯 30Hz 直发，用户定案；"
                     ">1 启用帧插值，帧距 = ctrl_hz 倒数/帧数）")
AP.add_argument("--rows-per-cmd", type=int, default=1,
                help="每次 set_joint_commands 打包行数 R（命令窗口=R×行周期，"
                     "与行周期解耦）。10-28 真机窗口律：卡顿⇔单点无未来"
                     "（R=1 且 fr=1）或窗口过短（30Hz 单行=33.3ms 仅『还好』）；"
                     "66.7ms 窗全安静、点距细更佳（15Hz fr24 用户评最佳）。"
                     "30Hz 实时等价配方=R=2（窗口 66.7ms）×frames_per_row 12"
                     "（点距≈2.8ms）。1=逐行原路径（默认）")
AP.add_argument("--tfs-zero", dest="tfs_zero", action="store_true",
                help="【实验开关，仅回放台 A/B 用】batch 路径把全部 tfs 清零"
                     "（=G1 式 ASAP 直发形态，10-29 用户提议的对照假设）。"
                     "singorix 文档 §5.1 预测：无时间轴 → 采样器塌缩/立即跳变，"
                     "预期复现或加剧 fr1 卡卡；若安静则时间轴必要性被证伪。"
                     "默认关（时间轴是现役配方的根基，勿在闭环用）")
AP.add_argument("--max-joint-delta-per-step", "--delta-max",
                dest="max_joint_delta_per_step", type=float, default=0.05,
                help="包络护栏：链式限幅 ±rad/步（臂；相对最后已发目标，"
                     "G1 traj 同款参考不重锚；块界从新读数重锚=安全行为。"
                     "10-29 修正：曾用块首读数当基准=整块钳死在 ±0.05 龟速）"
                     "〔旧名 --delta-max〕")
AP.add_argument("--filter-alpha", "--ema-alpha",
                dest="filter_alpha", type=float, default=0.0,
                help="行目标 EMA 低通系数（同事 action_filter.lowpass.alpha 同名；"
                     "0=关，默认关；α=新样本权重。最小版：每块从当前读数重锚，"
                     "G1 的接缝窗 α 调度/链钳/爪列 EMA 未带。真机抖动露出再 A/B）"
                     "〔旧名 --ema-alpha〕")
AP.add_argument("--grip-velocity", type=float, default=0.05)
AP.add_argument("--grip-effort", type=float, default=10)
AP.add_argument("--subtask-every", type=int, default=0,
                help="每 N 块生成一次 System 2 子任务（0=关闭；fineart only）")
AP.add_argument("--live-frames", action="store_true",
                help="用真机相机取图（init 相机订阅）；默认合成噪声帧")
AP.add_argument("--no-home-start", dest="home_start", action="store_false",
                help="--exec 就位段开关（默认开）：开始前慢速就位到任务起始"
                     "位姿 S1_HOME_POSE（47 集首帧中位，base.py；链钳渐进+2点"
                     "时间轴批，回放台同款机制不卡卡）")
AP.add_argument("--no-return-home", dest="return_home", action="store_false",
                help="--exec 归位段开关（默认开）：结束/Ctrl-C 后慢速归位到"
                     "同一位姿（再按一次 Ctrl-C 可跳过归位直接退出）")
AP.add_argument("--home-tol", type=float, default=0.03,
                help="就位/归位收敛判据：臂最大差 rad（默认 0.03）")
AP.add_argument("--home-timeout", type=float, default=30.0,
                help="就位/归位超时 s，超时警告放行（默认 30）")
AP.add_argument("--exec", dest="do_exec", action="store_true",
                help="发送指令驱动本体（默认 dry-run；需适配器有 S1 映射+交互确认）")
AP.add_argument("--synthetic-seed", type=int, default=8)
AP.add_argument("--out", default=os.path.expanduser("~/holy/evidence/s1_loop"))
AP.add_argument("--list-policies", action="store_true")
AP.add_argument("--config", default=os.path.expanduser("~/holy/config/s1.toml"),
                help="toml 档（顶层键=选项 dest 名；CLI 显式传参覆盖 toml；"
                     "文件不存在则全用内置默认。G1 对照=config/g1.toml [lerobot]）")

# toml 档预扫描：在 parse 前把顶层键填成 argparse 默认（CLI 显式传参仍覆盖）；
# 未传 --config 时用默认档路径（文件不存在则静默跳过）
_cfg_path = None
for _i, _a in enumerate(sys.argv):
    if _a == "--config" and _i + 1 < len(sys.argv):
        _cfg_path = sys.argv[_i + 1]
        break
    if _a.startswith("--config="):
        _cfg_path = _a.split("=", 1)[1]
        break
if _cfg_path is None:
    _cfg_path = next((a.default for a in AP._actions if a.dest == "config"), None)
if _cfg_path:
    _cfg_path = os.path.expanduser(str(_cfg_path))
    if os.path.exists(_cfg_path):
        import tomllib
        _cfg = tomllib.loads(pathlib.Path(_cfg_path).read_text())
        _known = {a.dest for a in AP._actions}
        AP.set_defaults(**{k: v for k, v in _cfg.items() if k in _known})
args = AP.parse_args()

_ADAPTER = [None]      # 当前已加载适配器（grab_obs 经此取 camera_map）
_args = [args]         # issue_targets/sleep_to 经此读 CLI 参数


def main() -> int:
    cls = ADAPTERS[args.policy]
    if args.list_policies:
        for n, c in sorted(ADAPTERS.items()):
            print(f"{n:12s} action_dim={c.action_dim:2d} state_dim={c.state_dim:2d} "
                  f"exec={'可' if c.needs_mapping else '否(dry-run only)'} | {c.notes}")
        return 0
    if args.do_exec and not cls.needs_mapping:
        raise SystemExit(
            f"policy {args.policy!r} 无 S1 语义映射（{cls.notes}）"
            f"\n→ 拒绝 --exec（安全闸②）。映射随微调线在适配器里落地后开启。")

    # ── SDK ──
    from galbot_sdk.s1 import GalbotRobot, SensorType

    robot = GalbotRobot()
    sensors = ({getattr(SensorType, c) for c in cls.camera_map.values()}
               if args.live_frames else set())
    if not robot.init(sensors):
        raise SystemExit("robot.init 失败（服务在跑吗？）")
    print(f"✓ SDK init（sensors={len(sensors)}）")
    time.sleep(3)

    # 显式名字读 17 body 关节（group 模式配对全错，G1 教训；返回顺序与本次配对）
    body_names = robot.get_joint_names(only_active_joint=True,
                                       joint_groups=["left_arm", "right_arm",
                                                     "torso", "head"])
    print(f"body 关节 {len(body_names)}: {body_names}")

    executing = False
    if args.do_exec:
        blind = "" if args.live_frames else (
            "\n⚠⚠ 未带 --live-frames：本次用合成噪声图（盲跑）——模型看不到真实场景，"
            "动起来无任务意义，仅验链路！")
        print(f"\n⚠⚠ --exec 将发送关节指令。急停按钮已释放？手臂旁无人/无障碍？"
              f"\n⚠ 模式切换/接管瞬间手臂可能失力矩坠落（10-27 教训）——手不离急停。{blind}")
        if input("确认继续？(y/N) ").strip().lower() != "y":
            raise SystemExit("未确认，退出。")
        robot.start_controller("all")
        executing = True
        print("✓ start_controller(all)——执行模式")
        if args.home_start:
            home_to(robot, body_names, home_pose_targets(), "就位")
    else:
        print("（dry-run：不发指令，动作目标只打印）")

    # ── 引擎（单载；grab_obs 经 _ADAPTER 取 camera_map）──
    adapter = create(args.policy)
    _ADAPTER[0] = adapter
    ckpt = args.ckpt or getattr(cls, "CKPT_DIR", None)
    if not ckpt:
        raise SystemExit(f"policy {args.policy} 未定义 CKPT_DIR，须 --ckpt 传入")
    obs0 = grab_obs(robot) if args.live_frames else synthetic_obs(0)
    t0 = time.time()
    adapter.load(ckpt, args.prompt, calib_obs=obs0)
    print(f"✓ 引擎加载 {args.policy} ← {ckpt}（{time.time() - t0:.1f}s）")

    # ── 循环：取观测 → 读状态 →（可选 S2）→ 推理 → 按 ctrl_hz 消费 n 步 ──
    period = 1.0 / args.ctrl_hz
    rows = []
    try:
        for r in range(args.rounds):
            tr = time.perf_counter()
            obs = obs0 if r == 0 else (grab_obs(robot) if args.live_frames
                                       else synthetic_obs(r))
            state_raw, _extras = read_s1_state(robot, body_names)
            text = None
            if args.subtask_every and r % args.subtask_every == 0:
                t2 = time.perf_counter()
                text = adapter.subtask(args.prompt, obs)
                print(f"  [S2] {text!r}（{(time.perf_counter() - t2) * 1000:.0f} ms）")
            t1 = time.perf_counter()
            chunk = adapter.infer(obs, state_raw)
            infer_ms = (time.perf_counter() - t1) * 1000
            # 水位语义（同事/lerobot 同名键同款；发令=G1 traj 同款逐帧墙钟流）：
            # 0=整块行流完立即重推理；N=消费 N 行即重推理。两种都按绝对截止
            # tick 直发（rows_per_cmd 控打包粒度，节拍权在脚本，不整块打包给控制器）。
            full_chunk = args.execute_actions_per_chunk <= 0
            n = (len(chunk) if full_chunk
                 else max(0, min(args.execute_actions_per_chunk, len(chunk))))
            sent, clamped, frames = 0, 0, 0
            R = max(1, args.rows_per_cmd)
            chain = state_to_prev(state_raw)     # 流插值链基准 + delta-max 钳位
                                                 # 基准 = 最后已发（块界从新读数
                                                 # 重启=重锚，安全行为；G1 同款）
            ema = dict(chain)                    # EMA 状态（每块从读数重锚，最小版）
            pending = []                         # 已规划待打包的行 (names, tgt)
            for k in range(n):
                tg = adapter.to_s1(chunk[k], state_raw)
                if tg is None:
                    if k == 0:
                        print(f"  chunk{r} step0（dry-run）: "
                              f"{np.array2string(chunk[k], precision=3, suppress_small=True)}")
                else:
                    tg = ema_row(tg, ema, args.filter_alpha)
                    names, tgt, cl = plan_row(tg, chain)
                    clamped += cl
                    pending.append((names, tgt))
                    if executing:
                        send_grippers(robot, tg, state_raw)
                    else:
                        for nm, v in zip(names, tgt):
                            chain[nm] = v       # dry-run：链随规划走（无下发）
                if len(pending) >= R or (k == n - 1 and pending):
                    if executing:
                        sent += 1
                        frames += stream_rows(robot, pending[0][0],
                                              [t for _, t in pending], chain)
                    for names_i, tgt_i in pending:
                        for nm, v in zip(names_i, tgt_i):
                            chain[nm] = v       # 批末行 = 新链基准（最后已发）
                    pending.clear()
                sleep_to(tr + period * (k + 1))  # 绝对截止节拍（G1 同款；欠账不追发）
            rows.append({"round": r, "infer_ms": round(infer_ms, 1),
                         "steps": n, "sent": sent, "frames": frames,
                         "clamped": clamped, "text": text})
            print(f"  round {r}: infer {infer_ms:.0f} ms | 消费 {n} 步 | "
                  f"流式 {frames} 帧 | 钳位 {clamped}")
    except KeyboardInterrupt:
        print("\n→ Ctrl-C 退出")
    finally:
        report = {"policy": args.policy, "ckpt": ckpt, "prompt": args.prompt,
                  "executing": executing, "rounds": rows,
                  "max_joint_delta_per_step": args.max_joint_delta_per_step,
                  "ctrl_hz": args.ctrl_hz,
                  "execute_actions_per_chunk": args.execute_actions_per_chunk,
                  "filter_alpha": args.filter_alpha,
                  "live_frames": args.live_frames}
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"run_{args.policy}_{int(time.time())}.json"
        f.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"落盘 {f}")
        if executing and args.return_home:
            try:
                home_to(robot, body_names, home_pose_targets(), "归位")
            except KeyboardInterrupt:
                print("（跳过归位，直接退出）")
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
    return 0


# ── 就位 / 归位（--exec 专用；dry-run 零运动不经过） ──

def home_pose_targets() -> S1Targets:
    """S1_HOME_POSE（base.py，47 集首帧中位）→ S1Targets（爪=全开宽度 m）。"""
    return S1Targets(
        left_arm=np.array(S1_HOME_POSE["left_arm"], dtype=np.float32),
        right_arm=np.array(S1_HOME_POSE["right_arm"], dtype=np.float32),
        left_grip_m=gripper_pct_to_width_m(S1_HOME_POSE["grip_pct"]),
        right_grip_m=gripper_pct_to_width_m(S1_HOME_POSE["grip_pct"]),
        torso=float(S1_HOME_POSE["torso"]),
        head=np.array(S1_HOME_POSE["head"], dtype=np.float32))


def home_to(robot, body_names, goal: S1Targets, tag: str) -> bool:
    """慢速就位/归位（回放台 home_to 同款机制，真机已验不卡卡）：
    每 ctrl_hz 拍重读臂态 → plan_row 链钳（fresh-read 字典=真值基准）→
    攒 2 行 → 2 点时间轴批 tfs=period×(i+1)（与 R/fr 档解耦——就位段自己
    不卡卡）。爪只发一次（全开）；torso/head 随批慢跟不阻塞收敛判据。
    收敛 --home-tol（臂最大差 rad）/ 超时 --home-timeout 警告放行。"""
    args = _args[0]
    period = 1.0 / args.ctrl_hz
    state_raw, _extras = read_s1_state(robot, body_names)
    send_grippers(robot, goal, state_raw)
    deadline = time.perf_counter() + args.home_timeout
    pending, t_print, ok, err = [], 0.0, False, float("inf")
    try:
        while True:
            tr = time.perf_counter()
            arm_vals = robot.get_joint_positions(
                joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)  # 请求序配对（正确系）
            for i in range(7):
                state_raw[i] = float(arm_vals[i])        # 左臂 ← state_raw[0:7]
                state_raw[8 + i] = float(arm_vals[7 + i])  # 右臂 ← [8:15]
            err = max(
                max(abs(float(state_raw[i]) - float(v))
                    for i, v in enumerate(goal.left_arm)),
                max(abs(float(state_raw[8 + i]) - float(v))
                    for i, v in enumerate(goal.right_arm)))
            if err < args.home_tol:
                ok = True
                break
            if tr > deadline:
                print(f"⚠ {tag}超时 {args.home_timeout:.0f}s"
                      f"（臂差 {err * 1000:.0f} mrad）——放行，链钳兜底")
                break
            if tr - t_print > 1.0:
                t_print = tr
                print(f"  {tag}… 臂最大差 {err * 1000:.0f} mrad", flush=True)
            names, tgt, _cl = plan_row(goal, state_to_prev(state_raw))
            pending.append((names, tgt))
            if len(pending) >= 2:
                traj = Trajectory()
                traj.joint_names = names
                pts = []
                for i, (_, tv) in enumerate(pending):
                    p = TrajectoryPoint()
                    p.time_from_start_second = round(period * (i + 1), 6)
                    vec = []
                    for q in range(len(names)):
                        c = JointCommand()
                        c.position = tv[q]
                        c.velocity = c.acceleration = c.effort = 0.0
                        vec.append(c)
                    p.joint_command_vec = vec   # ⚠ getter 返副本：整体赋值
                    pts.append(p)
                traj.points = pts
                st = robot.set_joint_commands_batch(traj)
                if st is not None and getattr(st, "value", 0) != 0:
                    print(f"⚠ {tag} batch → {st}")
                pending = []
            sleep_to(tr + period)
        print(f"✓ {tag}{'收敛' if ok else '放行'}（臂差 {err * 1000:.0f} mrad）")
        return ok
    except KeyboardInterrupt:
        print(f"\n→ {tag}被 Ctrl-C 打断")
        raise


# ── 观测 / 状态 / 指令（S1 通用件，policy 无关） ──

def synthetic_obs(seed=None) -> dict:
    s = args.synthetic_seed + (0 if seed is None else seed)
    return {k: np.random.default_rng(s + i).integers(0, 256, (224, 224, 3),
                                                     dtype=np.uint8)
            for i, k in enumerate(_ADAPTER[0].views)}


def grab_obs(robot) -> dict:
    """真机取图：get_rgb_data(JPEG) → BGR 解码 → letterbox224 → RGB（适配器 camera_map）。"""
    import cv2
    from galbot_sdk.s1 import RgbOutputFormat, SensorType

    obs = {}
    for key, cam in _ADAPTER[0].camera_map.items():
        d = robot.get_rgb_data(getattr(SensorType, cam), RgbOutputFormat.JPEG, True)
        assert d and d.get("data"), f"{cam} 无图"
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        obs[key] = np.ascontiguousarray(
            cv2.cvtColor(letterbox224(img), cv2.COLOR_BGR2RGB))
    return obs


def read_s1_state(robot, body_names):
    """19 维原始状态（quickref §3 布局；臂/头=rad、夹爪=%、torso=升降高度 m）+ 夹爪宽度米。显式名读，勿 group 模式。"""
    vals = robot.get_joint_positions(joint_names=body_names)
    pos = dict(zip(body_names, [float(v) for v in vals]))
    grip_m = {}
    for side in ("left", "right"):
        st = robot.get_gripper_state(f"{side}_gripper")
        grip_m[side] = float(getattr(st, "width", 0.0) or 0.0)
    state_raw = np.array(
        [pos[n] for n in LEFT_ARM_JOINTS]
        + [gripper_width_m_to_pct(grip_m["left"])]
        + [pos[n] for n in RIGHT_ARM_JOINTS]
        + [gripper_width_m_to_pct(grip_m["right"])]
        + [pos["torso_lift_joint1"], pos["head_joint1"], pos["head_joint2"]],
        dtype=np.float32)
    return state_raw, {"grip_m": grip_m}


def state_to_prev(state_raw: np.ndarray) -> dict[str, float]:
    """19 维读数 → 关节名→当前值表（流式行插值链起点；臂/头=rad，torso=m）。"""
    return {**{nm: float(v) for nm, v in zip(LEFT_ARM_JOINTS, state_raw[0:7])},
            **{nm: float(v) for nm, v in zip(RIGHT_ARM_JOINTS, state_raw[8:15])},
            TORSO_JOINTS[0]: float(state_raw[16]),
            HEAD_JOINTS[0]: float(state_raw[17]),
            HEAD_JOINTS[1]: float(state_raw[18])}


def ema_row(tg: S1Targets, ema: dict[str, float], alpha: float) -> S1Targets:
    """行目标 EMA 低通（G1 --ema-alpha 同语义：α=新样本权重，0=关）。
    最小版：仅臂/torso/head，每块由调用方从读数重锚；安全钳位在其后
    （plan_row），滤波值不会绕过限位/delta-max。爪不滤（无翻转证据）。"""
    if alpha <= 0:
        return tg
    import dataclasses

    def filt(arr, jnames):
        out = np.asarray(arr, dtype=np.float64).copy()
        for i, nm in enumerate(jnames):
            v = alpha * float(out[i]) + (1.0 - alpha) * ema.get(nm, float(out[i]))
            out[i] = v
            ema[nm] = v
        return out

    f = dataclasses.replace(tg)
    if tg.left_arm is not None:
        f.left_arm = filt(tg.left_arm, LEFT_ARM_JOINTS)
    if tg.right_arm is not None:
        f.right_arm = filt(tg.right_arm, RIGHT_ARM_JOINTS)
    if tg.torso is not None:
        f.torso = float(filt([tg.torso], TORSO_JOINTS)[0])
    if tg.head is not None:
        f.head = filt(tg.head, HEAD_JOINTS)
    return f


def plan_row(tg: S1Targets, prev) -> tuple[list[str], list[float], int]:
    """包络护栏（限位表钳位 + delta-max 链式限幅，臂 only）。
    prev = 钳位基准 name→值映射：执行器传演进链 chain（最后已发/已规划，
    G1 run_g1_loop_traj:1830 同款「参考不重锚」——曾用块首读数当基准会让
    整块 50 行全钳在块首 ±delta-max 内=臂速 0.05 rad/块 龟速，10-29 对质
    G1 修正）；回放台传每 tick 重读的 state_raw 转字典（真值链钳）。
    → (names, targets, clamped)。爪不在此列（独立 API）。"""
    args = _args[0]
    names, tgt = [], []
    if tg.left_arm is not None:
        names += LEFT_ARM_JOINTS
        tgt += [float(v) for v in tg.left_arm]
    if tg.right_arm is not None:
        names += RIGHT_ARM_JOINTS
        tgt += [float(v) for v in tg.right_arm]
    if tg.torso is not None:
        names += TORSO_JOINTS
        tgt += [float(tg.torso)]
    if tg.head is not None:
        names += HEAD_JOINTS
        tgt += [float(v) for v in tg.head]
    out, clamped = [], 0
    for nm, v in zip(names, tgt):
        lo, hi = S1_LIMITS[nm]
        v = min(max(v, lo), hi)
        if nm in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:   # delta-max 只限臂（G1 v3 配方）
            cur = prev.get(nm) if hasattr(prev, "get") else None
            if cur is not None:
                dv = v - float(cur)
                if abs(dv) > args.max_joint_delta_per_step:
                    v = float(cur) + float(np.clip(
                        dv, -args.max_joint_delta_per_step,
                        args.max_joint_delta_per_step))
                    clamped += 1
        out.append(v)
    return names, out, clamped


def stream_rows(robot, names, rows_tgt, prev: dict[str, float]) -> int:
    """一组行目标→一次下发（跨行 batch，窗口=R×行周期，与行周期解耦）。
    10-28 真机窗口律：卡顿⇔控制器看不到未来（单点 tfs=0）或窗口过短
    （30Hz 单行=33.3ms）；66.7ms 窗全安静、点距细更佳。行 j 段从上一行
    末目标出发 fr 帧线性走到本行目标，tfs 全批累计；末点恰在窗口末。
    R=1 且 fr=1 退化为单点 tfs=0（原路径逐位不变）。返回帧数。"""
    args = _args[0]
    fr = max(1, args.frames_per_row)   # JointCommand/Trajectory 已顶层预热（勿懒 import 进热路径）

    def _cmds(vals):
        out = []
        for tv in vals:
            c = JointCommand()
            c.position = tv
            c.velocity = 0.0
            c.acceleration = 0.0
            c.effort = 0.0
            out.append(c)
        return out

    if len(rows_tgt) == 1 and fr == 1:
        st = robot.set_joint_commands(_cmds(rows_tgt[0]), joint_names=names,
                                      time_from_start_s=0.0)
        if st is not None and getattr(st, "value", 0) != 0:
            print(f"⚠ set_joint_commands → {st}")
        return 1
    dt = (1.0 / args.ctrl_hz) / fr
    traj = Trajectory()
    traj.joint_names = names
    pts = []
    base = [float(prev.get(nm, tv)) for nm, tv in zip(names, rows_tgt[0])]
    idx = 0
    for tgt in rows_tgt:
        end = [float(v) for v in tgt]
        for i in range(1, fr + 1):
            a = i / fr
            p = TrajectoryPoint()
            p.time_from_start_second = 0.0 if args.tfs_zero else round(
                dt * (idx + i), 6)  # 全批累计时刻（--tfs-zero=全零，ASAP 实验对照）
            vec = []
            for q in range(len(names)):
                c = JointCommand()
                c.position = base[q] + (end[q] - base[q]) * a
                c.velocity = 0.0
                c.acceleration = 0.0
                c.effort = 0.0
                vec.append(c)
            p.joint_command_vec = vec
            pts.append(p)
        base = end
        idx += fr
    traj.points = pts
    st = robot.set_joint_commands_batch(traj)
    if st is not None and getattr(st, "value", 0) != 0:
        print(f"⚠ set_joint_commands_batch → {st}")
    return len(pts)


def stream_row(robot, names, tgt, prev: dict[str, float]) -> int:
    """单行→帧流下发（兼容入口；=stream_rows 单行版）。"""
    return stream_rows(robot, names, [tgt], prev)


def send_grippers(robot, tg: S1Targets, state_raw: np.ndarray) -> None:
    """夹爪独立 API（不在 batch 轨迹里；G1 --grip-state-cmd 同位）。"""
    args = _args[0]
    for side, w in (("left", tg.left_grip_m), ("right", tg.right_grip_m)):
        if w is None:
            continue
        cur_pct = float(state_raw[7] if side == "left" else state_raw[15])
        cur_m = gripper_pct_to_width_m(cur_pct)
        if abs(w - cur_m) > 0.002:
            robot.set_gripper_command(f"{side}_gripper", float(w),
                                      velocity_mps=args.grip_velocity,
                                      effort=args.grip_effort, is_blocking=False)


def sleep_to(deadline: float) -> None:
    """固定 tick 节拍（run_g1_loop_native 同款）；欠账时无 sleep 直接追赶。"""
    dt = deadline - time.perf_counter()
    if dt > 0:
        time.sleep(dt)


if __name__ == "__main__":
    sys.exit(main())
