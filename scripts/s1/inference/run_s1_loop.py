#!/usr/bin/env python
"""S1 通用推理执行路线（policy 无关）——run_g1_loop_traj.py 的 S1 镜像 + 适配器解耦。

架构（用户定案：上层配置与代码隔离）：
  run_s1_loop.py           本文件 = S1 SDK 交互 + 循环节拍 + 安全栏 + 遥测（policy 无关）
  policies/<name>.py       policy 专属（引擎加载/观测预处理/动作语义/System 2）全部住适配器
  policies/base.py         PolicyAdapter 接口 + S1Targets（执行器唯一认识的动作表达）
新增 policy = policies/ 加一个文件注册进 ADAPTERS，执行器零改动。

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
    HEAD_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, S1_LIMITS, S1Targets,
    TORSO_JOINTS, gripper_pct_to_width_m, gripper_width_m_to_pct, letterbox224)

AP = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--policy", default="fineart_vla", choices=sorted(ADAPTERS))
AP.add_argument("--ckpt", default=None, help="默认用适配器类属性 CKPT_DIR")
AP.add_argument("--task", default="pick up the object")
AP.add_argument("--rounds", type=int, default=3, help="换块轮数")
AP.add_argument("--ctrl-hz", type=float, default=30.0)
AP.add_argument("--n-action-steps", type=int, default=10,
                help="每块消费步数（水位；默认 10=在役配方）")
AP.add_argument("--delta-max", type=float, default=0.05,
                help="包络护栏：单步相对当前读数的最大关节增量（rad，臂）")
AP.add_argument("--joint-speed", type=float, default=0.2,
                help="set_joint_positions speed_rad_s（SDK 默认 0.2，保守）")
AP.add_argument("--grip-velocity", type=float, default=0.05)
AP.add_argument("--grip-effort", type=float, default=10)
AP.add_argument("--subtask-every", type=int, default=0,
                help="每 N 块生成一次 System 2 子任务（0=关闭；fineart only）")
AP.add_argument("--live-frames", action="store_true",
                help="用真机相机取图（init 相机订阅）；默认合成噪声帧")
AP.add_argument("--exec", dest="do_exec", action="store_true",
                help="发送指令驱动本体（默认 dry-run；需适配器有 S1 映射+交互确认）")
AP.add_argument("--synthetic-seed", type=int, default=8)
AP.add_argument("--out", default=os.path.expanduser("~/holy/evidence/s1_loop"))
AP.add_argument("--list-policies", action="store_true")
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
        print("\n⚠⚠ --exec 将发送关节指令。急停按钮已释放？手臂旁无人/无障碍？"
              "\n⚠ 模式切换/接管瞬间手臂可能失力矩坠落（10-27 教训）——手不离急停。")
        if input("确认继续？(y/N) ").strip().lower() != "y":
            raise SystemExit("未确认，退出。")
        robot.start_controller("all")
        executing = True
        print("✓ start_controller(all)——执行模式")
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
    adapter.load(ckpt, args.task, calib_obs=obs0)
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
                text = adapter.subtask(args.task, obs)
                print(f"  [S2] {text!r}（{(time.perf_counter() - t2) * 1000:.0f} ms）")
            t1 = time.perf_counter()
            chunk = adapter.infer(obs, state_raw)
            infer_ms = (time.perf_counter() - t1) * 1000
            n = max(0, min(args.n_action_steps, len(chunk)))
            sent, clamped = 0, 0
            for k in range(n):
                tg = adapter.to_s1(chunk[k], state_raw)
                if tg is None:
                    if k == 0:
                        print(f"  chunk{r} step0（dry-run）: "
                              f"{np.array2string(chunk[k], precision=3, suppress_small=True)}")
                elif executing:
                    clamped += issue_targets(robot, tg, state_raw)
                    sent += 1
                sleep_to(tr + period * (k + 1))
            rows.append({"round": r, "infer_ms": round(infer_ms, 1),
                         "steps": n, "sent": sent, "clamped": clamped,
                         "text": text})
            print(f"  round {r}: infer {infer_ms:.0f} ms | 消费 {n} 步 | "
                  f"发送 {sent} | 钳位 {clamped}")
    except KeyboardInterrupt:
        print("\n→ Ctrl-C 退出")
    finally:
        report = {"policy": args.policy, "ckpt": ckpt, "task": args.task,
                  "executing": executing, "rounds": rows,
                  "delta_max": args.delta_max, "ctrl_hz": args.ctrl_hz,
                  "n_action_steps": args.n_action_steps,
                  "live_frames": args.live_frames}
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"run_{args.policy}_{int(time.time())}.json"
        f.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"落盘 {f}")
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
    return 0


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


def issue_targets(robot, tg: S1Targets, state_raw: np.ndarray) -> int:
    """包络护栏（限位+delta-max）→ 关节/夹爪指令。返回钳位数。"""
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
        if nm in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:   # delta-max 只限臂（v3 配方）
            cur = float(state_raw[LEFT_ARM_JOINTS.index(nm)]
                        if nm in LEFT_ARM_JOINTS
                        else state_raw[8 + RIGHT_ARM_JOINTS.index(nm)])
            dv = v - cur
            if abs(dv) > args.delta_max:
                v = cur + float(np.clip(dv, -args.delta_max, args.delta_max))
                clamped += 1
        out.append(v)
    st = robot.set_joint_positions(out, joint_names=names, is_blocking=False,
                                   speed_rad_s=args.joint_speed)
    if st is not None and getattr(st, "value", 0) != 0:
        print(f"⚠ set_joint_positions → {st}")
    for side, w in (("left", tg.left_grip_m), ("right", tg.right_grip_m)):
        if w is None:
            continue
        cur_pct = float(state_raw[7] if side == "left" else state_raw[15])
        cur_m = gripper_pct_to_width_m(cur_pct)
        if abs(w - cur_m) > 0.002:
            robot.set_gripper_command(f"{side}_gripper", float(w),
                                      velocity_mps=args.grip_velocity,
                                      effort=args.grip_effort, is_blocking=False)
    return clamped


def sleep_to(deadline: float) -> None:
    """固定 tick 节拍（run_g1_loop_native 同款）；欠账时无 sleep 直接追赶。"""
    dt = deadline - time.perf_counter()
    if dt > 0:
        time.sleep(dt)


if __name__ == "__main__":
    sys.exit(main())
