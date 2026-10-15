#!/usr/bin/env python
"""G1 双臂闭环 receding-horizon：推理→执行→再观测 连续循环（真机唯一闭环入口）。

【B 侧脚本，2026-10-15】A/B 对比：A=scripts/inference/run_g1_loop.py（原样），
本脚本唯一差异=头相机观测追加「训练同款裁剪放大」（head_training_crop，
见 letterbox224 下方块）——sft_1012 数据（colleague clean_ready_224）训练
视频头图经 image_preprocessor.py 的 head_training_crop 处理（letterbox224
后裁 [126:214, 52:172] → Lanczos4 拉回 164×224 ≈1.87× 放大 → 上下补 30px
黑条），推理须复现同变换否则 train/serve 图像分布失配。配对权重
models/pi05_g1_sft1012_dropout20_deploy（stats 与 onlypick 逐维逐位一致，
10-15 核对）。--no-head-crop 可关（同权重自检 crop 差异）。其余逐行同 A。
实验记录见本目录 README.md。

推理迟到（本轮配额步消费完而下块未就绪）时的动作策略 --late-action 二态：
  hold（默认）：hold 最后目标单调收敛（几何逼近永不过冲），轮询早退（推理
     一落地立即换块，不等满窗），夹爪行不变被 chg 阈值自然抑制（不翻摆）。
     调度约定参照 Physical Intelligence "Real-Time Chunking" + lerobot
     src/lerobot/policies/rtc/（Apache-2.0）的执行侧（extension/等待）；
     去噪引导层（denoise_step 前缀引导）未搬——需 per-ODE-step 钩子 +
     autograd 穿注意力，与 CUDA graph 捕获冲突，且本链路 delay≈1 轮收益
     有限（2026-10-04 评估，详见 docs/问题-解决.md）
  stale：消费配额后继续喂旧 chunk 深尾步"续航"——深尾是衰减/反转的计划
     垃圾，pace 破水位时臂往复+夹爪翻摆（9/30 sweep 0.40/0.38 破位实录；
     原独立 A 脚本行为，存档于 legacy/run_g1_loop_stale.py，A/B 对照用）
同参数稳态两态行为一致（零 hold）；pace 压到推理水位下时 hold 平滑退化
（hold 窗=纯墙钟）而 stale 出深尾垃圾——主对照实验。
参数命名对 lerobot 术语的对照（n_action_steps/chunk_size 等）：
docs/lerobot-alignment.md。

⚠ 加 --exec 会连续真实驱动双臂！安全设计：
  1. 只动双臂 14 关节；腿/头维度永不下发；夹爪仅 --grip 显式开启时下发
     （0~100% → manifest 标定宽度，变化超 --grip-chg 才发，非阻塞不等反馈）
  2. 每条指令目标 = 当前读数 ± steps-per-command×--delta-max 限幅
     （config/g1.toml [loop] 现值 25×0.3 rad；逐值默认以 config 为准）；
     速度 = 配速 clip(导程÷(--pace-div×--pace), 0.02, --speed)——每窗按比例
     走导程（pace-div=2 半程：臂恒在途永不到点，被下一轮指令滑行重定向
     ；1=全程臂速翻倍）（2026-09-24 v4，与回放同款，治指令边界换向抖动；
     闭环每轮从真实状态重规划，滞后自校正）
  3. 漂移护栏：任一关节偏离起始位超 --max-excursion（config 现值 3.0 rad）
     → 立即停止循环（防模型单向漂移拖走机械臂）
  4. q 键即时退出（后台监听线程，SDK 阻塞中也可退）；物理急停第一优先级
  5. 执行前回车确认

每轮遥测：推理 ms / 取图 ms / 每步执行 ms / 回读跟踪误差 mrad /
相邻步指令增量（抖动代理）/ 汇总分位数（LatencyTracker，lerobot 同款 API）。
流水线：下一块推理在本轮首条指令执行期间后台完成（predict-only 入后台，
SDK 调用全留主线程）。可选坡升（--hold-ramp N，执行侧 crossfade）：hold
连续 ≥2 窗（臂真减速过）后的 N 轮臂速限幅 0.5→1.0 线性恢复，柔化再起步；
1 窗轮询早退的 hold 不触发（臂未减速，坡升纯属慢性拖慢）。零推理成本。

二相（--phase2-prompt，2026-10-14，需 --post-gate warm-move）：
【单模型档，现默认】只给 --phase2-prompt——0930 部署权重实为 pick+place
4 任务合并训练（train_config wandb notes「4-task merged pick/place
dataset」实锤；目录名 only_pick 只是部署命名骗人），place 无需换权重：
折叠成功后同引擎只换 prompt+相位状态（grip-state-cmd 关/HOME 重基准/
计数清零/place 轮从 0 重计），无第二构造载入（省 ~26s 与其 +45s 预算）。
【双构造档，--dual-load/--phase2-ckpt 显式选，A/B 回退用】place ckpt 开局
与 pick 一起载入+建引擎（v2 开局同载；v1 藏折叠窗口 26.0s>16.8s 漏出
~9s 已废），折叠后换 model/norm_stats/prompt/包络盒（place ckpt）。
两档共同：HOME 重基准/计数清零/直到编排器释放门注入 q（rc=2=任务完成）；
place 轮 grip-state-cmd 自动关闭（持物签名会把释放锁死）；导航栈保持
醒来（折叠横移已 resume），与独立 place 段（--nav-suspend）差 ~55ms/轮。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/run_g1_loop.py \
      [--ckpt ~/holy/models/pi05_g1_onlypick_deploy] [--exec] \
      [--rounds 10] [--n-action-steps 3] [--delta-max 0.05] \
      [--speed 0.15] [--max-excursion 0.25] [--prompt "..."]
"""
import argparse
import atexit
from collections import deque
import functools
import g1_config  # noqa: E402  同目录共享配置（CLI > config/g1.toml > 内置默认）
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "g1"))
import g1_warm_core  # noqa: E402  warm 复位共享核心（--post-gate 折叠段与 warmup 共用）

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--prompt", default=None, help="任务指令（默认 config [run].prompt，须用训练原句）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实连续驱动双臂（默认干跑只打印首轮计划；不进配置，仅 CLI）")
ap.add_argument("--rounds", type=int, default=None, help="推理→执行循环轮数（config [loop].rounds）")
ap.add_argument("--no-head-crop", dest="head_crop", action="store_false",
                help="关闭头图训练裁剪放大（B 侧默认开；同权重自检 crop "
                     "on/off 差异用；详见文件头 B 侧说明）")
ap.add_argument("--n-action-steps", type=int, default=None,
                help="每轮执行 chunk 前 K 步再重观测（≤ --chunk-size；"
                     "config [loop].n_action_steps；lerobot 同名术语，"
                     "旧名 --steps-per-round）")
ap.add_argument("--chunk-size", type=int, default=50,
                help="chunk 长度（lerobot chunk_size，旧名 --horizon；勿与 lerobot "
                     "RTC 的 execution_horizon 混淆），默认 50=训练原生长度"
                     "（2026-09-28 实锤：延迟几乎不变，质量甜点区≈前 15-20 步，"
                     "配合 --n-action-steps 消费；env FLASH_RT_PI05_ACTION_CHUNK_SIZE "
                     "自动设）。10=旧部署切片，已弃用")
ap.add_argument("--steps-per-command", type=int, default=None,
                help="合步：一条 SDK 指令跨 K 个 chunk 步（1=逐步；"
                     "3=整轮一条平滑轮廓，起停次数 1/3，实测提速只会加剧"
                     "每点全停的冲击，减停顿才是平滑正解；"
                     "config [loop].steps_per_command，旧名 steps_per_cmd）")
ap.add_argument("--delta-max", type=float, default=None, help="每步限幅 rad（config [loop].delta_max）")
ap.add_argument("--speed", type=float, default=None, help="关节速度上限 rad/s（config [loop].speed）")
ap.add_argument("--pace", type=float, default=None,
                help="节拍窗口 s：速度=导程÷(pace-div×窗口)，每窗按比例走导程"
                     "（须 ≥ 推理耗时+取图，否则退化为续航频发；config [loop].pace）")
ap.add_argument("--pace-div", type=float, default=None,
                help="配速系数：2=半程配速（默认，v4 同款，臂恒在途永不到点）；"
                     "1=全程（每窗走完导程，臂速翻倍，窗口末到点速度过零）；"
                     "<1 无意义（提前到点干等）；config [loop].pace_div")
ap.add_argument("--near-div", type=float, default=None,
                help="近距阻尼：导程 < --near-gap 时改用该配速系数——单调几何收敛"
                     "（永不过冲）+ 把模型放置位的犹豫摆幅每窗滤掉一半，治悬停来回摆；"
                     "0=关闭（默认）；config [loop].near_div")
ap.add_argument("--near-gap", type=float, default=None,
                help="近距阻尼触发导程 rad（默认 0.06=60 mrad）；config [loop].near_gap")
ap.add_argument("--near-bypass", type=int, default=None,
                help="抓取事件（任一爪开→合穿越）后近距阻尼旁路轮数：返回/后撤段"
                     "强制全速（ep0 后撤窗 ~4s≈11 轮，15 留余量），到期自动交还；"
                     "0=关闭；config [loop].near_bypass")
ap.add_argument("--near-n", type=int, default=None,
                help="近窄轮：近阻尼触发（--near-gap 命中）时本轮配额与指令宽度"
                     "收缩为 N 步（spc 联动=min(spc,N)，与 --near-div 同窗生效）"
                     "——每轮吃最新观测重规划的短导程，目标链贴臂无滞后税，"
                     "代价=时间线变慢 ~n/N；0=只动配速不动步宽；config [loop].near_n")
ap.add_argument("--near-exit", type=float, default=None,
                help="近态退出阈值 rad（滞回上阈：进态=--near-gap，退态=本值；"
                     "默认 0=2×near_gap）。单阈值时增量在阈值两侧的接近段会逐轮"
                     "窄↔宽翻转=速度方波往复摆动（10-13 实录后新增）；"
                     "config [loop].near_exit")
ap.add_argument("--cache-frames", type=int, default=None,
                help="K/V 时序复用周期：1=每帧全量（默认，无损）；2=全量/仅解码交替"
                     "（中间帧视觉前缀复用上一帧 K/V，推理提速但隔帧陈旧——本机节拍下"
                     "陈旧≈一个执行窗，接触相位慎用）；config [loop].cache_frames")
ap.add_argument("--max-excursion", type=float, default=None,
                help="偏离起始位护栏 rad（任一关节超限即停；config [loop].max_excursion）")
ap.add_argument("--settle", action="store_true", default=None,
                help="步进-停走（阻塞等待到位，旧行为）；默认追踪式：误差收窄即发下一目标（config [loop].settle）")
ap.add_argument("--settle-frac", type=float, default=None,
                help="追踪式换目标阈值：剩余误差 < frac×delta-max 即发下一步（config [loop].settle_frac）")
ap.add_argument("--switch-dist", type=float, default=None,
                help="提前换目标阈值 rad：剩余误差收到该值即重定向，滑行中转向、"
                     "速度不过零，消指令边界停顿；0=旧语义（config [loop].switch_dist）")
ap.add_argument("--chunk-mode", choices=("track", "settle", "traj"), default=None,
                help="步进引擎：track=追踪单步 / settle=停走 / traj=整块轨迹流"
                     "（PVT 原生，最平滑；--n-action-steps 不适用，整 chunk 一次发；config [loop].chunk_mode）")
ap.add_argument("--traj-dt", type=float, default=None,
                help="traj 模式轨迹点周期 s（0.033=采集原速 30fps；默认 0.1=3 倍慢放；config [loop].traj_dt）")
ap.add_argument("--grip", action="store_true", default=None,
                help="启用夹爪下发（dim7/dim15 0~100%% → manifest 标定宽度；config [gripper].enabled）")
ap.add_argument("--no-grip", action="store_true",
                help="显式关闭夹爪下发（覆盖 config 的 gripper.enabled=true）")
ap.add_argument("--grip-speed", type=float, default=None, help="夹爪速度 m/s（config [gripper].speed）")
ap.add_argument("--grip-effort", type=float, default=None, help="夹爪力矩 N（config [gripper].effort）")
ap.add_argument("--grip-chg", type=float, default=None,
                help="夹爪下发变化阈值 %%（config [gripper].chg）")
ap.add_argument("--config", default=g1_config.DEFAULT_PATH,
                help="配置文件路径（优先级 CLI > config > 内置默认）")
ap.add_argument("--log-file", default=None,
                help="日志文件路径（默认 <仓库>/logs/loop_<时间戳>.log；终端照常显示，"
                     "全文同步写文件，判读免复制）")
ap.add_argument("--late-action", choices=("hold", "stale"), default="hold",
                help="推理迟到（配额步消费完、下块未就绪）时的动作策略："
                     "hold=保持最后目标单调收敛等推理（默认，出处见文件头）；"
                     "stale=旧 chunk 深尾续航（原 A 脚本行为，A/B 对照用）")
ap.add_argument("--hold-max", type=int, default=6,
                help="hold 连续窗数上限：超过判定推理挂死，中止循环（默认 6 窗）")
ap.add_argument("--hold-ramp", type=int, default=0,
                help="坡升轮数 N：hold 连续 ≥2 窗（臂真减速过）后的 N 轮臂速限幅 "
                     "0.5→1.0 线性恢复，柔化再起步（执行侧 crossfade≈lerobot "
                     "blend 思路，零推理成本）。"
                     "1 窗早退的 hold 不触发——臂没减速，坡升纯属慢性拖慢。0=关闭（默认）")
ap.add_argument("--grip-state-cmd", action="store_true",
                help="state 夹爪维(dim7/15)改喂【指令值】而非 SDK 回读：回读滞后 "
                     "~6.3-8s 而 pick 关键窗口 ~7s——喂回读=模型全程看到冻结在起始值"
                     "的夹爪状态（数据集 state 跟随指令仅 ~0.3-1s，10-05 判读）。"
                     "10-05 pick 四跑 R 爪全程 0%% 即此病。需 --grip；默认关=喂回读"
                     "（place 线一直这么跑的）")
ap.add_argument("--envelope-margin", type=float, default=0.15,
                help="关节包络护栏余量，rad（默认 0.15；旧名 --env-margin）。"
                     "包络=<ckpt>/joint_envelope.json"
                     "（任务数据集臂关节逐维 [min,max]）：指令越界=模型在幻想训练分布"
                     "里不存在的构型——往桌下伸/顶桌即此类（刚性模式压桌→fault）")
ap.add_argument("--envelope-abort-n", type=int, default=5,
                help="连续 N 轮指令被包络截断即停循环（默认 5；旧名 --env-abort-n）："
                     "持续越界=感知漂移，继续跑只会反复撞桌")
ap.add_argument("--no-envelope-guard", action="store_true",
                help="关闭关节包络护栏（排障对照用；旧名 --no-env-guard）")
ap.add_argument("--nav-suspend", action="store_true",
                help="闭环期间 SIGSTOP 搁置导航栈 7 进程（localization/fusion/"
                     "navigation_plan/vtn/surround/swallows/lidar_capture）："
                     "2026-10-07 实测推理 375.3→320.5ms（−54.8ms）、GR3D 50→7。"
                     "退出自动 CONT（含 kill -9，守护网兜底）。"
                     "⚠ 只用于臂上任务+底盘不动窗口：冻结期 localization 位姿停更，"
                     "底盘先恢复导航。默认关")
ap.add_argument("--post-gate", choices=["none", "warm", "warm-move"],
                default="none",
                help="门触发后折叠（10-13）：编排器注入 'r' → 本轮止步 → "
                     "同会话执行 reset（摆臂到 --fold-pose-file 位姿；warm-move "
                     "档再加 GalbotNavigation 横移 --then-move）后退出 rc=0，"
                     "省 warmup 子进程 ~4-5s。none=q 即退（原行为，rc=2）。"
                     "折叠期间 q 仍可急停（rc=4=折叠中止）")
ap.add_argument("--fold-ckpt", default=None,
                help="折叠摆臂的目标位姿/norm_stats/夹爪标定来源目录（=place ckpt）")
ap.add_argument("--fold-pose-file", default=None,
                help="折叠摆臂目标位姿 json（23 维 start_pose，place 起始）")
ap.add_argument("--fold-speed", type=float, default=0.8,
                help="折叠摆臂臂/头速度 rad/s（=编排器 --reset-speed）")
ap.add_argument("--then-move", nargs=3, type=float, default=None,
                metavar=("X", "Y", "YAW_DEG"),
                help="--post-gate warm-move 的横移（base_link 系 x前/y左，"
                     "deg）；与 --then-goto 互斥")
ap.add_argument("--then-goto", default=None, choices=["place", "pick"],
                help="--post-gate warm-move 的横移走 map 系绝对坐标直达"
                     "（g1_warm_core.nav_goto_station：navigate_to_goal"
                     "(collision_check=False)+0.5s 轮询判到达；10-14 同"
                     " g1_base_move --goto）；与 --then-move 互斥。编排器"
                     "默认发 --then-goto place")
ap.add_argument("--move-timeout", type=float, default=30.0,
                help="--then-move/--then-goto 的导航内部超时 s")
ap.add_argument("--phase2-ckpt", default=None,
                help="二相双构造旧路径：place ckpt 目录，折叠窗口内后台载入"
                     "+烧引擎，折叠后换权重切相（10-14 起可选——0930 权重实为"
                     " pick+place 合并训练，只传 --phase2-prompt 即单模型切相，"
                     "省第二构造 ~26s 载入；本参数留作 A/B/回退）")
ap.add_argument("--phase2-prompt", default=None,
                help="二相 place 轮的 prompt（编排器传 place-prompt）。只给本参"
                     "不给 --phase2-ckpt=单模型切相（同权重只换 prompt）")
ap.add_argument("--phase2-pace", type=float, default=None,
                help="切相后 place 轮配速窗口 s（默认沿用 pick 值）。place 相"
                     "导航栈醒着（二相设计如此，+55ms/轮），pace 须盖住醒着"
                     "水位 ~0.42——pace<水位=续航背靠背→夹爪翻摆 500+ mrad"
                     "（9/30 sweep 0.38 判死实录，二相 place@0.35 同条件）")
ap.add_argument("--phase2-pace-div", type=float, default=None,
                help="切相后 place 轮配速系数（默认沿用；0.43 那套配 1.65，"
                     "9-30 a25a59c 定案）")
args = ap.parse_args()

# ── 全文日志落盘：q 退出走 os._exit 不刷缓冲（09-23 坑），故逐行强制 flush ──
_log_path = args.log_file or str(
    pathlib.Path(__file__).resolve().parents[3] / "logs" /
    time.strftime("loop_%Y%m%d_%H%M%S.log"))
pathlib.Path(_log_path).parent.mkdir(parents=True, exist_ok=True)
print(f"[B] sft1012 头图裁剪放大: {'开' if args.head_crop else '关'}"
      f"（--no-head-crop 可关）")


class _Tee:
    """stdout/stderr 双写：硬拷贝到日志文件（crash traceback 也收）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


_log_fh = open(_log_path, "a", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)
print(f"[log] 全文日志 → {_log_path}")
g1_config.apply(args, {
    "ckpt": ("run", "ckpt"),
    "prompt": ("run", "prompt"),
    "rounds": ("loop", "rounds"),
    "n_action_steps": ("loop", "n_action_steps"),
    "steps_per_command": ("loop", "steps_per_command"),
    "delta_max": ("loop", "delta_max"),
    "speed": ("loop", "speed"),
    "pace": ("loop", "pace"),
    "pace_div": ("loop", "pace_div"),
    "near_div": ("loop", "near_div"),
    "near_gap": ("loop", "near_gap"),
    "near_bypass": ("loop", "near_bypass"),
    "near_n": ("loop", "near_n"),
    "near_exit": ("loop", "near_exit"),
    "cache_frames": ("loop", "cache_frames"),
    "max_excursion": ("loop", "max_excursion"),
    "settle": ("loop", "settle"),
    "settle_frac": ("loop", "settle_frac"),
    "switch_dist": ("loop", "switch_dist"),
    "chunk_mode": ("loop", "chunk_mode"),
    "traj_dt": ("loop", "traj_dt"),
    "grip": ("gripper", "enabled"),
    "grip_speed": ("gripper", "speed"),
    "grip_effort": ("gripper", "effort"),
    "grip_chg": ("gripper", "chg"),
})
if args.no_grip:
    args.grip = False
print(f"[loop] 迟到兜底: "
      f"{'stale（--late-action stale）' if args.late_action == 'stale' else 'hold（默认）'}"
      f" | hold 上限 {args.hold_max} 窗"
      + (f" | 坡升 {args.hold_ramp} 轮（hold≥2 窗后限速 0.5→1.0）"
         if args.hold_ramp > 0 else ""))


# ── 导航栈 SIGSTOP 搁置（--nav-suspend 开启，默认关）──
# 机制：kill -STOP 进程原地冻结——无 exit 事件，launcher 的退出拉起逻辑不触发
# （2026-10-07 vtn 8s 探针实证：冻结期 launcher 日志零增量、兄弟全活），
# CONT 原地复活，不重初始化不重载图。约束：只用于「臂上任务+底盘不动」窗口。
# 恢复多重保险：① 各退出路径显式 resume（q/Ctrl-C/异常/正常完成）；
# ② 独立守护进程盯父进程，父进程任何死法（含 kill -9/段错误）即 CONT 全部。
NAV_SUSPEND_NAMES = (
    "localization_server", "galbot_fusion_main", "service_navigation_plan",
    "galbot_vtn", "surround_cameras_capture", "swallows", "service_lidar_capture",
)
_NAV_GUARD_SRC = """
import os, sys, time, signal
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
ppid = os.getppid()
pids = [int(x) for x in sys.argv[1].split(',') if x]
while True:                      # 父进程（本脚本）消失才往下走
    time.sleep(0.3)
    try:
        os.kill(ppid, 0)
    except OSError:
        break
for p in pids:
    try:
        os.kill(p, signal.SIGCONT)   # 最后一道网：主进程任何死法都恢复导航栈
    except OSError:
        pass
"""


class NavSuspend:
    """导航栈搁置器：只停 NAV_SUSPEND_NAMES 内、且当前不在冻结态的进程。

    只认领自己 STOP 的 pid（resume 时只 CONT 这些），对别人冻结的进程不碰。
    """

    def __init__(self):
        self.stopped = {}                     # pid -> name
        self._guard = None

    def _find_pids(self):
        found = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    arg0 = f.read().split(b"\x00")[0].decode("utf-8", "replace")
            except OSError:
                continue
            base = os.path.basename(arg0)
            if base in NAV_SUSPEND_NAMES:
                found[int(pid)] = base
        return found

    def suspend(self):
        cands = self._find_pids()
        missing = [n for n in NAV_SUSPEND_NAMES if n not in cands.values()]
        if missing:
            print(f"[nav-suspend] ⚠ 未找到（跳过）: {missing}")
        # 守护网先起（继承 pid 表），再下发 STOP——任何死法都有人收尾
        self._guard = subprocess.Popen(
            [sys.executable, "-c", _NAV_GUARD_SRC,
             ",".join(str(p) for p in sorted(cands))],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for pid, name in sorted(cands.items()):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    if f.read().split()[2] == "T":   # 已冻结（非我们所停）：不碰
                        continue
            except (OSError, IndexError):
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
                self.stopped[pid] = name
            except OSError as e:
                print(f"[nav-suspend] ⚠ STOP {name}({pid}) 失败: {e}")
        print(f"[nav-suspend] 已冻结 {len(self.stopped)} 进程: "
              + ", ".join(f"{n}({p})" for p, n in sorted(self.stopped.items()))
              + f" | 守护网 pid={self._guard.pid}")
        print("[nav-suspend] ⚠ 冻结期底盘勿动（localization 位姿停更）；"
              "相机 capture 不在冻结集，取图不受影响")

    def resume(self):
        if not self.stopped:
            return
        for pid in sorted(self.stopped):
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        print(f"[nav-suspend] 已恢复 {len(self.stopped)} 进程"
              "（CONT 幂等；守护网发现主进程退出后自退）")
        self.stopped.clear()


NAV = None   # suspend 后指向 NavSuspend 实例；各退出路径 None 守卫调用 resume

# ── 门触发折叠（--post-gate，10-13）：编排器门触发注入 'r' → 主循环止步 →
# 同会话复位+横移（省 warmup 子进程 ~4-5s）。QuitWatcher 收 'r' 置事件不退程；
# 折叠期间 q 仍可急停（rc=4=折叠中止，编排器按失败拦下不进 place）──
FOLD_EVENT = threading.Event()      # 'r' 触发 → 主循环下一检查点止步
FOLD_REQUESTED = False              # 参数校验后置位（--post-gate != none）
FOLD_MOVE = False                   # warm-move 档（摆臂后横移）


class QuitWatcher:
    """后台线程监听键盘 q：任何阶段即时退出（SDK 阻塞 C++ 调用吞 Ctrl-C）。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._old = None
        self._wake_r = None
        self.active = threading.Event()
        if os.isatty(self.fd):
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.restore)
            self.active.set()       # tty：构造即监听（提示符处 pause/resume 照旧）
        # else：编排器 stdin=管道。termios 设不了（ENOTTY），但 select+read 照常
        # 收 'q'——10-13 实锤 isatty 门把监听线程整个跳过=门触发写 q 无人读，
        # loop 一路跑到相机冻结看门狗收尾（131514~140210 五跑 q=0）。
        # active 留空=不与提示符 input() 抢预喂的 '\n'，resume()（提示符后）才开听。
        r, w = os.pipe()                # 信号直通管道：C 层收信号即写字节，
        os.set_blocking(w, False)       # 主线程卡死在 SDK C++ 调用也能退出
        signal.set_wakeup_fd(w)
        signal.signal(signal.SIGINT, lambda *_: None)     # 退出走管道，别靠
        signal.signal(signal.SIGTERM, lambda *_: None)    # 会被推迟的异常
        self._wake_r = r
        threading.Thread(target=self._loop, daemon=True).start()
        if os.isatty(self.fd):
            print("（循环期间随时按 q 或 Ctrl-C 退出；确认提示符处用回车；急停第一优先级）")
        else:
            print("（stdin=管道：'q' 监听已挂（编排器门触发/超时写 q 生效）；"
                  "Ctrl-C 由父进程转发）")

    def restore(self):
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def pause(self):
        self.active.clear()

    def resume(self):
        self.active.set()

    def _loop(self):
        while True:
            try:
                fds = [self._wake_r] if self._wake_r is not None else []
                if self.active.is_set():
                    fds.append(sys.stdin)
                if not fds or not select.select(fds, [], [], 0.2)[0]:
                    continue
                if (self._wake_r is not None
                        and select.select([self._wake_r], [], [], 0)[0]):
                    os.read(self._wake_r, 1)
                    print("\n⛔ 收到中断信号 —— 立即退出"
                          "（已下发目标可能仍在限速执行）")
                    self.restore()
                    if NAV is not None:
                        NAV.resume()
                    os._exit(130)
                ch = sys.stdin.read(1)
                if ch in ("q", "Q"):
                    folding = FOLD_EVENT.is_set()
                    print("\n⛔ 按下 q —— 立即退出（已下发目标可能仍在限速执行）"
                          + ("〔折叠中止 rc=4〕" if folding else ""))
                    self.restore()
                    if NAV is not None:
                        NAV.resume()
                    os._exit(4 if folding else 2)
                if ch in ("r", "R") and FOLD_REQUESTED and not FOLD_EVENT.is_set():
                    print("\n[fold] 收到 'r' —— 本轮止步后执行折叠 reset"
                          + ("+横移" if FOLD_MOVE else ""))
                    FOLD_EVENT.set()
            except Exception:
                self.restore()
                if NAV is not None:
                    NAV.resume()
                os._exit(3)   # 监听线程死了比静默更危险：宁可误退不可失控


# ── 折叠参数校验（fail loud，缺件直接退出，不进循环）──
FOLD_REQUESTED = args.post_gate != "none"
FOLD_MOVE = args.post_gate == "warm-move"
if FOLD_REQUESTED:
    if not args.fold_ckpt or not args.fold_pose_file:
        raise SystemExit("--post-gate warm/warm-move 需要 --fold-ckpt + "
                         "--fold-pose-file（place 起始位姿来源）")
    if FOLD_MOVE:
        if args.then_move is not None and args.then_goto is not None:
            raise SystemExit("--then-move 与 --then-goto 互斥，二选一")
        if args.then_move is None and args.then_goto is None:
            raise SystemExit("--post-gate warm-move 需要 --then-move X Y "
                             "YAW_DEG 或 --then-goto place|pick")

# ── 二相参数校验（fail loud）──
# 两态（2026-10-14 同事确认定案）：0930 权重=4 任务 pick+place 合并训练
# （train_config wandb notes 实锤），place 无需换权重——只给 --phase2-prompt
# 即单模型切相（省第二构造载入+建引擎 ~26s 与 +45s 预算）；仍传 --phase2-ckpt
# 则走旧双构造路径（A/B/回退用）。
PH2 = None
if args.phase2_prompt:
    if args.post_gate != "warm-move":
        raise SystemExit("--phase2-* 需要 --post-gate warm-move（切相发生在"
                         "折叠+横移之后，横移是 place 工位的必要位移）")
    if args.phase2_ckpt:
        _p2 = pathlib.Path(args.phase2_ckpt)
        if not (_p2 / "flashrt_deploy.json").exists():
            raise SystemExit(f"--phase2-ckpt 缺 flashrt_deploy.json: {_p2}")
        PH2 = {"ckpt": _p2, "prompt": args.phase2_prompt,
               "model": None, "error": None}
        print(f"[phase2] 二相开（双构造旧路径）：place ckpt {_p2.name}（折叠"
              "窗口内后台载入，折叠后原地切相；place 轮 grip-state-cmd 自动关）")
    else:
        PH2 = {"ckpt": None, "prompt": args.phase2_prompt,
               "model": None, "error": None}
        print("[phase2] 二相开（单模型）：同权重只换 prompt——0930 权重为 "
              "pick+place 合并训练（train_config 实锤），无第二构造载入；"
              "place 轮 grip-state-cmd 自动关、包络沿用（pick 侧包络本就派生"
              "自 199 集 pick+place 合并数据）")
elif args.phase2_ckpt:
    raise SystemExit("--phase2-ckpt 需要 --phase2-prompt（place 轮 prompt）")

WATCH = QuitWatcher()


def _fatal_hook(t, v, tb):
    """任何未捕获异常：关 SDK + 还原终端 + 硬退。

    裸退会让 SDK 残留线程在解释器关闭时段错误（fleet 已知坑）。
    """
    print(f"\n⛔ 异常退出: {t.__name__}: {v}")
    rob = globals().get("robot")
    if rob is not None:
        try:
            rob.request_shutdown(); rob.wait_for_shutdown(); rob.destroy()
        except Exception:
            pass
    WATCH.restore()
    if NAV is not None:
        NAV.resume()
    os._exit(1)


sys.excepthook = _fatal_hook

# ── 部署清单 ──
CKPT = pathlib.Path(args.ckpt)
mf_p = CKPT / "flashrt_deploy.json"
mf = json.loads(mf_p.read_text()) if mf_p.exists() else {}
ACTION_DIM = int(mf.get("action_dim", 16))
VIEWS = int(mf.get("views", 3))
CAM_MAP = mf.get("camera_map", {"image": "HEAD_RIGHT_CAMERA",
                                "wrist_image": "LEFT_ARM_CAMERA",
                                "wrist_image_right": "RIGHT_ARM_CAMERA"})

# ── 关节包络护栏（压桌防护，2026-10-06）：任务数据集臂关节合法构型盒。SDK 侧
# 臂端 fault/堵钻不回传（指令恒 SUCCESS，10-06 实录），刚性伺服撞桌只会硬压到
# 进 fault——唯一的软防线是在指令侧拦住"训练分布外的构型"。Motion.init 脱机
# 挂死（实测 >120s），FK 路线不可用，故用数据集包络做几何代理。
# ⚠ 2026-10-07 修范畴错误：模型输出=delta（relative_actions_processor，见
# plan_step），护栏比对对象必须是 BASE_ARM+delta 绝对构型（chunk_abs_block），
# 不能直接比 delta——否则首块 delta≈0 必"越界"拦停（10-07 首跑实录）──
ENV_LO = ENV_HI = None
_env_file = CKPT / "joint_envelope.json"
if args.no_envelope_guard:
    print("[护栏] 关节包络护栏已 --no-envelope-guard 关闭")
elif _env_file.exists():
    _env = json.loads(_env_file.read_text())
    # 此处在 import numpy 之前，只做纯 python 解析；数组化+余量在下方
    # import numpy 之后统一做（10-07 修 np 前置引用 bug——pick 时代同炸，
    # 只是护栏合入后没有真机全跑过所以没暴露）
    _inf = float("inf")
    ENV_LO = [(-_inf if v is None else float(v)) for v in _env["action_lo"]]
    ENV_HI = [(_inf if v is None else float(v)) for v in _env["action_hi"]]
    print(f"[护栏] 关节包络开：{_env_file.name}（来源 {_env.get('source', '?')}；"
          f"余量 ±{args.envelope_margin} rad；越界=截断 chunk，连续 {args.envelope_abort_n} "
          f"轮→停）")
else:
    print(f"[护栏] 未找到 {_env_file}——关节包络护栏关闭（当前 ckpt 无任务包络）")

# ── 夹爪下发配置（--grip；宽度来自 manifest 标定，无标定拒绝开启）──
GRIP = None
if args.grip:
    g = mf.get("gripper", {})
    wmin, wmax = g.get("width_min"), g.get("width_max")
    if wmin is None or wmax is None:
        raise SystemExit("--grip 需要 manifest 夹爪标定；先重跑 "
                         "g1_ckpt_prep.py --grip-wmin/--grip-wmax（2026-09-23 实测 "
                         "0.0005/0.1200 m）")
    GRIP = {"names": (("right_gripper", 7), ("left_gripper", 15)),
            "wmin": float(wmin), "wmax": float(wmax),
            "speed": args.grip_speed, "effort": args.grip_effort,
            "chg": args.grip_chg, "sent": {}, "sent_peak": {}, "grasp_r": -10**9}
    print(f"夹爪下发开启: 0%→{wmin} m | 100%→{wmax} m | 速度 {args.grip_speed} m/s | "
          f"力矩 {args.grip_effort} N | 变化阈值 {args.grip_chg}%")
    if args.grip_state_cmd:
        print("⚠ state 夹爪维(7/15)喂【指令值】（--grip-state-cmd）：SDK 回读滞后 "
              "~6.3-8s > pick 窗口 ~7s，回读=冻结起始值（10-05 四跑实证）")

# 2026-09-23 tf_matrix 实证：INT8 两档（全 INT8 / 仅编码器）均毁动作质量
# （块均 cos 0.15/0.27 vs bf16 0.98）——定档 bf16，显式锁定（详见 BENCHMARKS 附录）
os.environ.setdefault("FVK_PI05_RTX_FORCE_BF16", "1")
# 图开关：0=开图（默认，2026-09-29 WithFlags 热修已打、9/30 bench 图模式稳定，
# 收益 ~3%）；回退 eager：前缀 PI05_NO_GRAPH=1
os.environ.setdefault("PI05_NO_GRAPH", "0")
# state 以十进制文本拼进 prompt（format_pi05_prompt）：关节值一漂、bin 数位
# 变化 → token 数变；默认 exact 模式每种长度一条 pipeline，换长=整条重建+
# 重 autotune（~800ms，2026-09-23 合成实验实锤）。fixed=定长 200 一条
# pipeline 只换 embeds，实测含 state 切换恒定 240-253ms
os.environ.setdefault("FLASHRT_PI05_STATE_PROMPT_MODE", "fixed")
# chunk 长度：pi05_rtx 前端模块导入时读此 env，必须在下面 import flash_rt 前定死
os.environ["FLASH_RT_PI05_ACTION_CHUNK_SIZE"] = str(args.chunk_size)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

# 包络护栏数组化+余量（解析在上方 import 之前完成；margin 语义同 10-06 原版）
if ENV_LO is not None:
    _m = args.envelope_margin
    ENV_LO = np.array(ENV_LO, dtype=np.float32)
    ENV_HI = np.array(ENV_HI, dtype=np.float32)
    ENV_LO[:7] -= _m
    ENV_LO[8:15] -= _m
    ENV_HI[:7] += _m
    ENV_HI[8:15] += _m
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

# 回退 eager 时才封图（PI05_NO_GRAPH=1）；默认开图（scripts/test/bench_pi05.py 同款）
if os.environ.get("PI05_NO_GRAPH", "0") == "1":
    _orig_init = _fe.Pi05TorchFrontendRtx.__init__

    @functools.wraps(_orig_init)
    def _no_graph_init(self, *a, **kw):
        kw["use_cuda_graph"] = False
        _orig_init(self, *a, **kw)

    _fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init
    print("[loop] CUDA graph 已禁用（PI05_NO_GRAPH=1），eager 模式")

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.g1 import (  # noqa: E402
    GalbotRobot, SensorType, Trajectory, TrajectoryPoint, JointCommand,
    G1JointGroup)

# ── 关节表（数据集维序：右臂在前）──
LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT            # 与 chunk[0:7]+chunk[8:15] 逐位配对
STATE_NAMES = (RIGHT + ["right_gripper_joint1"]
               + LEFT + ["left_gripper_joint1"]
               + [f"leg_joint{i}" for i in range(1, 6)]
               + ["head_joint1", "head_joint2"])
GRIP_IDX = (7, 15)                  # state 里的夹爪维：SDK 米 → 数据集 0~100%
ARM_SLICE = list(range(0, 7)) + list(range(8, 15))   # state 里的臂维（0-6 右 / 8-14 左）
# 当前 chunk 的预测时刻臂位——chunk 臂维是 delta（见 plan_step），换块时更新
BASE_ARM = None
_gcal = mf.get("gripper", {})
GRIP_WMIN, GRIP_WMAX = _gcal.get("width_min"), _gcal.get("width_max")


def state_from_joints(vals) -> np.ndarray:
    """原始关节读数 → 模型 state：夹爪两维按 manifest 标定换算 0~100%。

    数据集 state dim7/15 是百分比；SDK 读数是米——直接喂会被归一化成
    恒 ≈闭合（0.12m 当 0.12% 算），模型看到的夹爪状态永远错。
    """
    st = np.array(vals, dtype=np.float32)
    if GRIP_WMIN is None or GRIP_WMAX is None:
        print("⚠ 夹爪未标定：SDK 宽度(米)原样进 state，与数据集 0~100% 单位不符!")
        return st
    for i in GRIP_IDX:
        st[i] = float(np.clip((st[i] - GRIP_WMIN)
                              / (GRIP_WMAX - GRIP_WMIN + 1e-9) * 100.0,
                              0.0, 100.0))
    return st


def grip_state_cmd_override(st_raw):
    """--grip-state-cmd：state 夹爪维(7/15)改喂指令值，持物段例外（10-12 修正）。

    SDK 夹爪反馈滞后 ~6.3-8s（2026-09-23 实测），pick 的关键窗口只有 ~7s——
    跟随段喂回读=模型全程看到冻结在起始值的夹爪状态；数据集 state 跟随指令
    仅 ~0.3-1s，故跟随段喂指令值。

    ⚠ 持物签名修正（2026-10-12 帧级实锤）：数据集「抓到物体」语义=指令 0%
    + state 被物体撑在 ~33.5%（ep0 t5.1s 起；40 集持物段 state 中位 33.5、
    p10-p90=33.0-33.5 恒定）——模型靠该组合识别「已持物→后撤→交接」。
    闭合后若照喂指令 0% = 把签名抹成「空爪闭紧」（预抓取/释放后分布），
    交接序列永不触发（loop_20261012_165104 实录：L t+5.3s 抓住、模型不后撤、
    R 爪全程 0%·hold）。故分三段：
      开爪/跟随段（指令 >10%）：喂指令值（含回读不可信的整个滞后窗）；
      持物段（指令 <10% 且该爪历史指令峰值 >50% = 开→合全程走完）：
        喂持物签名 33.5（数据集恒定值；SDK 回读在交接关键窗内仍处滞后，弃用）；
      未开过爪的闭合（峰值 ≤50%，如起步双爪 0%）：喂指令值（=空爪闭合，正确）。
    已知限制：释放后（指令仍 0）滞留持物签名——pick 线释放后任务即转 R 臂，
    state 不再参与关键决策，首版接受；空抓（夹空）时也会误报持物，行为=
    提前交接 R，属可接受恢复路径。
    """
    if GRIP is None or not args.grip_state_cmd:
        return st_raw
    st = np.array(st_raw, dtype=np.float32)
    for name, dim in GRIP["names"]:
        sent = GRIP["sent"].get(name)
        if sent is None:
            continue
        peak = GRIP["sent_peak"].get(name, 0.0)
        GRIP["sent_peak"][name] = max(peak, sent)
        if sent < 10.0 and GRIP["sent_peak"][name] > 50.0:
            st[dim] = 33.5
        else:
            st[dim] = sent
    return st


def read_joints(robot, names) -> np.ndarray:
    vals = robot.get_joint_positions([], names)
    if not vals or len(vals) != len(names):
        raise SystemExit(f"关节读取失败（返回 {len(vals) if vals else 0} 维）")
    return np.array(vals, dtype=np.float32)


_LAST_FRAMES = {}     # 相机冻结守卫：逐轮像素级对比（2026-09-24 右臂下压排查引入）
_FREEZE_STREAK = {}


def letterbox224(img):
    """等比缩放长边到 224 + 上下纯黑条填充——精确复现训练数据集的图像变换。

    2026-10-12 实锤（pick_place_balence 抽帧判读）：三视角训练帧全部为
    letterbox——224×224、上下各 28px 纯黑条、内容带 168px=224×960/1280（源帧
    4:3 等比缩放，非拉伸非裁剪；本机 SDK 原生 640×480/1280×960 均 4:3，
    infer_policy 的 crop 步为 no-op）。原直拉 resize 满幅无条=train/serve
    图像分布失配（run_g1_loop_traj.py 同款修复）。
    """
    h, w = img.shape[:2]
    _s = 224.0 / max(h, w)
    _nh, _nw = round(h * _s), round(w * _s)
    _cv = np.zeros((224, 224, 3), dtype=np.uint8)
    _y, _x = (224 - _nh) // 2, (224 - _nw) // 2
    _cv[_y:_y + _nh, _x:_x + _nw] = cv2.resize(img, (_nw, _nh))
    return _cv


# ── B 侧差异：头图训练裁剪放大（2026-10-15 A/B，sft_1012 数据）──────────
# 复现同事数据管线对头相机的处理：/home/galbot/image_preprocessor.py
# ImagePreprocessor._apply_head_training_crop 原样内联（其顶层 import
# emulate_data_collection 在 /home/galbot/infer_policy/，跨目录 import 脆弱，
# 故不 import 只内联，常量逐一对齐源文件）。
# 训练视频头图 = letterbox224 帧上裁 [126:214, 52:172] → Lanczos4 拉回
# 164×224（≈1.87× 放大）→ 上下各补 30px 黑条回 224。推理漏做 = train/serve
# 图像分布失配。仅作用于头相机键 "image"（双腕图训练无此处理）；裁剪/缩放/
# 补边均为逐通道空间算子，BGR/RGB 通道序无关。
HEAD_TRAINING_CROP_YX = (126, 214, 52, 172)   # y0, y1, x0, x1（224×224 输入上）
HEAD_TRAINING_CONTENT_HW = (164, 224)         # 裁剪内容拉回的目标 (H, W)
HEAD_TRAINING_PAD_TBLR = (30, 30, 0, 0)       # 上/下/左/右黑边 px


def head_training_crop(img):
    """letterbox224 后的头图 → 训练同款裁剪放大回 224×224。"""
    if img.ndim != 3 or img.shape[:2] != (224, 224):
        raise ValueError(f"head_training_crop 期望 224x224 HWC 输入，"
                         f"got shape={img.shape}")
    y0, y1, x0, x1 = HEAD_TRAINING_CROP_YX
    if not (0 <= y0 < y1 <= img.shape[0] and 0 <= x0 < x1 <= img.shape[1]):
        raise ValueError(f"head_training_crop 窗口 [{y0}:{y1}, {x0}:{x1}] 越界")
    ch, cw = HEAD_TRAINING_CONTENT_HW
    content = cv2.resize(img[y0:y1, x0:x1], (cw, ch),
                         interpolation=cv2.INTER_LANCZOS4)
    pt, pb, pl, pr = HEAD_TRAINING_PAD_TBLR
    out = cv2.copyMakeBorder(content, pt, pb, pl, pr,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if out.shape[:2] != (224, 224):
        raise ValueError(f"head_training_crop 输出形状异常: {out.shape[:2]}")
    return out


def grab_views(robot) -> dict:
    out = {}
    for key, cam in list(CAM_MAP.items())[:VIEWS]:
        d = robot.get_rgb_data(getattr(SensorType, cam))
        if not d or not d.get("data"):
            raise SystemExit(f"取图失败: {cam}")
        img = cv2.imdecode(np.frombuffer(d["data"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"解码失败: {cam}")
        last = _LAST_FRAMES.get(cam)
        _LAST_FRAMES[cam] = img
        if last is not None and np.array_equal(last, img):
            _FREEZE_STREAK[cam] = _FREEZE_STREAK.get(cam, 0) + 1
        else:
            _FREEZE_STREAK[cam] = 0
        if _FREEZE_STREAK[cam] >= 3:
            raise SystemExit(f"⛔ 相机流疑似冻结：{cam} 连续 3 轮逐字节相同"
                             f"（实时流有传感器噪声不可能全等）——观测已失真，"
                             f"停止循环。排查 RT 相机服务")
        _lb = letterbox224(img)
        if args.head_crop and key == "image":
            _lb = head_training_crop(_lb)   # B 侧差异：头图训练裁剪放大
        out[key] = np.ascontiguousarray(cv2.cvtColor(_lb, cv2.COLOR_BGR2RGB))
    return out


class LatencyTracker:
    """时延样本收集器。结构参照 lerobot policies/rtc/latency_tracker.py
    （Apache-2.0），两处有意偏离（docs/lerobot-alignment.md §七）：
    maxlen 无界（lerobot 默认 100 会截断 200 轮跑的统计窗口）、保持 float64
    （lerobot 的 float32 cast 可能让 .1f 打印差 0.1ms——本类与旧
    np.percentile 路径逐字节等价：np.quantile(x, 0.95)≡np.percentile(x, 95)）。"""

    def __init__(self):
        self._buf = deque()   # 无界；perf_counter 差恒正，负值丢弃只是护栏

    def add(self, value_ms):
        if value_ms >= 0.0:
            self._buf.append(value_ms)

    def __len__(self):
        return len(self._buf)

    def __getitem__(self, i):
        return self._buf[i]          # exec 轮内打印用 step_ms[-1]/round_ms[-1]
                                     # （10-08 对齐重构漏配，10-09 首次 exec 实证崩）

    def max(self):
        return max(self._buf) if self._buf else float("nan")

    def percentile(self, q):
        if not self._buf:
            return float("nan")
        return float(np.quantile(np.asarray(self._buf), q))

    def p95(self):
        return self.percentile(0.95)


def pstats(tr):
    """p50/p95/max 毫秒，一行（tr: LatencyTracker）。"""
    return (f"p50 {tr.percentile(0.5):.1f} | p95 {tr.percentile(0.95):.1f} | "
            f"max {tr.max():.1f} ms")


def chunk_abs_block(chunk):
    """chunk delta 块 → 绝对构型块（包络护栏的比对对象，2026-10-07 修范畴错误）。

    训练管线 relative_actions_processor：臂维=delta（相对本块预测时刻臂位
    BASE_ARM）、夹爪维=绝对 0-100（plan_step/build_traj 同款换算）。
    包络盒来自数据集 action 列逐维 [min,max]=绝对构型——必须用 BASE_ARM+delta
    比对；直接拿 delta 比对=必拦停（delta≈0 恒不在盒内，10-07 首跑实录）。
    """
    a = np.empty_like(chunk)
    a[:, :7] = chunk[:, :7] + BASE_ARM[None, :7]
    a[:, 7] = chunk[:, 7]            # 夹爪界=±inf，原值恒过
    a[:, 8:15] = chunk[:, 8:15] + BASE_ARM[None, 7:14]
    a[:, 15] = chunk[:, 15]
    return a


def env_first_violation(ch):
    """关节包络检查：返回首个越界行 (行号, 维, 值, 界)；全干净返回 None。

    ch: (n,16) 数据集维序动作块。臂维 0-6=右臂、8-15=左臂（ENV 已含余量），
    夹爪维界=±inf 恒过。逐行向量化，n≤chunk_size(50)，微秒级。
    """
    bad = (ch > ENV_HI[None, :]) | (ch < ENV_LO[None, :])
    rows = np.flatnonzero(bad.any(axis=1))
    if rows.size == 0:
        return None
    r = int(rows[0])
    j = int(np.flatnonzero(bad[r])[0])
    upper = bool(ch[r, j] > ENV_HI[j])
    return r, j, float(ch[r, j]), float(ENV_HI[j] if upper else ENV_LO[j])


def _phys_mm(robot, name):
    """物理爪宽回读（m→mm）；读不到回 None（编排器门按未持物处理=安全向）。"""
    try:
        gs = robot.get_gripper_state(getattr(G1JointGroup, name))
    except Exception:
        return None
    return None if gs is None else gs.width * 1000.0


def send_grip(robot, chunk_row):
    """chunk 行 dim7/dim15（0~100%）→ 标定宽度下发（超阈值才发，非阻塞）。

    本函数只发不查不等待（否则拖死控制环）；终态核对放循环结束后。
    行尾附【实 X.Xmm】=物理回读宽度（10-13 爪标定：回读新鲜 p50 10ms，
    推翻 6.3s 滞后旧说）——编排器相位门 v4 靠它判「真持物」：
    指令闭合≠抓住（143656 实证空爪闭合指令 0-3.5% 照样持续），实宽才是。
    """
    if GRIP is None:
        return None
    parts = []
    for name, dim in GRIP["names"]:
        p = float(np.clip(chunk_row[dim], 0.0, 100.0))
        pm = _phys_mm(robot, name)
        ptag = "[实 ?mm]" if pm is None else f"[实 {pm:.1f}mm]"
        last = GRIP["sent"].get(name)
        if last is not None and abs(p - last) < GRIP["chg"]:
            parts.append(f"{name[0].upper()} {p:.1f}%·hold{ptag}")
            continue
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        st = robot.set_gripper_command(getattr(G1JointGroup, name),
                                       w, GRIP["speed"], GRIP["effort"], False)
        GRIP["sent"][name] = p
        if last is not None and last >= 10.0 and p < 10.0:
            GRIP["grasp_r"] = globals().get("r", 0)   # 开→合穿越=抓取事件（近阻尼旁路锚点）
        parts.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm{ptag} "
                     f"{str(st).replace('ControlStatus.', '')}")
    return "  ".join(parts)


# ── ① SDK 初始化 + 模型 ──
print("== ① SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
sensors = {getattr(SensorType, c) for c in list(CAM_MAP.values())[:VIEWS]}
if not robot.init(sensors):
    raise SystemExit("robot.init 失败")
time.sleep(5)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")
if not str(st).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止（先跑 run_g1_execute.py 排查）")

if args.nav_suspend:
    NAV = NavSuspend()
    NAV.suspend()

HOME = read_joints(robot, ARM_NAMES)   # 起始位 = 漂移护栏基准（应在预热工作位）
print(f"起始位（漂移护栏基准）: {np.round(HOME, 3).tolist()}")
print("⚠ 若当前不是预热工作位，先跑 g1_pose_warmup.py 再来")

t0 = time.time()
model = flash_rt.load_model(str(CKPT), config="pi05", num_views=VIEWS,
                            cache_frames=args.cache_frames, action_dim=ACTION_DIM)
ns = model._pipe.norm_stats
print(f"tier=int8_full views={VIEWS} | load {time.time() - t0:.1f}s")

obs = grab_views(robot)
state0 = state_from_joints(read_joints(robot, STATE_NAMES))
state_n0 = normalize_state(state0, ns)
BASE_ARM = state0[ARM_SLICE]   # 首块 delta 基准 = 预测时刻臂位
chunk = np.asarray(model.predict(obs, prompt=args.prompt, state=state_n0))  # 建管线
# 首用引擎构建吸收：真实控制轮 0 曾撞 ~800ms 惰性构建（autotune 中途重现，
# 2026-09-22 tegrastats 已排除热/内存）。连做 3 次新鲜取图推理，把构建成本
# 烧在计时区外，避免首轮尖峰污染遥测
for _ in range(3):
    model.predict(grab_views(robot), prompt=args.prompt,
                  state=normalize_state(read_joints(robot, STATE_NAMES), ns))
print("预热推理 ×3 完成（吸收惰性引擎构建）")

# ── 二相开局同载（2026-10-14 v2）：place 构造在启动段就建好引擎——GPU 空载
# 串行，不与 pick 推理竞争。v1 藏折叠窗口实测 26.0s > 窗口 16.8s，漏出 ~9s
# 空档；v2 代价=pick 起跑晚一次 place 载入+建引擎（编排器预算已放宽）。
# autotune 只是把每层 GEMM 算法提前选定，不影响推理数值/速度（不同算法=同
# 数学不同归约序，差异在既有运行间抖动带内）。失败不拦 pick：PH2["error"]
# 留到切相时响亮失败（pick 是贵的一段，别让 place 侧问题烧掉它）──
if PH2 is not None and PH2["ckpt"] is not None:
    t_p2 = time.time()
    try:
        _m2 = flash_rt.load_model(str(PH2["ckpt"]), config="pi05",
                                  num_views=VIEWS,
                                  cache_frames=args.cache_frames,
                                  action_dim=ACTION_DIM)
        _rng = np.random.default_rng(0)
        _obs = {k: _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
                for k in list(CAM_MAP)[:VIEWS]}
        _st = np.linspace(-0.5, 0.5, ACTION_DIM, dtype=np.float32)
        _m2.predict(_obs, prompt=PH2["prompt"], state=_st)   # 建引擎 autotune
        _m2.predict(_obs, prompt=PH2["prompt"], state=_st)
        PH2["model"] = _m2
        print(f"[phase2] place 构造开局就绪（载入+建引擎 "
              f"{time.time() - t_p2:.1f}s，pick 起跑相应延后）")
    except Exception as e:
        PH2["error"] = e
        print(f"⛔ [phase2] place 构造开局载入失败（pick 照跑，切相时中止）: {e!r}")
n_steps = max(1, min(args.n_action_steps, args.chunk_size))
spc = max(1, min(args.steps_per_command, n_steps))   # 合步宽度（≤ 每轮步数）
BUDGET = spc * args.delta_max                    # 每条指令位移限幅
V_MIN = 0.02                                     # 半程配速下限 rad/s（驻停时缓爬）
if args.pace < 0.55:
    print(f"⚠ --pace {args.pace}s 低于 bf16 推理+取图水位 ~0.42s："
          + ("推理迟到将以 hold 填补（单调收敛无深尾，但每个 hold 窗是"
             "实打实墙钟；轮询早退只免掉窗内剩余）" if args.late_action != "stale" else
             "推理未就绪段以旧 chunk 深尾续航填补（stale 模式：计划垃圾、臂往复风险）"))
if args.pace_div < 1.0:
    print(f"⚠ --pace-div {args.pace_div} < 1：臂会提前到点干等窗口结束"
          "（速度过零停顿），无收益；半程=2 / 全程=1")
if 0 < args.near_div < 1.0:
    print(f"⚠ --near-div {args.near_div} < 1：近距反而提速会加剧过冲，建议 2；0=关闭")
near_n_eff = None
if args.near_n:
    if not (1 <= args.near_n <= n_steps):
        near_n_eff = max(1, min(args.near_n, n_steps))
        print(f"⚠ --near-n {args.near_n} 出界（1~n_action_steps {n_steps}），钳到 {near_n_eff}")
    else:
        near_n_eff = args.near_n


def plan_step(k, cur, budget=None):
    """chunk 第 k 步 → 限幅后目标（14 维，右臂在前）；budget=位移限幅。

    ⚠ 2026-09-23 语义修正：训练管线 relative_actions_processor 把臂维动作
    转成 delta（相对预测时刻 state，夹爪维除外）——chunk 臂维不是绝对目标，
    须加回本块预测时刻臂位 BASE_ARM（换块时随 job 更新）。
    """
    arm_tgt = BASE_ARM + np.concatenate([chunk[k][:7], chunk[k][8:15]])
    b = args.delta_max if budget is None else budget
    return cur + np.clip(arm_tgt - cur, -b, b)


def build_traj(chunk, cur0):
    """chunk → 整块 PVT 轨迹。逐点追踪式钳位推进；任一点越护栏即截断。

    返回 (traj, n_pts, final_p)；n_pts=0 表示首点就越护栏（勿下发）。
    """
    traj = Trajectory()
    traj.joint_names = ARM_NAMES
    pts, p = [], cur0.copy()
    for k in range(len(chunk)):
        arm_tgt = BASE_ARM + np.concatenate([chunk[k][:7], chunk[k][8:15]])
        new_p = p + np.clip(arm_tgt - p, -args.delta_max, args.delta_max)
        if float(np.max(np.abs(new_p - HOME))) > args.max_excursion:
            return traj, k, (p if k else None)   # p = 最后一个合法点
        p = new_p
        tp = TrajectoryPoint()
        tp.time_from_start_second = round((k + 1) * args.traj_dt, 4)
        vec = []
        for v in p:
            c = JointCommand()
            c.position = float(v)
            vec.append(c)
        tp.joint_command_vec = vec
        pts.append(tp)
    traj.points = pts
    return traj, len(chunk), p


# ── ② 干跑：只打印首轮计划，绝不下发 ──
if not args.do_exec:
    n_cmd = (n_steps + spc - 1) // spc
    print(f"\n== [干跑] 首轮 {n_steps} 步（合步 {spc} 步/指令 → {n_cmd} 条指令，"
          f"每条位移限幅 ±{BUDGET} rad）未下发任何命令 ==")
    for k in range(0, n_steps, spc):
        k_end = min(k + spc, n_steps)
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(k_end - 1, cur, BUDGET)
        drift = float(np.max(np.abs(tgt - HOME)))
        print(f"  指令[步{k}-{k_end - 1}]: 限幅后 {np.round(tgt, 3).tolist()}"
              f"\n        离起始位峰值 {drift * 1000:.0f} mrad"
              f"（护栏 {args.max_excursion * 1000:.0f}）")
        if ENV_LO is not None:
            _v = env_first_violation(chunk_abs_block(chunk))
            _m = ("全块通过" if _v is None else
                  f"⛔ 行{_v[0]} 维{_v[1]}={_v[2]:.3f} 越界 {_v[3]:.3f}")
            print(f"        包络预检: {_m}")
        if GRIP:
            desc = []
            for name, dim in GRIP["names"]:
                p = float(np.clip(chunk[k_end - 1][dim], 0.0, 100.0))
                w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
                desc.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm")
            print(f"        夹爪目标: {'  '.join(desc)}")
    print("\n[干跑] 未下发任何命令。加 --exec 真实执行。")
    WATCH.restore()
    if NAV is not None:
        NAV.resume()
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(0)

# ── ③ 连续循环（流水线：推理与执行重叠，消除轮间停顿）──
WATCH.pause()
input(f"\n⚠ 将连续 {args.rounds} 轮 × 每轮 {n_steps} 步真实驱动双臂"
      f"（合步 {spc} 步/指令，每条限幅 ±{BUDGET} rad，配速窗口 {args.pace}s"
      f"×1/{args.pace_div:g}，漂移护栏 ±{args.max_excursion} rad"
      + ("，夹爪下发开启）。\n" if GRIP else ")。\n")
      + "急停就绪后回车开始，循环期间随时按 q 退出...")
WATCH.resume()

infer_ms, grab_ms, round_ms, step_ms, track_err = (LatencyTracker(), LatencyTracker(),
                                                   LatencyTracker(), LatencyTracker(), [])
pace_hist = LatencyTracker()         # 配速 rad/s（追踪式，速度=导程÷(div×窗口)）
cmd_hist = None
aborted = False
sustain_cmds = 0                     # stale 路径（--late-action stale）的深尾"续航"条数
sust_last = 0                        # 上一轮续航条数（打进下轮表头，破水位一眼可见）
hold_windows = 0                     # hold 等待窗总数（迟到兜底核心遥测）
hold_last = 0                        # 上一轮 hold 窗数（打进下轮表头）
ramp_left = 0                        # 坡升剩余轮数（>0 时臂速按 0.5→1.0 线性限幅）
env_hits = 0                         # 包络护栏：连续越界轮数（干净轮清零）
env_trunc = 0                        # 包络护栏：累计截断轮数（汇总用）
res_bad = 0                          # 导程残差 ≥100 mrad 连续段数（告警用）
res_warned = False                   # 残差告警每跑只打一次
ramp_fired = 0                       # 坡升激活次数（多窗 hold 次数，汇总遥测）


class _PredictJob:
    """后台推理作业：GPU 推理与 SDK 阻塞执行重叠。

    只有 predict 放后台；SDK 取图/读关节/下发全部留在主线程，不碰 SDK
    线程安全。同一时刻只有一个 predict 在飞（消费完才启动下一个）。
    """

    def __init__(self, model, prompt):
        self._model, self._prompt = model, prompt
        self._done = threading.Event()
        self._res = None
        self._err = None
        self.dur_ms = 0.0

    def start(self, obs, state_n, state_arm):
        self._done.clear()
        self._err = None
        self.state_arm = state_arm   # 本块 delta 基准（预测时刻臂位）

        def _run():
            t = time.perf_counter()
            try:
                self._res = np.asarray(self._model.predict(
                    obs, prompt=self._prompt, state=state_n))
            except Exception as e:      # 主线程 result() 时再抛
                self._err = e
            self.dur_ms = (time.perf_counter() - t) * 1000
            self._done.set()

        threading.Thread(target=_run, daemon=True).start()

    def done(self):
        return self._done.is_set()

    def result(self):
        self._done.wait()
        if self._err is not None:
            raise self._err
        return self._res


def fresh_obs():
    """取图+读关节+夹爪换算+归一化（主线程，~22ms），耗时计入 grab_ms。

    返回 (obs, 归一化 state, 臂位原始值)——臂位原始值作该块 delta 基准随块走。
    """
    t_g = time.perf_counter()
    obs = grab_views(robot)
    st_raw = grip_state_cmd_override(
        state_from_joints(read_joints(robot, STATE_NAMES)))
    st_n = normalize_state(st_raw, ns)
    grab_ms.add((time.perf_counter() - t_g) * 1000)
    return obs, st_n, st_raw[ARM_SLICE]


job = _PredictJob(model, args.prompt)
job.start(*fresh_obs())              # 轮 0 的 chunk

t_loop0 = time.perf_counter()        # 任务秒表：轮头打印 t+s，对齐数据集 13s 看进度
near_sticky = False                  # 近态驻留标志（滞回进/出双阈值，10-13 泵动修复）

def _run_post_gate_fold():
    """门触发后同会话折叠 reset（+横移）。摆臂序列=g1_warm_core（与 warmup
    同源）。顺序=【摆臂先行，导航栈仍吊起】：--nav-suspend 下机器最安静，
    与 pick 推理期臂动平滑的条件完全一致——10-13 首跑先 resume 再摆臂，
    7 个导航进程苏醒风暴（GR3D 50%）正压在运动段上，用户目击摆臂卡顿；
    warmup 旧流程摆臂前有 ~10s 间隔故无此象。resume 挪到摆臂后、仍必在
    nav_straight_move 的 nav.init 之前（约束不变；栈未醒透的 WAIT_INITIALIZED
    由 nav_straight_move 限时重试兜住）。达标判定比 warmup 独立跑更严：
    warmup 仅告警，折叠臂未达标=place 输入 OOD，直接响亮失败（rc=1 编排器
    中止），不空臂进 place。"""
    print("\n== 折叠 reset（同会话）：摆臂 place 起始位（导航栈吊起保安静）"
          " → 恢复导航栈" + (" → 横移" if FOLD_MOVE else "") + " ==",
          flush=True)
    t_fold = time.perf_counter()
    t_warm_s = 0.0
    t_move_s = 0.0
    try:
        target, ns_w, _mode = g1_warm_core.load_target(
            args.fold_ckpt, args.fold_pose_file)
        mf_p = pathlib.Path(args.fold_ckpt) / "flashrt_deploy.json"
        cal = (json.loads(mf_p.read_text()).get("gripper", {})
               if mf_p.exists() else {})
        wmin, wmax = cal.get("width_min"), cal.get("width_max")

        def _die_fold(msg):
            raise RuntimeError(msg)

        st0 = g1_warm_core.read23(robot, g1_warm_core.STATE_NAMES,
                                  wmin, wmax, _die_fold)
        g1_warm_core.report(st0, target, g1_warm_core.STATE_NAMES)
        print(f"[fold] 摆臂（臂/头 {args.fold_speed} rad/s，腿 0.2）...",
              flush=True)
        g1_warm_core.send_warm(robot, target, args.fold_speed)
        cur = g1_warm_core.read23(robot, g1_warm_core.STATE_NAMES,
                                  wmin, wmax, _die_fold)
        g1_warm_core.report(cur, target, g1_warm_core.STATE_NAMES)
        arms_out = g1_warm_core.judge(cur, target, ns_w)
        t_warm_s = time.perf_counter() - t_fold   # 含首末两次读表+判定
        if arms_out:
            print(f"⛔ [fold] 臂维未达标: {arms_out}（warm 段 {t_warm_s:.1f}s）",
                  flush=True)
            return False
        if NAV is not None:
            NAV.resume()   # 摆臂后再醒导航栈；失败路径由外层尾部兜底 resume
        if FOLD_MOVE:
            t_m0 = time.perf_counter()
            if args.then_goto is not None:
                ok, detail = g1_warm_core.nav_goto_station(
                    robot, args.then_goto, args.move_timeout)
            else:
                ok, detail = g1_warm_core.nav_straight_move(
                    robot, args.then_move[0], args.then_move[1],
                    args.then_move[2], args.move_timeout)
            t_move_s = time.perf_counter() - t_m0
            if not ok:
                print(f"⛔ [fold] 横移失败: {detail}（warm {t_warm_s:.1f}s"
                      f" + 横移 {t_move_s:.1f}s）", flush=True)
                return False
        print(f"✅ [fold] 折叠完成（warm {t_warm_s:.1f}s"
              + (f" + 横移 {t_move_s:.1f}s" if FOLD_MOVE else "")
              + f"，总 {time.perf_counter() - t_fold:.1f}s）", flush=True)
        return True
    except Exception as e:
        print(f"⛔ [fold] 异常: {e!r}（warm {t_warm_s:.1f}s"
              f" + 横移 {t_move_s:.1f}s）", flush=True)
        return False


# ── 二相切换（--phase2-ckpt，2026-10-14）：fold 检测即后台载 place 构造，
# 折叠+横移窗口（~16.6s 实测）足够载入 6.9-10.6s + 首推建引擎 ~3s；折叠成功
# 后 join 载入线程 → 换 model/ns/prompt/包络/HOME 重基准/计数清零 → place 轮。
# 双构造共存前置实验 dual_model_swap_probe 全绿（统计判据：交替抖动落在单
# 构造自抖带内，位等本就不是本管线性质——原子/split-k 非确定性）──
def _load_env_arrays(ckpt_dir):
    """place ckpt 的关节包络盒（同启动段解析；无文件则沿用 pick 盒并告警）。"""
    _ef = pathlib.Path(ckpt_dir) / "joint_envelope.json"
    if not _ef.exists():
        print(f"[phase2] ⚠ {_ef} 不存在——place 轮沿用 pick 包络盒")
        return ENV_LO, ENV_HI
    _e = json.loads(_ef.read_text())
    _inf = float("inf")
    _lo = [(-_inf if v is None else float(v)) for v in _e["action_lo"]]
    _hi = [(_inf if v is None else float(v)) for v in _e["action_hi"]]
    print(f"[phase2] 包络盒换 {_ef.name}（来源 {_e.get('source', '?')}；"
          f"余量 ±{args.envelope_margin} rad）")
    lo = np.array(_lo, dtype=np.float32)
    hi = np.array(_hi, dtype=np.float32)
    _m = args.envelope_margin     # 余量同启动段：只放开臂维（夹爪维 ±inf 恒过）
    lo[:7] -= _m
    lo[8:15] -= _m
    hi[:7] += _m
    hi[8:15] += _m
    return lo, hi


def _switch_to_phase2():
    """全局换相进 place 轮。双构造路径（--phase2-ckpt）：place 构造已在
    启动段建好引擎，此处换 model/ns/包络；单模型路径（只给 --phase2-prompt）：
    同权重同引擎，只换 prompt+相位状态。失败返 False（调用方 rc=1 退）。"""
    global model, ns, HOME, ENV_LO, ENV_HI, job, t_loop0
    global infer_ms, grab_ms, round_ms, step_ms, track_err, pace_hist
    global cmd_hist, sustain_cmds, sust_last, hold_windows, hold_last
    global ramp_left, env_hits, env_trunc, res_bad, res_warned, ramp_fired
    global near_sticky
    if PH2["error"] is not None or (PH2["ckpt"] is not None
                                    and PH2["model"] is None):
        print(f"⛔ [phase2] place 构造不可用: {PH2['error']!r}", flush=True)
        return False
    if PH2["ckpt"] is not None:
        model = PH2["model"]
        ns = model._pipe.norm_stats
        ENV_LO, ENV_HI = _load_env_arrays(PH2["ckpt"])
    # 单模型路径（ckpt=None）：同引擎/同 ns/同包络——只换相位状态与 prompt
    # （job 在下方用 PH2["prompt"] 重建）
    args.grip_state_cmd = False     # place 轮不喂指令值：持物签名 33.5% 会把
    #                               释放序列锁死（10-12 判读；独立 place 段同款）
    if args.phase2_pace is not None:
        args.pace = args.phase2_pace        # place 相导航栈醒着（二相设计如
    if args.phase2_pace_div is not None:    # 此），pace 显式传大窗盖住水位
        args.pace_div = args.phase2_pace_div
    if GRIP is not None:            # 爪下发保持开（place 模型要发开爪指令），
        GRIP["sent"] = {}           # 只清指令历史/持物相位；标定 wmin/wmax 是
        GRIP["sent_peak"] = {}      # 物理量与 ckpt 无关，沿用
        GRIP["grasp_r"] = -10**9
    HOME = read_joints(robot, ARM_NAMES)   # place 起始位重基准（漂移护栏基线）
    # 计数器全清零（同 ③ 初始化清单）——place 轮统计独立于 pick
    infer_ms = LatencyTracker()
    grab_ms = LatencyTracker()
    round_ms = LatencyTracker()
    step_ms = LatencyTracker()
    track_err = []
    pace_hist = LatencyTracker()
    cmd_hist = None
    sustain_cmds = sust_last = hold_windows = hold_last = 0
    ramp_left = env_hits = env_trunc = res_bad = ramp_fired = 0
    res_warned = False
    near_sticky = False
    job = _PredictJob(model, PH2["prompt"])
    job.start(*fresh_obs())         # place 轮 0 的 chunk（fresh_obs 用新 ns）
    t_loop0 = time.perf_counter()
    print(f"\n{'=' * 14} [phase2] place 相开始（轮数重计，prompt："
          f"{PH2['prompt']}；pace {args.pace}s×1/{args.pace_div:g}）"
          f"{'=' * 14}", flush=True)
    return True


fold_active = False
phase2_active = False
r = -1
while True:
    r += 1
    if aborted or r >= args.rounds:
        break
    if FOLD_EVENT.is_set() and not phase2_active:
        if PH2 is None:
            print(f"[fold] 门折叠请求——第 {r} 轮前止步（已完成 {r}/{args.rounds} 轮）")
            break
        print(f"[fold] 门折叠请求——第 {r} 轮前切相（pick 已完成 {r} 轮；"
              + ("place 构造已在启动段建好，折叠后原地换相" if PH2["ckpt"]
                 is not None else "单模型同权重，折叠后只换 prompt 原地换相")
              + "）")
        if not _run_post_gate_fold():
            WATCH.restore()
            if NAV is not None:
                NAV.resume()
            print("SDK 关闭中...")
            robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
            os._exit(1)
        if not _switch_to_phase2():
            WATCH.restore()
            if NAV is not None:
                NAV.resume()
            print("SDK 关闭中...")
            robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
            os._exit(1)
        phase2_active = True
        FOLD_EVENT.clear()          # q 语义回 rc=2（place 相的 q=释放门/手动中止）
        r = -1                      # place 轮数从 0 重计
        continue
    t_r = time.perf_counter()
    chunk = job.result()             # 上轮执行期间启动的推理——此刻早已就绪
    BASE_ARM = job.state_arm         # 新块 delta 基准 = 该块预测时刻臂位
    infer_ms.add(job.dur_ms)
    if hold_last:
        _sus = f"hold {hold_last} 窗（等推理：hold 最后目标单调收敛，不喂深尾）"
    elif sust_last:
        _sus = (f"续航 {sust_last} 条 ⚠ 破推理水位（--late-action stale 深尾模式："
                f"计划垃圾、臂往复风险）")
    else:
        _sus = "零等待"
    print(f"\n── 轮 {r} | t+{time.perf_counter() - t_loop0:6.1f}s | "
          f"推理 {job.dur_ms:.0f} ms，与上轮执行重叠，{_sus} ──")
    if ENV_LO is not None:
        _v = env_first_violation(chunk_abs_block(chunk))
        if _v is None:
            env_hits = 0
        else:
            _r, _j, _val, _bnd = _v
            env_hits += 1
            env_trunc += 1
            _side = "上" if _val > _bnd else "下"
            print(f"⛔ 包络护栏：指令行{_r} 维{_j}={_val:.3f} 越{_side}界 "
                  f"{_bnd:.3f}——截到行{_r}前缀（连续 {env_hits}/"
                  f"{args.envelope_abort_n} 轮）")
            if _r == 0:
                print("⛔ 行 0 即越界，无处可截——停止循环（机械臂留在原地）")
                aborted = True
                continue
            if env_hits >= args.envelope_abort_n:
                print("⛔ 连续越界=感知漂移（场景认错了），继续跑只会反复撞桌"
                      "——停止循环（机械臂留在原地）")
                aborted = True
                continue
            chunk = chunk[:_r]
    if args.chunk_mode == "traj":
        cur0 = read_joints(robot, ARM_NAMES)
        traj, n_pts, final_p = build_traj(chunk, cur0)
        if n_pts == 0:
            print("⛔ 轨迹首点越护栏，停止循环")
            aborted = True
        else:
            print(f"  轨迹 {n_pts} 点 × {args.traj_dt * 1000:.0f} ms 下发...")
            gp = send_grip(robot, chunk[-1])
            if gp:
                print(f"  夹爪: {gp}")
            t_s = time.perf_counter()
            st = robot.execute_joint_trajectory(traj, is_blocking=False)
            if r < args.rounds - 1:
                job.start(*fresh_obs())   # 推理藏进轨迹执行期（sleep 放 GIL）
            t_end = t_s + n_pts * args.traj_dt
            while time.perf_counter() < t_end:
                time.sleep(0.05)          # 主线程小睡，GIL 让给推理线程
            tss = robot.check_trajectory_execution_status([])
            step_ms.add((time.perf_counter() - t_s) * 1000)
            ach = read_joints(robot, ARM_NAMES)
            if final_p is not None:
                track_err.append(float(np.max(np.abs(ach - final_p))) * 1000)
            d_cmd = (float(np.max(np.abs(final_p - cmd_hist))) * 1000
                     if cmd_hist is not None and final_p is not None else 0.0)
            if final_p is not None:
                cmd_hist = final_p.copy()
            print(f"  轨迹完成: {st} | 执行 {step_ms[-1]:.0f} ms | "
                  f"末端偏差 {track_err[-1] if track_err else float('nan'):.1f} mrad | "
                  f"|Δcmd| {d_cmd:.1f} mrad | PVT 状态 "
                  f"{[s.name for s in tss] or '未上报'}")
            if any(s.value != 2 for s in tss):   # ≠ COMPLETED
                print("⛔ PVT 状态非 COMPLETED，停止循环")
                aborted = True
        round_ms.add((time.perf_counter() - t_r) * 1000)
        if round_ms[-1] > 1:
            print(f"  轮耗时 {round_ms[-1]:.0f} ms（重规划频率 {1000 / round_ms[-1]:.1f} Hz）")
        continue
    # 消费配额 n_steps 步后若新块未就绪：
    #   hold（默认）→ hold 最后目标单调收敛等推理（轮询早退、夹爪行不变、
    #     永不喂深尾步；出处见文件头——PI RTC 执行侧调度层）；
    #   --late-action stale → 旧 chunk 剩余步"续航"追踪（深尾=计划垃圾，
    #     原独立 A 脚本行为，存档于 legacy/run_g1_loop_stale.py）。
    # 两条路都受 ±BUDGET 限幅 + 漂移护栏，不放大行程风险
    sust_before = sustain_cmds
    hold_before = hold_windows
    k = 0
    n_steps_r, spc_r = n_steps, spc   # 本轮配额/步宽，每轮重置=退出近态自动回全宽
    while k < len(chunk):
        k_end = min(k + spc_r, len(chunk))   # 本条指令消费 chunk 步 [k, k_end)
        cur = read_joints(robot, ARM_NAMES)
        tgt = plan_step(k_end - 1, cur, BUDGET)
        drift = float(np.max(np.abs(tgt - HOME)))
        if drift > args.max_excursion:
            print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                  f"{args.max_excursion * 1000:.0f}，停止循环（机械臂留在原地）")
            aborted = True
            break
        # 近态判定用全宽导程（先宽后缩，触发语义与不开 --near-n 完全一致）；
        # 滞回：进态用 near_gap、退态用 near_exit（默认 2×near_gap）——单阈值
        # 时增量在阈值两侧的接近段逐轮窄↔宽翻转=1.59×↔1.0× 速度方波往复
        # 摆动（10-13 loop_20261013_100407 轮 9-21 实录），滞回后整段稳定驻留；
        # 命中且开近窄轮时仅本轮首条指令降档，轮首已重置=退出自动回全宽
        gap0 = float(np.max(np.abs(tgt - cur)))
        near_gate = args.near_exit if args.near_exit else args.near_gap * 2.0
        near_thr = near_gate if near_sticky else args.near_gap
        near_cmd = (args.near_div >= 1.0 and gap0 < near_thr
                    and r - GRIP["grasp_r"] >= args.near_bypass)
        near_sticky = near_cmd
        if near_cmd and near_n_eff and k == 0:
            n_steps_r = max(1, min(near_n_eff, len(chunk)))
            spc_r = min(spc, n_steps_r)
            if spc_r < k_end - k:
                k_end = min(k + spc_r, len(chunk))
                tgt = plan_step(k_end - 1, cur, BUDGET)
                gap0 = float(np.max(np.abs(tgt - cur)))
        ramp_f = 1.0                       # 本窗坡升系数（1.0=不限速）
        cmd_delta = (tgt - cmd_hist) if cmd_hist is not None else np.zeros_like(tgt)
        cmd_hist = tgt.copy()
        gp = send_grip(robot, chunk[k_end - 1])   # 夹爪伴随后臂目标，发在计时区外
        if gp:
            print(f"  夹爪: {gp}")
        t_s = time.perf_counter()
        if args.settle:
            # 旧步进-停走：阻塞到位再发下一条 → 速度曲线锯齿（"卡卡的"根因）
            st = robot.set_joint_positions(tgt.tolist(), joint_names=ARM_NAMES,
                                           is_blocking=True, speed_rad_s=args.speed,
                                           timeout_s=10.0)
        else:
            # 追踪式 v4（合步 + 配速系数）：速度=导程÷(pace-div×窗口)，每窗按
            # 比例走导程（2=半程 → 臂恒在途永不到点，窗口末被下一轮指令滑行
            # 重定向，速度不过零，无到达门；1=全程 → 每窗走完导程，臂速翻倍）。
            # 闭环每轮从真实状态重规划，滞后自校正。窗口自发令前起算（推理
            # 重叠在内）。0.6 提速实测更抖（每点全停，冲击∝速度），平滑靠
            # 永不到点而非提速
            div_eff = args.pace_div
            if near_cmd:
                div_eff = args.near_div   # 近距阻尼（抓取后旁路窗内跳过=后撤全速）
            pace_v = max(V_MIN, min(gap0 / (div_eff * args.pace), args.speed))
            if ramp_left > 0:
                # 坡升：多窗 hold（臂真减速过）后的再起步柔化，0.5→1.0 线性恢复
                ramp_f = 0.5 + 0.5 * (args.hold_ramp - ramp_left) / args.hold_ramp
                pace_v *= ramp_f
                ramp_left -= 1
            robot.set_joint_positions(tgt.tolist(), joint_names=ARM_NAMES,
                                      is_blocking=False, speed_rad_s=pace_v)
            pace_hist.add(pace_v)
            if k == 0 and r < args.rounds - 1:
                # 下一块推理藏进本指令执行期（观测取自滑行中途，即真实当前态）
                job.start(*fresh_obs())
            while time.perf_counter() - t_s < args.pace:
                time.sleep(0.02)          # 主线程小睡，GIL 让给推理线程
            st = "ControlStatus.SUCCESS(tracked)"
        step_ms.add((time.perf_counter() - t_s) * 1000)
        ach = read_joints(robot, ARM_NAMES)
        err = float(np.max(np.abs(ach - tgt))) * 1000
        track_err.append(err)
        res_bad = res_bad + 1 if err >= 100.0 else 0
        if res_bad >= 3 and not res_warned:
            res_warned = True
            print("⚠ 导程残差 ≥100 mrad 持续 3 段——疑似爪尖受阻（顶桌/卡物），"
                  "注意观察，必要时按 q 停")
        tag = f"指令[步{k}-{k_end - 1}]" + ("·续航" if k >= n_steps_r else "") \
            + ("·近阻尼" if div_eff != args.pace_div else "") \
            + ("·窄轮" if n_steps_r != n_steps else "") \
            + (f"·坡升{ramp_f:.2f}" if ramp_f < 1.0 else "")
        if k >= n_steps_r:
            sustain_cmds += 1
        print(f"  {tag}: |Δcmd| "
              f"{float(np.max(np.abs(cmd_delta))) * 1000:5.1f} mrad | "
              f"执行 {step_ms[-1]:5.0f} ms | 导程残差 {err:4.1f} mrad | {st}")
        if not str(st).startswith("ControlStatus.SUCCESS"):
            print("⛔ 下发非 SUCCESS，停止循环")
            aborted = True
            break
        k = k_end
        # 配额消费完：新块就绪（或已是最后一轮）→ 立即换块零等待
        if k >= n_steps_r and (r == args.rounds - 1 or job.done()):
            break
        if k >= n_steps_r:
            if args.late_action == "stale":
                continue               # stale：旧 chunk 深尾续航（原 A 行为）
            # ── hold：等推理，单调收敛到本块已消费段终点（chunk[n_steps-1]）──
            # 与 A 的深尾续航差异：目标不动（几何逼近永不过冲→无往复）、夹爪行
            # 不变（chg 阈值抑制重发→无翻摆）、逐 10ms 轮询（推理落地立即换块，
            # 不烧满剩余窗）。护栏/限幅/配速律照旧，hold 窗是纯墙钟成本
            hold_n = 0
            while not job.done() and not aborted and not FOLD_EVENT.is_set():
                hold_n += 1
                hold_windows += 1
                if hold_n > args.hold_max:
                    print(f"⛔ 推理连续 {args.hold_max} 窗未归——判定挂死，"
                          "停止循环（机械臂留在原地）")
                    aborted = True
                    break
                cur = read_joints(robot, ARM_NAMES)
                tgt_h = cur + np.clip(tgt - cur, -BUDGET, BUDGET)
                drift = float(np.max(np.abs(tgt_h - HOME)))
                if drift > args.max_excursion:
                    print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                          f"{args.max_excursion * 1000:.0f}，停止循环（机械臂留在原地）")
                    aborted = True
                    break
                cmd_delta = ((tgt_h - cmd_hist) if cmd_hist is not None
                             else np.zeros_like(tgt_h))
                cmd_hist = tgt_h.copy()
                gp = send_grip(robot, chunk[min(n_steps_r, len(chunk)) - 1])
                if gp:
                    print(f"  夹爪: {gp}")
                t_s = time.perf_counter()
                gap0 = float(np.max(np.abs(tgt_h - cur)))
                div_eff = args.pace_div
                if (args.near_div >= 1.0 and gap0 < near_thr
                        and r - GRIP["grasp_r"] >= args.near_bypass):
                    div_eff = args.near_div   # 近距阻尼同主路径（同轮同滞回口径）
                pace_v = max(V_MIN, min(gap0 / (div_eff * args.pace), args.speed))
                robot.set_joint_positions(tgt_h.tolist(), joint_names=ARM_NAMES,
                                          is_blocking=False, speed_rad_s=pace_v)
                pace_hist.add(pace_v)
                while (time.perf_counter() - t_s < args.pace
                       and not job.done()):
                    time.sleep(0.01)      # 轮询早退：推理一落地立即出窗
                step_ms.add((time.perf_counter() - t_s) * 1000)
                ach = read_joints(robot, ARM_NAMES)
                err = float(np.max(np.abs(ach - tgt_h))) * 1000
                track_err.append(err)
                res_bad = res_bad + 1 if err >= 100.0 else 0
                if res_bad >= 3 and not res_warned:
                    res_warned = True
                    print("⚠ 导程残差 ≥100 mrad 持续 3 段——疑似爪尖受阻"
                          "（顶桌/卡物），注意观察，必要时按 q 停")
                print(f"  指令[hold {hold_n}]: |Δcmd| "
                      f"{float(np.max(np.abs(cmd_delta))) * 1000:5.1f} mrad | "
                      f"执行 {step_ms[-1]:5.0f} ms | 导程残差 {err:4.1f} mrad | "
                      f"ControlStatus.SUCCESS(tracked)")
            if aborted:
                break
            if args.hold_ramp > 0 and hold_n >= 2:
                # 臂在 hold 中真减速过（≥1 个完整额外窗收敛）→ 后 N 轮限速再起步
                ramp_left = args.hold_ramp
                ramp_fired += 1
            break                          # 推理就绪 → 正常换块（零深尾）
    round_ms.add((time.perf_counter() - t_r) * 1000)
    if round_ms[-1] > 1:
        print(f"  轮耗时 {round_ms[-1]:.0f} ms（重规划频率 {1000 / round_ms[-1]:.1f} Hz）")
    sust_last = sustain_cmds - sust_before
    hold_last = hold_windows - hold_before

# ── ④ 汇总 ──
fold_active = FOLD_EVENT.is_set()   # 门触发折叠：汇总后同会话复位（+横移）
if GRIP and not fold_active:
    time.sleep(8.0)   # 夹爪反馈滞后 ~6.3s（2026-09-23 实测），留足再读终态
    for name, _ in GRIP["names"]:
        gs = robot.get_gripper_state(getattr(G1JointGroup, name))
        if gs is not None:
            print(f"夹爪终态 {name}: {gs.width * 1000:.1f} mm (moving={gs.is_moving})")
fin = read_joints(robot, ARM_NAMES)
exc = float(np.max(np.abs(fin - HOME))) * 1000
print(f"\n== 汇总（{len(round_ms)} 轮 / {len(step_ms)} 步）==")
print(f"总用时: {time.perf_counter() - t_loop0:.1f} s"
      f"（数据集单条任务 13s / 390 步参照）")
if len(infer_ms):
    print(f"推理: {pstats(infer_ms)}")
if len(grab_ms):
    print(f"取图+读关节: {pstats(grab_ms)}")
if len(step_ms):
    print(f"单指令执行({spc} 步合步"
          + (f"，近窄 {near_n_eff} 步" if near_n_eff else "") + f"): {pstats(step_ms)}")
if len(round_ms):
    print(f"整轮(重规划周期): {pstats(round_ms)}")
if track_err:
    print(f"导程残差: mean {np.mean(track_err):.1f} | max {max(track_err):.1f} mrad"
          f"（配速系数 {args.pace_div:g}"
          + (f"，近距 {args.near_gap*1000:.0f} mrad 内阻尼 {args.near_div:g}"
             f"（退出门限 {(args.near_exit or args.near_gap * 2.0) * 1000:.0f} mrad，"
             "抓取后旁路 "
             f"{args.near_bypass} 轮）"
             + (f"，近窄轮 {near_n_eff} 步" if near_n_eff else "")
             if args.near_div >= 1.0 else "")
          + (f"；包络护栏截断 {env_trunc} 轮" if ENV_LO is not None else "")
          + "：残差≈导程×(1-1/div) 属预期；"
          "≈1 到点，残差≈0 且臂停=到点干等推理）")
if len(pace_hist):
    print(f"配速: p50 {pace_hist.percentile(0.5):.3f} | "
          f"p95 {pace_hist.percentile(0.95):.3f} | max {pace_hist.max():.3f} rad/s"
          f"（导程÷({args.pace_div:g}×窗口 {args.pace}s)，上限 {args.speed}，下限 {V_MIN}）")
    print("抖动判读：|Δcmd| 快速变号=抖动；持续同号=漂移（由护栏兜底）")
if hold_windows:
    print(f"hold 等待窗: {hold_windows} 个（推理迟到，hold 最后目标单调收敛；"
          f"stale 同场景为深尾续航=计划垃圾；上限 {args.hold_max} 窗未触发挂死判定）")
if sustain_cmds:
    print(f"深尾续航: {sustain_cmds} 条（--late-action stale 路径：配额外消费旧 chunk 步）")
if args.hold_ramp > 0:
    print(f"坡升触发: {ramp_fired} 次（hold≥2 窗后 {args.hold_ramp} 轮限速 0.5→1.0 再起步；"
          f"1 窗早退的 hold 不触发，臂未减速无需柔化）")
print(f"最终偏离起始位: {exc:.0f} mrad（护栏 {args.max_excursion * 1000:.0f} mrad）"
      + (" ⛔ 护栏触发过" if aborted else ""))

# ── ③b 门触发折叠：同会话复位+横移，失败 rc=1（编排器拦下不进 place）──
if fold_active and not _run_post_gate_fold():
    WATCH.restore()
    if NAV is not None:
        NAV.resume()
    print("SDK 关闭中...")
    robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
    os._exit(1)

WATCH.restore()
if NAV is not None:
    NAV.resume()
print("SDK 关闭中...")
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
os._exit(0)   # SDK 残留线程，干净退出
