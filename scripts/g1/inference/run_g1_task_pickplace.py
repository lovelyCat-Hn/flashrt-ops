#!/usr/bin/env python
"""pick→复位→横移→place 主编排器。

【默认二相档（2026-10-14，--until full）】全流程单 loop 子进程：pick 门触发
→ 注 'r' → 折叠+横移（同会话，10-13 折叠档）→ place 构造已在折叠窗口内
后台载好 → 原地切相跑 place 轮 → 释放门触发注 'q'。编排器决策层（两道
PhaseGate v4 + 注入 + 预算）不变，执行全在握 SDK 会话的 loop 进程——
会话进程私有，编排器没有手。

【折叠档（10-13，--until reset/move/place）】reset+横移并入 pick 会话：门触发后编排器注入 'r'，
loop 本轮止步、同会话摆臂到 place 起始位姿+GalbotNavigation 横移（省 warmup
子进程 ~4-5s）；loop 新增 --post-gate（默认 none=原行为）。--no-fold 退回
旧四段子进程路径。决策（门/注入/预算）在本编排器，执行在握 SDK 会话的
loop 进程——会话进程私有，编排器没有手。

流程（用户 2026-10-13 定义，合成任务=pick_place）：
  ① pick   起子进程 run_g1_loop.py（pick ckpt+句+--grip-state-cmd；配速/near 系
           参数走 config 或 --pick-args 透传）。
           【结束判据=本编排器的灵魂，v4 物理持物带】loop 每轮爪行带【实 X.Xmm】
           物理回读（10-13 爪标定：回读新鲜 p50 10ms）；全部监听侧实宽连续
           --pick-stable-s 落在持物带 [--held-min-mm,--held-max-mm]（默认
           12-80；v4.1 再交指令≤--pick-cmd-close-pct，治开合过渡态实宽
           落带误触发 141150）→ 向子进程 stdin 写 'q'（走 loop
           已验证的 q 退出路径，nav-suspend 自动 CONT）。
           ⚠ 指令闭合≠抓住（143656 实证空爪闭合指令 0-3.5% 照样持续，
           v2/v3 的穿越门+静默兜底全被空爪骗到）——门只认物理实宽，
           读不到回读=按未持物处理（宁可超时中止，不误触发）。
  ② reset  【折叠档（默认）】不再起子进程：门触发 → 注入 'r' → loop 同会话
           摆臂到 place 起始位姿（g1_warm_core 与 warmup 同源序列；臂/头
           --reset-speed 默认 0.8 rad/s、腿 0.2；不发任何夹爪指令=保住
           爪中物体）。达标判据比 warmup 独立跑更严：臂维归一化 [-1,1]
           外非零 → rc=1 中止（不空臂进 place）。折叠期 q 急停有效
           （rc=4=折叠中止，编排器拦下）。
           【--no-fold 旧路径】起子进程 g1_pose_warmup.py --pose-file
           <place 起始位姿> --no-grip --speed <reset_speed>
           --init-settle 0.5（pick 子进程刚释放控制器、SDK 栈是热的）。
           ⚠ --no-grip 是命门（两档同理）：物体在爪里物理撑着 ~33.5%
           （恰为 place 起始语义），闭爪到 0%=把东西挤掉。
  ③ move   已并入复位会话：折叠档=pick 进程横移（--post-gate warm-move），
           --no-fold 档=warmup 达标后 --then-move 同会话横移。API=官方
           GalbotNavigation.move_straight_to 四步（nav.init → switch_
           controller〔只切 chassis 组不碰臂〕→ move_straight_to 阻塞=
           到位才 True → stop_navigation；7 元 Pose 相对 base_link/odom
           系，免定位、无碰撞检查、不走全局规划；⚠ move_straight_to 不在
           GalbotRobot 本体绑定上，10-13 AttributeError 案）。
           --until reset 时不带横移。
           g1_base_move.py 保留=手动工具 + 预检（--exec 前置 --probe-only
           ~5s 零运动，SDK 栈不健康 pick 前红灯；10-13 段错误案=该段代码
           漏了 robot.init()，已补）。
           距离以同事为准（用户 10-13 指示）：nav_points.json 09-10 录制的
           pick→place 位移 = x+0.057 / y+1.487 / yaw+1.068°（≈1.49m 左移）。
           ⚠ base_link +y=左 是 SDK 头文件明文（x前/y左/z上），真机首跑仍建议
           低速先核一次方向；同事本体走的是 SLAM navigate_to_goal 绝对点
           （见 project_yuqiz behavior_tree），我们改用相对横移=不依赖地图
           与定位、场景挪动免疫，代价是里程计漂移（短距无碍）。
  ④ place  【二相档（2026-10-14 默认，--until full）】不再起子进程：
           【单模型，现默认】0930 权重实为 pick+place 4 任务合并训练
           （train_config wandb notes 实锤，目录名 only_pick 是部署命名），
           place 不换权重——折叠完成后同引擎只换 prompt 切相（HOME 重基准/
           计数清零/place 轮 grip-state-cmd 自动关），无第二构造载入。
           【--dual-load 旧档】loop 开局与 pick 一起载入 place ckpt+建引擎
           （v2 开局同载；v1 藏折叠窗口 26.0s>16.8s 漏出 ~9s 已废弃），
           折叠后换 model/ns/prompt/包络切相——省 place 冷启动 ~22-28s
           （10-13 晚 180822 实录：start_controller 还会偶发超时）。
           【--until place 旧路径】起子进程 run_g1_loop.py（place ckpt+句；
           【不带】--grip-state-cmd——33% 持物起步喂指令值反而与训练分布
           相反，10-12 判读）。
           结束判据（与 pick 门同源 v4，两档同一套）：全部监听侧实宽连续
           --place-stable-s ≥ --open-min-mm（默认 60，放开≈90-120mm）=
           松爪完成 → 'q' 收尾；二相下释放门布防于 loop「折叠完成」行之后，
           触发写 q，整体成功=释放门触发 且 rc∈{0,2}；或 --place-timeout
           预算兜底。

用法（bring-up 期逐段验证，勿直接全流程）：
  干跑（只打印各段命令）:   python run_g1_task_pickplace.py
  只跑 pick:                python run_g1_task_pickplace.py --exec --until pick
  跑到复位为止:             python run_g1_task_pickplace.py --exec --until reset
  跑到横移为止:             python run_g1_task_pickplace.py --exec --until move
  全流程（默认=full 二相）:  python run_g1_task_pickplace.py --exec
  旧四段路径（place 独立子进程，A/B 用）:
                            python run_g1_task_pickplace.py --exec --until place --no-fold
  pick 带近态实验参数:      ... --pick-args "--near-n 15 --near-div 1.43"

安全：--exec 真实运动，不进配置（与四脚本约定一致）。任一段失败=整个流程
中止（宁可不 place，不自动跑"空爪 place"的错误剧本）。子进程即使被 SIGKILL，
nav-suspend 有独立守护网兜底 CONT。
"""
import argparse
import faulthandler
import pathlib
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

faulthandler.enable()   # 原生崩溃（SDK segfault 等）打印 Python 级位置，不再无声死

HERE = pathlib.Path(__file__).resolve().parent
HOLY = HERE.parent.parent
LOOP_PY = HERE / "run_g1_loop.py"
WARMUP_PY = HERE.parent / "g1" / "g1_pose_warmup.py"
MOVE_PY = HERE.parent / "g1" / "g1_base_move.py"   # move 段/预检子进程

# 与 config/g1.toml [run] 同源（10-12 pick 在役；place 三键存档）
PICK_CKPT = HOLY / "models/pi05_g1_onlypick_deploy"
PLACE_CKPT = HOLY / "models/pi05_g1_place_deploy"
PLACE_POSE = PLACE_CKPT / "episode_start_task0.json"    # 33% 持物起始语义
PICK_PROMPT = "Left arm pick up A. Right arm pick up A."
PLACE_PROMPT = ("Left arm places A in the top-left corner. "
                "Right arm places A in the top-left corner.")

# 底盘横移距离：以同事为准（用户 2026-10-13 指示"以同事的距离为准"）。
# 来源 /home/galbot/project_yuqiz/g1_pick_place/behavior_tree/.../nav_points.json
# （09-10 录制）：pick=[0,0,0] yaw0，place=[0.0569, 1.4869, ...] yaw≈1.068°
# → pick→place 相对位移 = x+0.057 / y+1.487 / yaw+1.068°（≈1.49m 左移，非 1m）。
SHIFT_X = 0.057
SHIFT_Y = 1.487
SHIFT_YAW_DEG = 1.068

# 爪行物理实宽（loop send_grip 行尾 [实 X.Xmm]；[实 ?]=回读失败）。
# 字符类排除对方侧大写字母+ '['：防某侧缺括号时滑段错配到另一侧的值
PHYS_VAL = {"R": re.compile(r"\bR\s+[\d.]+%[^[\nL]*\[实\s*([\d.]+|\?)mm"),
            "L": re.compile(r"\bL\s+[\d.]+%[^[\nR]*\[实\s*([\d.]+|\?)mm")}
# 爪行指令百分数（行内 R 71.1%→85mm 的第一个数=模型指令目标%）。
# v4.1 交集判据用：开合过渡态（开爪爬坡）实宽会落在持物带内（141150 实录：
# R 开爪爬坡停在 15.5mm×4 拍 ≈3.7s>2s 窗，L 已持物→门误触发折叠），但该态
# 指令必然高（71-83%）；真持物时指令≈0-5%。物理带∩指令低=四态全覆盖。
CMD_VAL = {"R": re.compile(r"\bR\s+([\d.]+)%"),
           "L": re.compile(r"\bL\s+([\d.]+)%")}


class PhaseGate:
    """任务相位门 v4.1（2026-10-15：物理带 ∩ pick 指令低，交集判据）。

    演进史一句话：v1 实现_bug MISS → v2/v3 指令穿越+臂静默（基于「R 从未
    穿越」误读，越做越重）→ 143656 实证空爪闭合=指令 0-3.5% 照样持续，
    穿越/闭合/静默全被空爪骗到 → v4 只认物理实宽（loop send_grip 行尾
    【实 X.Xmm】回读，10-13 爪标定：回读新鲜 p50 10ms）→ **141150 实录
    v4 盲区：开合过渡态实宽落持物带**（R 开爪爬坡 4→15.5mm 停 4 拍
    ≈3.7s>2s 窗，L 已持物 34mm→全侧在带→门误触发折叠，R 实际还没抓）
    → v4.1 pick 门加指令维：物理在带 ∩ 指令≤close_max_pct。

    四态覆盖（v4.1）：
      空爪闭合 0-2mm   物理✗              → 不触发
      开合过渡（141150）物理✓ 指令 71-83%✗ → 不触发（v4 在此翻车）
      真持物 30-40mm   物理✓ 指令 0-5%✓   → 触发
      掉落/放开        物理✗              → 不触发
    v2/v3 指令门单独判死于「空爪闭合指令 0-3.5% 与持物同域」——与物理带
    取交集后互补（物理排除空爪闭合，指令排除开合过渡）。

    完成判据 = 全部监听侧物理宽度连续 stable_s 落在判据带：
      pick（done_open=False）持物带 [--held-min-mm, --held-max-mm]
           默认 12-80mm：空爪闭合≈0-2mm 不进带→不触发→段预算中止
           （=「没抓住就别停」，用户定案）；持物≈30-40mm（数据集撑 33.5%
           语义）；张开着也不进带（>80）。再交指令≤close_max_pct
           （0=关闭退回纯物理带=v4）。
      place（done_open=True）释放带 ≥ --open-min-mm（默认 60mm；
           放开≈90-120mm）。指令维不参与（释放=指令高，交叉方向相反，
           释放带与开爪爬坡的重叠本就无害——臂已静默，早触发 ≤1 轮）。
    实宽/指令读不到（[实 ?]/行缺失/无 %）→ 按不满足处理（安全向：宁可
    超时中止不误触发）。判据带连续窗，任何一轮出带即清零重计。
    """

    def __init__(self, sides, done_open, stable_s,
                 held_min_mm=12.0, held_max_mm=80.0, open_min_mm=60.0,
                 close_max_pct=0.0):
        self.sides = list(sides)
        self.done_open = done_open          # False=pick（完成=持续持物）
        self.stable_s = stable_s
        self.held_min_mm, self.held_max_mm = held_min_mm, held_max_mm
        self.open_min_mm = open_min_mm
        self.close_max_pct = close_max_pct  # pick 指令上限%（0=v4 纯物理带）
        self.last_mm = {}                   # 侧 → 最近物理实宽 mm（None=未知）
        self.last_cmd = {}                  # 侧 → 最近指令 %（None=未知）
        self.t_ok = None                    # 全侧在带连续窗起点
        self.fired = False

    def _in_band(self, v, cmd=None):
        if v is None:
            return False                    # 回读缺失=不满足（安全向）
        if self.done_open:
            return v >= self.open_min_mm
        if not (self.held_min_mm <= v <= self.held_max_mm):
            return False
        if self.close_max_pct > 0:
            # 开合过渡消歧：开爪爬坡实宽可能落带内但指令必高；指令读不
            # 到=不满足（安全向，与实宽缺失同纪律）
            if cmd is None or cmd > self.close_max_pct:
                return False
        return True

    def feed(self, line, now):
        for s in self.sides:
            m = PHYS_VAL[s].search(line)
            if m:
                # [实 ?]=回读失败→显式归 None（不沿用旧宽：陈旧值会让
                # 判据带窗假性连续）；窗随即清零=安全向
                self.last_mm[s] = None if m.group(1) == "?" else float(m.group(1))
            mc = CMD_VAL[s].search(line)
            if mc:                      # 与实宽同纪律：非爪行（轮头/指令/
                self.last_cmd[s] = float(mc.group(1))   # 耗时行）不得冲掉
                #       上次指令——v4.1 首版此处无条件重写，每轮 4/5 非爪行
                #       把 last_cmd 冲成 None→判据恒 False→143851 真机不触发
                #       （全行流重演破案：last_mm 对、last_cmd 全 None）
        if self.fired:
            return True
        if all(self._in_band(self.last_mm.get(s), self.last_cmd.get(s))
               for s in self.sides):
            if self.t_ok is None:
                self.t_ok = now
        else:
            self.t_ok = None
        if self.t_ok is not None and now - self.t_ok >= self.stable_s:
            self.fired = True
        return self.fired


def run_child(cmd, gate, timeout, label, stop_key="q", ok_rc=(0, 2),
              gate2=None, gate2_timeout=45.0):
    """起子进程（-u 防管道缓冲延迟检测），逐行回显+喂门。

    门触发（gate.fired）→ stdin 写 stop_key（'q'=立即退；'r'=loop 折叠
    reset+横移后退）；timeout/Ctrl-C 恒写 'q'（中止语义，绝不触发折叠）；
    再不退 → SIGINT→SIGKILL 升级（nav-suspend 守护网兜底 CONT）。

    gate2（二相模式，2026-10-14）：pick 门注入 'r' 后进程不退——折叠段跑完
    （stdout 出现「折叠完成」行）即布防释放门（done_open 带）；释放门触发
    → 写 'q' 收尾；place 预算超时 → 写 'q' 中止。成功判据 reason=="释放门"
    且 rc∈{0,2}——折叠失败（rc=1）、释放前 q（rc=2 无释放门）、超时都不放行。
    返回 (ok, 停止原因)。
    """
    print(f"[{label}] $ {' '.join(cmd)}")
    p = subprocess.Popen([sys.executable, "-u"] + cmd,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    p.stdin.write("\n")          # 预喂回车=loop 的"急停就绪回车开始"确认
    p.stdin.flush()

    def _pump():
        # 时间戳在泵线程打（产出时刻）：主线程回显/落盘一旦滞后，
        # 滞后会 1:1 顶晚静默窗起点（10-13 133625 实测触发晚 4.8s 的根因）
        for l in p.stdout:
            lines.put((time.time(), l))
        lines.put(None)          # EOF 哨兵

    lines = queue.Queue()
    threading.Thread(target=_pump, daemon=True).start()

    t0, reason = time.time(), None
    gate2_armed, t_place0 = False, None
    try:
        while True:
            # 超时检查放循环顶：只挂在 queue.Empty 上会被密集输出饿死
            if time.time() - t0 > timeout:
                reason = f"超时 {timeout:.0f}s"
                print(f"  [{label}] ⚠ {reason} → 仍先试 'q' 优雅退出")
                try:
                    p.stdin.write("q\n")   # 恒 q：超时=中止，绝不触发折叠
                    p.stdin.flush()
                except Exception:
                    pass
                break
            try:
                item = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            stamp, line = item
            sys.stdout.write(line)
            sys.stdout.flush()
            if gate is not None and not gate.fired and gate.feed(line, stamp):
                reason = "门触发"
                if gate.done_open:
                    why = (f"实宽≥{gate.open_min_mm:.0f}mm（持续放开）"
                           f"×{gate.stable_s:.1f}s")
                else:
                    why = (f"实宽 {gate.held_min_mm:.0f}-{gate.held_max_mm:.0f}mm"
                           f"（持续持物）×{gate.stable_s:.1f}s")
                print(f"  [{label}] ✓ 门触发〔{why}〕 → stdin '{stop_key}' 请求退出")
                try:
                    p.stdin.write(stop_key + "\n")
                    p.stdin.flush()
                except Exception:
                    pass
            if (gate2 is not None and reason == "门触发" and not gate2_armed
                    and "折叠完成" in line):
                gate2_armed = True
                t_place0 = time.time()
                print(f"  [{label}] 折叠完成——布防释放门（实宽≥"
                      f"{gate2.open_min_mm:.0f}mm × {gate2.stable_s:.1f}s，"
                      f"预算 {gate2_timeout:.0f}s）")
            elif (gate2 is not None and gate2_armed and not gate2.fired
                  and reason == "门触发"):
                if gate2.feed(line, stamp):
                    reason = "释放门"
                    print(f"  [{label}] ✓ 释放门触发〔实宽≥"
                          f"{gate2.open_min_mm:.0f}mm 持续 {gate2.stable_s:.1f}s〕"
                          " → stdin 'q' 收尾")
                    try:
                        p.stdin.write("q\n")
                        p.stdin.flush()
                    except Exception:
                        pass
                elif time.time() - t_place0 > gate2_timeout:
                    reason = f"place 超时 {gate2_timeout:.0f}s"
                    print(f"  [{label}] ⚠ {reason} → 'q' 中止")
                    try:
                        p.stdin.write("q\n")
                        p.stdin.flush()
                    except Exception:
                        pass
            if reason == "门触发" and p.poll() is not None:
                break
    except KeyboardInterrupt:
        print(f"  [{label}] ⌨ 编排器被 Ctrl-C —— 转发 'q' 给子进程再退"
              f"（防孤儿挂起 nav）")
        try:
            p.stdin.write("q\n")   # 恒 q：Ctrl-C=中止，绝不触发折叠
            p.stdin.flush()
        except Exception:
            pass
    # 退出兜底升级：q 15s 没退 → SIGINT 5s → SIGKILL（守护网兜底 nav CONT）
    deadline = time.time() + 15
    while p.poll() is None and time.time() < deadline:
        time.sleep(0.2)
    if p.poll() is None:
        print(f"  [{label}] ⚠ 未响应，SIGINT")
        p.send_signal(signal.SIGINT)
        deadline = time.time() + 5
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.2)
    if p.poll() is None:
        print(f"  [{label}] ⚠ SIGKILL（nav 守护网将兜底 CONT）")
        p.kill()
    p.wait()
    rc = p.returncode
    # 带门段：只有门触发算成功（q 退 rc=2 / 自然收尾 rc=0；冻结看门狗
    # SystemExit rc=1、跑满轮数无签名=完成探测失败，都必须拦下，
    # 不带空爪/带物滚进下一段）。折叠档 ok_rc=(0,)——rc=2=折叠前 q 退
    # （门到注入之间手滑）或 rc=4=折叠中途 q（急停），都不算成功；
    # reset 段（无门）照常 rc==0。
    # 二相（gate2）：释放门触发才算成功——折叠失败 rc=1、释放前 q（rc=2
    # 但 reason≠释放门）、place 超时、跑满轮数未释放（rc=0），全拦下
    if gate2 is not None:
        ok = (reason == "释放门") and rc in (0, 2)
    elif gate is not None:
        ok = (reason == "门触发") and rc in ok_rc
    else:
        ok = (rc == 0)
    print(f"[{label}] 退出 rc={rc} 原因={reason or '自然结束'} 用时 {time.time()-t0:.1f}s"
          f" {'✓' if ok else '⛔'}")
    return ok, reason


def stage_loop_cmd(args, which):
    if which == "pick":
        base = [str(LOOP_PY), "--ckpt", str(args.pick_ckpt), "--prompt", args.pick_prompt]
        if not args.no_nav_suspend:
            base += ["--nav-suspend"]
        base += ["--exec", "--grip-state-cmd"]
        # 折叠档（10-13）：--until 到 reset 及以后时，门触发注入 'r'，
        # loop 同会话跑 reset（+横移）后 rc=0，省 warmup 子进程 ~4-5s。
        # --until pick / --no-fold 不带（原 q 退出语义）。
        # 摆臂目标/norm_stats 取 place ckpt（g1_warm_core 与 warmup 同源序列）
        if (not args.no_fold) and args.until in ("reset", "move", "place", "full"):
            base += ["--post-gate",
                     "warm-move" if args.until in ("move", "place", "full") else "warm",
                     "--fold-ckpt", str(args.place_ckpt),
                     "--fold-pose-file", str(args.place_pose),
                     "--fold-speed", str(args.reset_speed)]
            if args.until in ("move", "place", "full"):
                # 横移两态（10-14）：默认 map 系绝对坐标直达工位（治 odom
                # 漂移）；--fold-relative 退回 base_link 相对位移
                if args.fold_relative:
                    base += ["--then-move", str(args.shift_x), str(args.shift_y),
                             str(args.shift_yaw_deg)]
                else:
                    base += ["--then-goto", "place"]
                base += ["--move-timeout", str(args.move_timeout)]
            if args.until == "full":
                # 二相（2026-10-14 单模型定案）：0930 权重=pick+place 合并
                # 训练（train_config wandb notes 实锤），place 不换权重——
                # 只传 prompt，折叠后同引擎换 prompt 切相。--dual-load 才
                # 传 place ckpt 走旧双构造路径（A/B/回退）
                if args.dual_load:
                    base += ["--phase2-ckpt", str(args.place_ckpt)]
                base += ["--phase2-prompt", args.place_prompt]
        extra = shlex.split(args.pick_args or "")
    else:
        base = [str(LOOP_PY), "--ckpt", str(args.place_ckpt), "--prompt", args.place_prompt]
        if not args.no_nav_suspend:
            base += ["--nav-suspend"]
        base += ["--exec"]     # place 不带 grip-state-cmd（10-12 判读）
        extra = shlex.split(args.place_args or "")
    return base + extra


def stage_reset_cmd(args, with_move=False):
    """reset 段命令；with_move=True 时 --then-move 同会话顺带横移
    （reset+move 合并：省一次 init+destroy+段间空档 ~3-4s；
    --until reset 的 bring-up 语义=只复位不挪车，不带横移）。"""
    cmd = [str(WARMUP_PY), "--ckpt", str(args.place_ckpt),
           "--pose-file", str(args.place_pose), "--no-grip",
           "--speed", str(args.reset_speed),
           "--init-settle", str(args.init_settle)]
    if with_move:
        # 横移两态（10-14）：默认 --then-goto place（map 系绝对坐标直达，
        # 需定位锁在）；--fold-relative 退回相对位移 --then-move
        if args.fold_relative:
            cmd += ["--then-move", str(args.shift_x), str(args.shift_y),
                    str(args.shift_yaw_deg)]
        else:
            cmd += ["--then-goto", "place"]
        cmd += ["--move-timeout", str(args.move_timeout)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exec", dest="do_exec", action="store_true",
                    help="真实运动（缺省=干跑只打印各段命令；不进配置）")
    ap.add_argument("--until", choices=["pick", "reset", "move", "place", "full"],
                    default="full",
                    help="跑到该段为止（bring-up 逐段验证）。full=pick+门触发"
                         "折叠+横移+同进程二相切 place 轮（2026-10-14，默认；"
                         "place 冷启动 ~22-28s 全省）。place=独立 place 子进程"
                         "（旧路径，A/B 对照用）")
    ap.add_argument("--pick-ckpt", default=str(PICK_CKPT))
    ap.add_argument("--pick-prompt", default=PICK_PROMPT)
    ap.add_argument("--place-ckpt", default=str(PLACE_CKPT))
    ap.add_argument("--place-prompt", default=PLACE_PROMPT)
    ap.add_argument("--place-pose", default=str(PLACE_POSE))
    ap.add_argument("--pick-args", default="",
                    help='透传 pick 段 loop 的额外旗标，如 "--near-n 15 --near-div 1.43"')
    ap.add_argument("--place-args", default="", help="透传 place 段 loop 的额外旗标")
    ap.add_argument("--pick-stable-s", type=float, default=2.0,
                    help="门秒数：全部监听侧实宽在判据带内需连续满足"
                         "（10-13 用户定案『持续抓住 ~2s 才算』；3s 判长）")
    ap.add_argument("--pick-cmd-close-pct", type=float, default=25.0,
                    help="v4.1 交集判据：pick 门要求物理在带 且 该侧指令%%"
                         "≤本值（开爪爬坡实宽会落持物带但指令 71-83%%=高，"
                         "141150 误触发实录；持物时指令≈0-5%%。0=退回 v4 "
                         "纯物理带）")
    ap.add_argument("--held-min-mm", type=float, default=12.0,
                    help="pick 持物带下限 mm（空爪闭合≈0-2mm 不进带=不触发；"
                         "持物≈30-40mm=数据集撑 33.5%% 语义）")
    ap.add_argument("--held-max-mm", type=float, default=80.0,
                    help="pick 持物带上限 mm（张开着≈90-120 不算持物）")
    ap.add_argument("--open-min-mm", type=float, default=60.0,
                    help="place 释放带下限 mm（实宽≥它=松爪完成）")
    ap.add_argument("--pick-timeout", type=float, default=120.0,
                    help="pick 段硬预算 s（超时=整个流程中止，不空爪去 place；"
                         "默认 120>200 轮自然时长 ~70s——爪慢相位是 ckpt 常态"
                         "〔10-08 L 爪开→合在轮 198〕，60s 会误杀成功晚抓）")
    ap.add_argument("--place-stable-s", type=float, default=1.5,
                    help="place 门秒数：实宽≥--open-min-mm 需连续满足（放开 ~1.5s 即收）")
    ap.add_argument("--place-timeout", type=float, default=45.0,
                    help="place 段硬预算 s（防梦游期空挥；释放早停通常先到）")
    ap.add_argument("--reset-speed", type=float, default=0.8,
                    help="复位段臂/头关节速度 rad/s（warmup --speed 透传；"
                         "默认 0.8≈回放台实证天花板 1.03 的八成（10-13 用户判"
                         "'尽量快'）；腿固定 0.2 不受此参影响）")
    ap.add_argument("--init-settle", type=float, default=0.5,
                    help="复位段 SDK init 后 settle 秒（warmup --init-settle "
                         "透传；编排流程里 pick 子进程刚释放控制器、栈是热的，"
                         "默认 0.5 而非独立运行时的 5.0；不足则 start_controller "
                         "非 SUCCESS 响亮失败）")
    ap.add_argument("--shift-x", type=float, default=SHIFT_X,
                    help="横移 x 分量 m（base_link 前向；同事 nav_points 值）")
    ap.add_argument("--shift-y", type=float, default=SHIFT_Y,
                    help="横移 y 分量 m，base_link +y=左（同事实测 1.487，非 1m）")
    ap.add_argument("--shift-yaw-deg", type=float, default=SHIFT_YAW_DEG,
                    help="横移伴随转角 deg（同事 nav_points 值 ~1.07°）")
    ap.add_argument("--move-timeout", type=float, default=30.0)
    ap.add_argument("--no-preflight", action="store_true",
                    help="跳过 move 段 SDK 健康预检（省 ~5s；10-13 案：缺 "
                         "robot.init() 的 start_controller 必 segfault）")
    ap.add_argument("--no-fold", action="store_true",
                    help="退回旧四段路径（reset=独立 warmup 子进程；默认折叠="
                         "reset+横移并入 pick 会话，gate 后注入 'r'，省 "
                         "~4-5s；--until pick 时本就无折叠，此参无效果）")
    ap.add_argument("--no-nav-suspend", action="store_true",
                    help="不冻结导航栈（A/B 用）：循环全程导航栈在线，推理 "
                         "+55ms/轮→水位抬高，pace 须配 0.43（9/30 六连跑 "
                         "sweep 定案 0.43/1.65）；折叠横移反而免解冻等待/"
                         "重定位闸秒过")
    ap.add_argument("--fold-relative", action="store_true",
                    help="折叠横移退回相对位移 --then-move（10-14 起默认="
                         "--then-goto place 走 map 系绝对坐标直达：治 odom "
                         "累计漂移、同事实测落点 3cm/2° 修正内建；goto 需定位"
                         "锁在，未锁=拒发）")
    ap.add_argument("--dual-load", action="store_true",
                    help="place 相换权重（旧双构造路径）：开局后台载入 place "
                         "ckpt+建引擎，折叠后换 model/ns/包络切相。10-14 定案"
                         "默认单模型（0930 权重=pick+place 合并训练，train_"
                         "config 实锤）只换 prompt，本参数留作 A/B/回退")
    args = ap.parse_args()
    if args.until == "full" and args.no_fold:
        raise SystemExit("--until full（二相全流程）需要折叠档——--no-fold 时 "
                         "place 走独立子进程，请用 --until place")

    # 折叠档：--until 到达 reset 及以后才实际生效（pick 档=原 q 退出）
    fold = (not args.no_fold) and args.until in ("reset", "move", "place", "full")
    fold_move = fold and args.until in ("move", "place", "full")
    phase2 = fold_move and args.until == "full"   # 二相：place 并入 pick 会话

    if phase2:
        plan = [("pick", "pick 推理（双臂抓取）→ 门触发折叠+横移 → 同进程切 "
                 "place 轮（二相，冷启动零空档；释放门 实宽≥"
                 f"{args.open_min_mm:.0f}mm×{args.place_stable_s:.1f}s 收，"
                 f"place 预算 {args.place_timeout:.0f}s）")]
    else:
        plan = [
            ("pick", "pick 推理（双臂抓取）"
             + ("；门触发后同会话折叠 reset" + ("+横移" if fold_move else "")
                if fold else "")),
            ("reset", "复位到 place 起始持物位姿（--no-grip 保住爪中物体）"
             + ("（已并入 pick 会话折叠）" if fold else "")),
            ("move", f"底盘横移 y {args.shift_y:+.2f} m"
             + ("（已并入 pick 会话折叠）" if fold else "（已并入 reset 同会话）")),
            ("place", "place 推理（放盒）"),
        ]
    print("== pick_place 流程计划 ==")
    for i, (k, desc) in enumerate(plan, 1):
        print(f"  {i}. [{k}] {desc}" + ("（本跑到此为止）" if k == args.until else ""))
    if not args.do_exec:
        print("\n-- 干跑：各段实际命令 --")
        print(f"[pick]  $ {' '.join(stage_loop_cmd(args, 'pick'))}")
        print(f"[reset] $ {' '.join(stage_reset_cmd(args, args.until in ('move', 'place', 'full')))}"
              f"   # 臂速 {args.reset_speed} rad/s（原 0.15 的 "
              f"{args.reset_speed/0.15:.1f}×），init settle {args.init_settle}s"
              f"（独立运行默认 5s）"
              + ("   # ⚠ 折叠档：--exec 时跳过本段（已并入 pick 会话）"
                 if (not args.no_fold) and args.until in ("reset", "move", "place", "full")
                 else ""))
        print(f"[预检]  $ {' '.join([str(MOVE_PY), '--probe-only'])}   # --exec 时 pick 前零运动 ~5s")
        if args.fold_relative:
            mv_desc = (f"--then-move {args.shift_x} {args.shift_y} "
                       f"{args.shift_yaw_deg}")
        else:
            mv_desc = "--then-goto place（map 系绝对坐标直达）"
        print(f"[move]  已并入 reset 会话（{mv_desc}；"
              f"--until reset 时不带）。独立手动工具仍可用：{MOVE_PY}")
        if phase2:
            p2_desc = (f"双构造换权重：--phase2-ckpt {args.place_ckpt}"
                       if args.dual_load else
                       "单模型：同权重（0930 合并训练）只换 prompt")
            print(f"[place] 已并入 pick 会话（二相·{p2_desc}；"
                  "place 轮 grip-state-cmd 自动关）")
        else:
            print(f"[place] $ {' '.join(stage_loop_cmd(args, 'place'))}")
        print(f"[pick 门] 全部监听侧实宽在持物带 "
              f"{args.held_min_mm:.0f}-{args.held_max_mm:.0f}mm 连续 "
              f"{args.pick_stable_s}s（超时 {args.pick_timeout:.0f}s 中止）")
        print(f"[place 门] 全部监听侧实宽≥{args.open_min_mm:.0f}mm 连续 "
              f"{args.place_stable_s}s（预算 {args.place_timeout:.0f}s）"
              + ("；二相模式下布防于 loop「折叠完成」行之后" if phase2 else ""))
        print("⚠ 门只认物理实宽回读〔loop 爪行 [实 X.Xmm]〕；指令闭合≠抓住"
              "（143656 空爪实证），读不到回读=按未持物处理")
        return

    if args.do_exec and not args.no_preflight:
        print("== 预检 SDK 健康（g1_base_move --probe-only，零运动 ~5s）==")
        ok_p, _ = run_child([str(MOVE_PY), "--probe-only"], None, 30.0, "预检")
        if not ok_p:
            print("⛔ SDK 栈不健康（原生崩/非 SUCCESS）——勿烧 pick。头号嫌疑="
                  "客户端脚本缺 robot.init()（10-13 案：缺 init 的 "
                  "start_controller=has_controller_group this=NULL Segfault，"
                  "对照 reset_pos/warmup/loop 构造块）")
            raise SystemExit(1)

    t_all = time.time()
    for k, desc in plan:
        print(f"\n== 段[{k}] {desc} ==")
        if k == "pick":
            gate = PhaseGate(["R", "L"], done_open=False,
                             stable_s=args.pick_stable_s,
                             held_min_mm=args.held_min_mm,
                             held_max_mm=args.held_max_mm,
                             close_max_pct=args.pick_cmd_close_pct)
            # 折叠档：门触发注入 'r'（loop 折叠 reset 后 rc=0）；成功判据
            # 收紧为 rc==0——rc=2=折叠前 q 退、rc=4=折叠中途 q（急停），
            # 都不放行进 place。预算加折叠段（摆臂 ~15s + 定位闸 ~90s
            # 【10-14 实测：解冻后 localization 重锁 60-90s，盲走=回退舞步】
            # + 就绪重试 ~60s + 横移 ≤ move_timeout）
            budget = args.pick_timeout + (150.0 + args.move_timeout if fold else 0.0)
            if phase2:
                # 二相：'r' 注入后进程不退——折叠完成行布防释放门，触发写
                # 'q'；ok=释放门触发 且 rc∈{0,2}（run_child gate2 分支）。
                # +45 仅双构造档：loop 开局同载 place 构造（10-14 首跑载入
                # +建引擎 26s）——pick 起跑晚这些，任务窗不能被启动吃掉；
                # 单模型档无第二载入，pick 即刻起跑不加
                if args.dual_load:
                    budget += args.place_timeout + 45.0
                else:
                    budget += args.place_timeout
                gate2 = PhaseGate(["R", "L"], done_open=True,
                                  stable_s=args.place_stable_s,
                                  open_min_mm=args.open_min_mm)
                ok, _ = run_child(stage_loop_cmd(args, "pick"), gate, budget,
                                  "pick", stop_key="r", gate2=gate2,
                                  gate2_timeout=args.place_timeout)
            else:
                ok, _ = run_child(stage_loop_cmd(args, "pick"), gate, budget,
                                  "pick", stop_key="r" if fold else "q",
                                  ok_rc=(0,) if fold else (0, 2))
        elif k == "reset":
            if fold:
                print("  [reset] 已并入 pick 会话折叠（--post-gate warm），跳过独立段")
                ok = True
            else:
                with_move = args.until in ("move", "place")
                budget = 120.0 + (args.move_timeout + 15.0 if with_move else 0.0)
                ok, _ = run_child(stage_reset_cmd(args, with_move), None,
                                  budget, "reset")
        elif k == "move":
            print("  [move] 已并入同会话横移（折叠/reset-then-move），跳过独立段")
            ok = True
        elif k == "place":
            gate = PhaseGate(["R", "L"], done_open=True,
                             stable_s=args.place_stable_s,
                             open_min_mm=args.open_min_mm)
            ok, _ = run_child(stage_loop_cmd(args, "place"), gate,
                              args.place_timeout, "place")
        if not ok:
            print(f"\n⛔ 段[{k}] 失败——流程中止（已完成 "
                  f"{time.time()-t_all:.1f}s，机械臂/底盘留在原地）")
            raise SystemExit(1)
        if k == args.until:
            print(f"\n== --until {k} 到达，提前收工（总用时 {time.time()-t_all:.1f}s）==")
            return
    print(f"\n✅ 全流程完成（总用时 {time.time()-t_all:.1f}s）")


if __name__ == "__main__":
    main()
