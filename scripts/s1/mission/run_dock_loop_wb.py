#!/usr/bin/env python
"""S1 home↔dock 循环任务编排器（导航 → 推理 → 返航 → 循环）。

════════════════════════════════════════════════════════════════════════
【本文件由 WorkBuddy 侧生成（2026-11-12），对照实现】
vs code 侧 agent 正在实现同目录 run_dock_mission.py（尚未落地）；本脚本
按 s1_mission_worker.py 头注声明的 D4 协议（@@READY / task / @@PREAMBLE_OK /
@@TASK_DONE rc=N / quit / @@BYE）独立实现编排层，可作为对照与备份。
两版编排器职责相同，任一可用；真机首跑前建议两版都过一遍代码评审。
════════════════════════════════════════════════════════════════════════

流程（单循环）：
  pns_point.py go dock   →  发 task 行给常驻 worker（推理 --rounds 块后归位）
  →  等 @@TASK_DONE rc=0  →  pns_point.py go home  →  下一循环

常驻约束的实现：
  推理引擎：worker 子进程全程存活，引擎/GPU 上下文/SDK 会话跨任务复用
            （worker 内 D1/D2 修改点负责保活，编排器不碰）；
  导航栈：  pns 服务系统级常驻，编排器只在任务间隙拉起短生命周期导航
            子进程（pns_point.py go），每次独立 SDK 会话，退出即释放。

完成判据（用户定案的模糊口径）：跑满 --rounds 次推理（换块次数）即任务完成，
不做语义级完成检测；护栏/星饿停机由 worker 内部护栏负责（rc=10 上抛）。

冒烟模式（--smoke，零真机零 SDK 零权重）：
  worker 由本脚本 --fake-worker 模式模拟（同协议假实现），导航打印代替执行；
  用于验证握手、循环、计数、rc 处理与中断升级链，建议真机前先跑：
      python run_dock_loop_wb.py --smoke --cycles 2 --fake-rounds 2

真机用法（须 WORKING 模式 + pns 在跑 + 定位有效 + 已录 home/dock 点位）：
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python run_dock_loop_wb.py \
      --cycles 3 --rounds 3 --exec --prompt "<训练原句>"

中断升级链（Ctrl-C，三段式）：
  第 1 次：任务进行中 → SIGINT 转发 worker（同 q 语义，臂留原地，rc=9）；
           导航进行中 → 终止导航子进程；
  第 2 次：发 quit 行，worker 走 @@BYE 真三连；
  第 3 次：kill -9 兜底。
任一非零退出（导航失败/任务 rc 非 0）即停止后续循环，先尝试回 home 再退出。

退出码：0 全部循环成功 / 3 导航失败 / 4 任务失败（@@TASK_DONE rc≠0）/
        9 用户中断 / 1 握手超时或其他异常。
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

HERE = os.path.dirname(os.path.abspath(__file__))
HOLY = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # mission → s1 → scripts → holy
WORKER = os.path.join(HERE, "s1_mission_worker.py")
NAV_POINT = os.path.join(HERE, "..", "nav", "pns_point.py")
PY = os.path.expanduser("~/miniforge3/envs/flash_pyrt311/bin/python")
if not os.path.isfile(PY):
    PY = sys.executable


def log(tag, msg):
    print("[%s %s] %s" % (tag, time.strftime("%H:%M:%S"), msg), flush=True)


# ════════════════════════════ worker 交互 ════════════════════════════

class Worker:
    """常驻推理 worker 子进程：stdin 行协议 + stdout @@ 标记握手。

    非标记行（引擎日志、遥测）由读线程直接转发打印，标记行进队列供主线程
    等待；worker 崩溃以 EOF（None）入队呈现。"""

    def __init__(self, cmd, env=None, tag="worker"):
        self.tag = tag
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1, env=env)
        self.q = Queue()
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@"):
                self.q.put(line)
            else:
                print("  │%s│ %s" % (self.tag, line), flush=True)
        self.q.put(None)

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait_mark(self, prefix, timeout):
        """阻塞等待以 prefix 开头的 @@ 标记行；超时/EOF 返回 None。"""
        deadline = time.time() + timeout
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                return None
            try:
                line = self.q.get(timeout=remain)
            except Empty:
                return None
            if line is None:
                return None
            if line.startswith(prefix):
                return line

    def shutdown(self, sig_chain=(signal.SIGINT, signal.SIGKILL), wait_s=5.0):
        """中断升级链：SIGINT → 等待 → SIGKILL。"""
        if self.proc.poll() is not None:
            return self.proc.returncode
        try:
            self.proc.send_signal(sig_chain[0])
        except ProcessLookupError:
            return self.proc.returncode
        try:
            self.proc.wait(timeout=wait_s)
            return self.proc.returncode
        except subprocess.TimeoutExpired:
            for sig in sig_chain[1:]:
                try:
                    self.proc.send_signal(sig)
                except ProcessLookupError:
                    break
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log("orch", "worker 未响应 SIGKILL，放弃等待")
            return self.proc.returncode


# ════════════════════════════ 导航 ════════════════════════════

def nav_to(label, speed, timeout, stop_evt, smoke=False):
    """调用 nav/pns_point.py go <label>；返回退出码。smoke=True 时只打印。"""
    if smoke:
        log("nav", "[smoke] 假导航 → %s（0.3s 模拟，零真机零 SDK）" % label)
        time.sleep(0.3)
        return 0
    cmd = [PY, NAV_POINT, "go", label, "--speed", str(speed),
           "--timeout", str(timeout)]
    log("nav", "→ %s（speed=%.2f timeout=%.0fs）" % (label, speed, timeout))
    try:
        p = subprocess.Popen(cmd)
        while p.poll() is None:
            if stop_evt.is_set():
                p.terminate()
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    p.kill()
                return 9
            time.sleep(0.2)
        rc = p.returncode
    except OSError as e:
        log("nav", "启动失败：%s" % e)
        return 1
    log("nav", "%s rc=%d%s" % (label, rc, "" if rc == 0 else "（失败）"))
    return rc


# ════════════════════════════ 冒烟 worker ════════════════════════════

def fake_worker(fake_rounds):
    """--fake-worker：与 s1_mission_worker.py 同协议的假实现（零 SDK）。"""
    print("@@READY", flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            print("@@BYE", flush=True)
            return
        cmd = line.strip().lower()
        if cmd == "quit":
            print("@@BYE", flush=True)
            return
        if cmd != "task":
            continue
        time.sleep(0.3)
        print("[fake] 前奏：start_controller/warm/HOME 基准（模拟）", flush=True)
        print("@@PREAMBLE_OK", flush=True)
        for i in range(1, fake_rounds + 1):
            time.sleep(0.4)
            print("[fake] 推理块 %d/%d 消费完毕" % (i, fake_rounds), flush=True)
        print("[fake] 归位 + 底盘交接 POSE（模拟）", flush=True)
        print("@@TASK_DONE rc=0", flush=True)


# ════════════════════════════ 主流程 ════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=1,
                    help="循环次数：去 dock 推理并回 home 算一次（默认 1）")
    ap.add_argument("--rounds", type=int, default=3,
                    help="单次任务的推理次数（换块次数，透传 worker --rounds；默认 3）")
    ap.add_argument("--policy", default="pi05",
                    help="权重族：pi05 | fineart_vla_s1（透传 worker）")
    ap.add_argument("--prompt", default=None, help="任务指令（训练原句，透传 worker）")
    ap.add_argument("--ckpt", default=None, help="部署目录（透传 worker）")
    ap.add_argument("--subtask-goal", default=None,
                    help="fineart 专属高层目标（透传 worker；pi05 硬拒）")
    ap.add_argument("--exec", dest="do_exec", action="store_true",
                    help="真机模式（真实驱动双臂）；不加 = worker 干跑（仍会切控制器）")
    ap.add_argument("--go-home", default="home", help="返航/起点标签（默认 home）")
    ap.add_argument("--go-dock", default="dock", help="作业位标签（默认 dock）")
    ap.add_argument("--speed", type=float, default=0.15, help="导航速度 m/s（默认 0.15）")
    ap.add_argument("--nav-timeout", type=float, default=120.0,
                    help="单段导航超时 s（默认 120）")
    ap.add_argument("--task-timeout", type=float, default=900.0,
                    help="单次任务（--rounds 次推理+归位）超时 s（默认 900）")
    ap.add_argument("--ready-timeout", type=float, default=60.0,
                    help="等 worker @@READY 超时 s（默认 60）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：假 worker + 假导航，零真机零 SDK")
    ap.add_argument("--fake-worker", action="store_true",
                    help=argparse.SUPPRESS)  # 内部：以 worker 身份运行假协议
    ap.add_argument("--fake-rounds", type=int, default=2,
                    help=argparse.SUPPRESS)  # 内部：假 worker 的块数
    args = ap.parse_args()

    if args.fake_worker:
        fake_worker(args.fake_rounds)
        return 0

    if args.smoke:
        worker_cmd = [PY, os.path.abspath(__file__),
                      "--fake-worker", "--fake-rounds", str(args.rounds)]
        worker_env = None
    else:
        if args.policy == "pi05" and args.subtask_goal:
            log("orch", "--subtask-goal 为 fineart 专属，pi05 会硬拒，已去掉")
            args.subtask_goal = None
        worker_cmd = [PY, WORKER, "--rounds", str(args.rounds),
                      "--policy", args.policy]
        if args.prompt:
            worker_cmd += ["--prompt", args.prompt]
        if args.ckpt:
            worker_cmd += ["--ckpt", args.ckpt]
        if args.subtask_goal:
            worker_cmd += ["--subtask-goal", args.subtask_goal]
        if args.do_exec:
            worker_cmd += ["--exec"]
        worker_env = dict(os.environ,
                          LD_LIBRARY_PATH="/data/galbot/lib",
                          PYTHONPATH="/data/galbot/lib")

    log("orch", "循环编排：cycles=%d rounds=%d policy=%s exec=%s smoke=%s"
        % (args.cycles, args.rounds, args.policy, args.do_exec, args.smoke))
    log("orch", "点位：%s ⇄ %s  speed=%.2f" % (args.go_home, args.go_dock, args.speed))

    stop_evt = threading.Event()
    worker = Worker(worker_cmd, env=worker_env)

    # ── 中断升级链：1 转 SIGINT → 2 quit → 3 kill ──
    state = {"phase": "boot"}   # boot / nav / task / idle

    def on_sigint(signum, frame):
        if state["phase"] == "task":
            log("orch", "Ctrl-C #1 → SIGINT 转发 worker（臂留原地，等 @@TASK_DONE）")
            try:
                worker.proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        elif state["phase"] == "nav":
            log("orch", "Ctrl-C #1 → 终止当前导航")
            stop_evt.set()
        else:
            stop_evt.set()
        signal.signal(signal.SIGINT, on_sigint_2nd)

    def on_sigint_2nd(signum, frame):
        log("orch", "Ctrl-C #2 → 发 quit 行（worker 真三连退出）")
        stop_evt.set()
        try:
            worker.send("quit")
        except Exception:
            pass
        signal.signal(signal.SIGINT, on_sigint_3rd)

    def on_sigint_3rd(signum, frame):
        log("orch", "Ctrl-C #3 → SIGKILL worker")
        stop_evt.set()
        try:
            worker.proc.kill()
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, on_sigint)

    rc_final = 0
    done_cycles = 0
    try:
        # ── 握手 ──
        state["phase"] = "boot"
        line = worker.wait_mark("@@READY", args.ready_timeout)
        if line is None or worker.proc.poll() is not None:
            log("orch", "worker 未在 %.0fs 内 @@READY（或已退出 rc=%s），中止"
                % (args.ready_timeout, worker.proc.poll()))
            return 1
        log("orch", "worker @@READY（pid=%d）" % worker.proc.pid)

        # ── 主循环 ──
        for cycle in range(1, args.cycles + 1):
            if stop_evt.is_set():
                rc_final = 9
                break
            log("orch", "──── 循环 %d/%d ────" % (cycle, args.cycles))

            # 1) 导航去 dock
            state["phase"] = "nav"
            rc = nav_to(args.go_dock, args.speed, args.nav_timeout, stop_evt, smoke=args.smoke)
            if rc != 0:
                log("orch", "去 %s 失败（rc=%d），中止后续循环" % (args.go_dock, rc))
                rc_final = 3
                break

            # 2) 推理任务
            state["phase"] = "task"
            log("orch", "发 task（rounds=%d）→ 等待 @@TASK_DONE" % args.rounds)
            worker.send("task")
            done = worker.wait_mark("@@TASK_DONE", args.task_timeout)
            if done is None:
                log("orch", "任务超时或 worker 失联，中止")
                rc_final = 4
                break
            rc_task = int(done.split("rc=")[1]) if "rc=" in done else 1
            if rc_task != 0:
                log("orch", "任务失败 @@TASK_DONE rc=%d（9=中断 10=护栏/星饿），"
                            "不再继续" % rc_task)
                rc_final = 4
                break
            log("orch", "任务完成（rc=0）")

            # 3) 返航 home
            state["phase"] = "nav"
            rc = nav_to(args.go_home, args.speed, args.nav_timeout, stop_evt, smoke=args.smoke)
            if rc != 0:
                log("orch", "回 %s 失败（rc=%d），中止后续循环" % (args.go_home, rc))
                rc_final = 3
                break

            done_cycles = cycle
            state["phase"] = "idle"
            log("orch", "循环 %d/%d 完成" % (cycle, args.cycles))
    finally:
        # ── 收尾：quit → @@BYE → 等待 → 升级链兜底 ──
        state["phase"] = "idle"
        if worker.proc.poll() is None:
            try:
                log("orch", "发 quit → 等待 worker 收尾")
                worker.send("quit")
                bye = worker.wait_mark("@@BYE", 30)
                log("orch", "worker %s" % ("@@BYE" if bye else "未回 @@BYE"))
                try:
                    worker.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    worker.shutdown()
            except (BrokenPipeError, OSError):
                worker.shutdown()
        wrc = worker.proc.poll()

    log("orch", "════ 汇总 ════")
    log("orch", "完成循环 %d/%d  worker 退出码 %s  编排退出码 %d"
        % (done_cycles, args.cycles, wrc, rc_final))
    return rc_final


sys.exit(main())
