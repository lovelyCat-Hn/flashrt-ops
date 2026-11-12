#!/usr/bin/env python
"""S1 dock 任务编排器：home→dock→traj 推理任务→home 循环往复（场景编排测试）。

循环体（每 cycle）：
  1. 底盘导航 dock（pns_point.py go dock，子进程=独立 SDK 会话，可达性闸+
     平台判到位，全程 proven 路径零新导航代码）
  2. 推理任务（s1_mission_worker.py，常驻进程 stdin 协议驱动：TE.run() 跑满
     --rounds 次推理=模糊完成判据，结束保持推理末状态返程（--no-tail-home，
     11-12 用户定案：归位含松爪，持物返程=扔货；真实任务=推理到某阶段触发
     返程，末状态就是要带回去的东西），引擎/SDK 会话不解体）
  3. 底盘导航 home（同 1）

常驻约束的落法：
  - 推理引擎：worker 进程全程存活（引擎/权重/GPU 上下文只装载一次），
    任务间不重启——接缝见 s1_mission_worker.py 文件头（D1/D2 修改点）
  - 导航栈：pns/localization 是 WORKING 模式服务树，编排器不碰；每次导航
    用 pns_point 独立子进程（自带切控制器/优雅退出），worker 任务间隙自身
    只切 SWERVE_CHASSIS_POSE_CTRL 让位（不 start_controller，pns 通路保全）
  - 导航客户端常驻（可选 --nav-resident，2026-11-12）：s1_nav_worker.py
    SDK 会话跨腿保持，省每腿 ~8-10s 子进程冷启动（切控制器/定位/地图闸/
    可达闸每腿照跑）；默认关=pns_point 子进程 proven 路径

用法（先干跑后真跑，两次确认纪律）：
  # ① 干跑（默认态，无旗标）：导航只做可达性查询、任务无下发彩排
  #    （引擎真装载、控制器真切）
  python run_dock_mission.py --cycles 2 --rounds 6
  # ② 真跑（急停在 手边！起点应在 home 附近）
  python run_dock_mission.py --cycles 2 --rounds 6 --exec
  # fineart 线（RTC 权重必带引导窗）
  python run_dock_mission.py --cycles 2 --rounds 6 --exec \
      --policy fineart_vla_s1 --worker-extra "--execution-horizon 10"

退出码: 0 全部 cycle 完成 / 2 导航失败 / 3 任务失败 / 4 worker 异常死亡 /
        130 人工中断
"""
import argparse
import faulthandler
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time

PY = "/home/galbot/miniforge3/envs/flash_pyrt311/bin/python"
SDK_LIB = "/userdata/update/manual_update/lib"
HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "s1_mission_worker.py")
NAV_DIR = os.path.join(os.path.dirname(HERE), "nav")
PNS_POINT = os.path.join(NAV_DIR, "pns_point.py")
NAV_WORKER = os.path.join(HERE, "s1_nav_worker.py")
POINTS_FILE = os.path.expanduser("~/holy/data/s1_nav_points.json")
MAP_CUR = "/var/maps/cur"
LOG_DIR = os.path.expanduser("~/holy/logs")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cycles", type=int, default=2, help="任务循环次数（默认 2）")
ap.add_argument("--rounds", type=int, default=6,
                help="每任务推理块数=模糊完成判据（worker --rounds，默认 6；"
                     "每块≈1.7s 消费+推理）")
ap.add_argument("--policy", default=None, help="透传 worker：pi05（默认）| fineart_vla_s1")
ap.add_argument("--prompt", default=None, help="透传 worker（默认 config [run].prompt）")
ap.add_argument("--ckpt", default=None, help="透传 worker（默认 config [run].ckpt）")
ap.add_argument("--config", default=None, help="透传 worker --config（默认 config/s1.toml）")
ap.add_argument("--worker-extra", default="",
                help="透传 worker 的额外参数串（如 '--execution-horizon 10'）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实执行（默认干跑：导航只查可达性、任务无下发彩排）")
ap.add_argument("--speed", type=float, default=0.15, help="导航速度上限 m/s（默认 0.15；规划层硬顶 0.3）")
ap.add_argument("--nav-omni", action="store_true",
                help="导航下发 omni_plan=True：允许横移/倒移不掉头（平移偏好）。"
                     "出厂默认关的来历=车尾 180° 雷达盲区（平移时前雷达不照行驶"
                     "方向）；本点位线 1.4m 直线空旷风险低。真机 A/B 用，默认关")
ap.add_argument("--nav-resident", action="store_true",
                help="常驻导航 worker：SDK 会话跨腿保持，每腿省 pns_point 子进程"
                     "冷启动 ~8-10s（切控制器/定位/地图/可达闸每腿照跑不省；"
                     "默认关=每腿 pns_point 子进程 proven 路径）")
ap.add_argument("--nav-timeout", type=float, default=120.0, help="单次导航超时 s（默认 120）")
ap.add_argument("--nav-retries", type=int, default=1,
                help="rc=4（未确认到位）时的重试次数（默认 1；pns 会从当前位重规划）")
ap.add_argument("--task-timeout", type=float, default=None,
                help="单任务总超时 s（默认 rounds×5+300，含 warm/归位/推理余量）")
ap.add_argument("--load-timeout", type=float, default=900.0,
                help="worker 装载超时 s（模型加载+预热，默认 900）")
ap.add_argument("--points-file", default=POINTS_FILE, help="点位文件（默认 ~/holy/data/s1_nav_points.json）")
ap.add_argument("--ignore-map", action="store_true", help="点位地图一致性不符也强制执行")
ap.add_argument("--mission-name", default=None, help="日志名（默认 mission_dock_<时间戳>）")
args = ap.parse_args()

TASK_TIMEOUT = args.task_timeout if args.task_timeout is not None else args.rounds * 5.0 + 300.0
DRY = not args.do_exec

# 看门狗：全任务预算（超时 dump 全线程栈并退——绕过一切 finally，预算须宽裕）
_budget = (args.load_timeout
           + args.cycles * (2 * (args.nav_timeout * 2 + 90) * (args.nav_retries + 1)
                            + TASK_TIMEOUT + 60)
           + 600)
faulthandler.dump_traceback_later(_budget, exit=True)

# ── 全文日志 ──
os.makedirs(LOG_DIR, exist_ok=True)
_mname = args.mission_name or time.strftime("mission_dock_%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, _mname + ".log")
_log_fh = open(LOG_PATH, "a", buffering=1, encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except (ValueError, OSError):
                pass   # 退出期另一支已关（11-12 实录：[exit] 行刷半截+
                       # ValueError + lost sys.stderr 噪声）

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except (ValueError, OSError):
                pass


sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)


def _ts():
    return time.strftime("%H:%M:%S")


def say(msg):
    print(f"[{_ts()}] [mission] {msg}", flush=True)


def banner(msg):
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72, flush=True)


# ── 点位与地图一致性闸（与 pns_point 同源口径：换图旧点拒绝执行）──
def load_point(label):
    try:
        with open(args.points_file, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception as e:
        say(f"⛔ 点位文件读取失败 {args.points_file}：{e}")
        sys.exit(2)
    pt = (db.get("points") or {}).get(label)
    if pt is None:
        say(f"⛔ 点位不存在：{label}（先在 nav/ 下 pns_point.py record {label}）")
        sys.exit(2)
    cur = os.path.realpath(MAP_CUR)
    name = os.path.basename(cur)
    mtime = size = 0
    try:
        st = os.stat(os.path.join(cur, "global_cloud_cleaned.pcd"))
        mtime, size = int(st.st_mtime), int(st.st_size)
    except OSError:
        pass
    mp = pt.get("map") or {}
    diffs = []
    if mp.get("name") and mp["name"] != name:
        diffs.append(f"地图目录 {mp['name']} → {name}")
    if mp.get("pcd_mtime") and mp["pcd_mtime"] != mtime:
        diffs.append(f"点云 mtime {mp['pcd_mtime']} → {mtime}")
    if diffs and not args.ignore_map:
        say(f"⛔ 点位 {label} 与当前地图不一致（{'；'.join(diffs)}）"
            "——重录点位或加 --ignore-map")
        sys.exit(2)
    say(f"点位 {label}: x={pt['x']:+.3f} y={pt['y']:+.3f} "
        f"yaw={pt.get('yaw_deg', 0.0):+.2f}°（地图 {name}{'/⚠强制' if diffs else ''}）")
    return pt


# ── worker 进程管理 ──
worker = None
_q = queue.Queue()


def _reader(proc):
    for raw in proc.stdout:
        _q.put(raw.rstrip("\n"))
    _q.put(None)   # EOF 哨兵


def spawn_worker():
    global worker
    cmd = [PY, "-u", WORKER]
    if not DRY:
        cmd.append("--exec")
    cmd += ["--rounds", str(args.rounds)]
    # 真实任务语义（11-12 用户定案）：返程保持推理末状态——归位含双爪打开，
    # 持物开车=先扔货；归位/投放由任务层自己决定。要回旧行为（任务完归位）
    # =删掉下面这行
    cmd.append("--no-tail-home")
    for flag, key in (("--policy", args.policy), ("--prompt", args.prompt),
                      ("--ckpt", args.ckpt), ("--config", args.config)):
        if key is not None:
            cmd += [flag, key]
    cmd += shlex.split(args.worker_extra)
    env = os.environ.copy()
    env["PYTHONPATH"] = SDK_LIB + (os.pathsep + env["PYTHONPATH"]
                                   if env.get("PYTHONPATH") else "")
    # 不加 LD_LIBRARY_PATH：本机 proven 干跑配方（11-02 八轮绿）=仅 PYTHONPATH，
    # 多挂 /data/galbot/lib 有 torch/flash_rt .so 遮蔽风险；导航子进程同口径
    say(f"启动 worker（{'exec' if not DRY else 'dry-run'}）：{' '.join(cmd)}")
    worker = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, env=env,
                              cwd=os.path.dirname(HERE), text=True, bufsize=1)
    threading.Thread(target=_reader, args=(worker,), daemon=True).start()


def relay(line):
    print(f"[{_ts()}]   {line}", flush=True)


def pump_until(pred, deadline, what):
    """消费 worker 输出直至 pred 命中/超时/进程亡。返回命中行或 None。"""
    while True:
        left = deadline - time.time()
        if left <= 0:
            say(f"⛔ 等待 {what} 超时")
            return None
        try:
            line = _q.get(timeout=min(left, 2.0))
        except queue.Empty:
            if worker.poll() is not None and _q.empty():
                say(f"⛔ worker 进程已退出（等待 {what} 期间）rc={worker.returncode}")
                return None
            continue
        if line is None:
            rc = worker.poll()
            say(f"⛔ worker 输出流结束（等待 {what} 期间）"
                + (f"，进程已退出 rc={rc}" if rc is not None else "，进程仍存活"))
            return None
        relay(line)
        if pred(line):
            return line


def wait_ready():
    line = pump_until(lambda s: s.startswith("@@READY"),
                      time.time() + args.load_timeout, "@@READY（装载完成）")
    return line is not None


def nav_to(label):
    """导航到点位标签；rc=4 重试。返回 True=到位。

    --nav-resident=常驻 worker 腿（nav_to_resident）；默认=每腿 pns_point
    子进程（proven 路径原样保留）。
    """
    if args.nav_resident:
        return nav_to_resident(label)
    for attempt in range(1, args.nav_retries + 2):
        cmd = [PY, "-u", PNS_POINT, "go", label, "--speed", str(args.speed),
               "--timeout", str(args.nav_timeout)]
        if args.nav_omni:
            cmd.append("--omni")
        if DRY:
            cmd.append("--dry-run")
        say(f"导航 → {label}（第 {attempt}/{args.nav_retries + 1} 次）"
            f"{' [dry-run]' if DRY else ''}")
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=NAV_DIR, text=True)
        for raw in proc.stdout:
            print(f"[{_ts()}]   {raw.rstrip()}", flush=True)
        proc.wait()
        el = time.time() - t0
        # 兜底看门狗：pns_point 自带 faulthandler 300s，这里再挂一层硬杀
        if proc.returncode is None or el > args.nav_timeout * 2 + 90:
            proc.kill()
            say(f"⛔ 导航子进程超时硬杀（{el:.0f}s）")
            return False
        say(f"导航 → {label} rc={proc.returncode}（{el:.1f}s）")
        if proc.returncode == 0:
            return True
        if proc.returncode == 4 and attempt <= args.nav_retries:
            say("未确认到位（rc=4），重试（pns 从当前位重规划）")
            continue
        say(f"⛔ 导航失败 rc={proc.returncode}"
            "（2 初始化/定位 3 不可达 4 未到位 5 控制器 8 地图不一致）")
        return False
    return False


def run_task(idx):
    """stdin 协议驱动一次任务。返回 rc（0=正常）。"""
    say(f"── 任务 {idx} 下发（--rounds {args.rounds}）──")
    worker.stdin.write("task\n")
    if not DRY:
        worker.stdin.write("go\n")   # 预填 TE.run() 的 exec 确认行（input 消费）
    worker.stdin.flush()
    t0 = time.time()
    line = pump_until(lambda s: s.startswith("@@TASK_DONE"),
                      t0 + TASK_TIMEOUT, f"@@TASK_DONE（任务 {idx}）")
    if line is None:
        return None   # 超时/死亡——调用方升级处置
    try:
        return int(line.strip().split("rc=")[1].split()[0])
    except (IndexError, ValueError):
        return 1


def kill_ladder(reason):
    """升级终止 worker：SIGINT →10s→ SIGTERM →5s→ SIGKILL。

    SIGINT=优雅档（任务中止、q 语义臂留原地、@@TASK_DONE rc=9 可见）；
    10s 未退升级 SIGTERM 起=硬终止（Python 层不参与：无归位、无汇总、
    无 @@TASK_DONE），臂停在半程任意位——事后需人工确认臂位/手动归位。
    """
    if worker is None or worker.poll() is not None:
        return
    say(f"{reason} → 终止 worker（SIGINT 升级链）")
    for sig, wait in ((signal.SIGINT, 10), (signal.SIGTERM, 5)):
        try:
            worker.send_signal(sig)
        except ProcessLookupError:
            return
        try:
            worker.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue
    worker.kill()
    worker.wait(timeout=5)


def quit_worker():
    """优雅收 worker（任务间隙 stdin 可读；任务中/挂死走 kill_ladder）。"""
    if worker is None or worker.poll() is not None:
        return worker.returncode if worker else None
    try:
        worker.stdin.write("quit\n")
        worker.stdin.flush()
    except (BrokenPipeError, OSError):
        kill_ladder("stdin 断（写 quit 失败）")
        return worker.poll()
    line = pump_until(lambda s: s.startswith("@@BYE") or "SDK 关闭中" in s,
                      time.time() + 90, "@@BYE（worker 收尾）")
    if line is None:
        kill_ladder("quit 未被响应")
    try:
        rc = worker.wait(timeout=30)
    except subprocess.TimeoutExpired:
        kill_ladder("退出等待超时")
        rc = worker.poll()
    say(f"worker 退出 rc={rc}")
    return rc


# ── 常驻导航 worker（--nav-resident；与推理 worker 同构的进程管理）──
navw = None
_nq = queue.Queue()


def _nav_reader(proc):
    for raw in proc.stdout:
        _nq.put(raw.rstrip("\n"))
    _nq.put(None)   # EOF 哨兵


def spawn_nav_worker():
    global navw
    cmd = [PY, "-u", NAV_WORKER]
    env = os.environ.copy()
    env["PYTHONPATH"] = SDK_LIB + (os.pathsep + env["PYTHONPATH"]
                                   if env.get("PYTHONPATH") else "")
    say(f"启动常驻导航 worker：{' '.join(cmd)}")
    navw = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env,
                            cwd=os.path.dirname(HERE), text=True, bufsize=1)
    threading.Thread(target=_nav_reader, args=(navw,), daemon=True).start()


def pump_nav_until(pred, deadline, what):
    """消费导航 worker 输出直至 pred 命中/超时/进程亡。返回命中行或 None。"""
    while True:
        left = deadline - time.time()
        if left <= 0:
            say(f"⛔ 等待 {what} 超时")
            return None
        try:
            line = _nq.get(timeout=min(left, 2.0))
        except queue.Empty:
            if navw.poll() is not None and _nq.empty():
                say(f"⛔ nav-worker 进程已退出（等待 {what} 期间）rc={navw.returncode}")
                return None
            continue
        if line is None:
            rc = navw.poll()
            say(f"⛔ nav-worker 输出流结束（等待 {what} 期间）"
                + (f"，进程已退出 rc={rc}" if rc is not None else "，进程仍存活"))
            return None
        relay(line)
        if pred(line):
            return line


def stop_nav_worker(reason):
    """收导航 worker：quit → 90s → SIGTERM 5s → SIGKILL（腿中 SIGTERM=硬终止）。"""
    if navw is None or navw.poll() is not None:
        return
    say(f"{reason} → 收导航 worker（quit 升级链）")
    try:
        navw.stdin.write("quit\n")
        navw.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    line = pump_nav_until(lambda s: s.startswith("@@BYE"),
                          time.time() + 90, "@@BYE（导航 worker 收尾）")
    if line is None:
        try:
            navw.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            navw.wait(timeout=5)
        except subprocess.TimeoutExpired:
            navw.kill()
    try:
        navw.wait(timeout=30)
    except subprocess.TimeoutExpired:
        navw.kill()
    say(f"nav-worker 退出 rc={navw.poll()}")


def nav_to_resident(label):
    """常驻导航 worker 腿；rc=4 重试，进程亡/腿挂死则重生。返回 True=到位。

    首腿惰性拉起（bring_up 不切控制器，可与任何状态共存）；腿内看门狗
    timeout+60s 先于编排器 timeout+120s 死亡，失联=硬杀重生不阻塞重试。
    """
    for attempt in range(1, args.nav_retries + 2):
        if navw is None or navw.poll() is not None:
            spawn_nav_worker()
            line = pump_nav_until(lambda s: s.startswith("@@NAV_READY"),
                                  time.time() + 60, "@@NAV_READY（导航会话建立）")
            if line is None:
                stop_nav_worker("导航会话建立失败")
                return False
            try:
                rrc = int(line.strip().split("rc=")[1].split()[0])
            except (IndexError, ValueError):
                rrc = 1
            if rrc != 0:
                say(f"⛔ 导航 worker bring_up 失败 rc={rrc}（2 初始化/定位）")
                return False
        cmd = f"go {label} --speed {args.speed} --timeout {args.nav_timeout}"
        if args.nav_omni:
            cmd += " --omni"
        if DRY:
            cmd += " --dry-run"
        say(f"导航 → {label}（第 {attempt}/{args.nav_retries + 1} 次）"
            f"{' [dry-run]' if DRY else ''}〔常驻〕")
        try:
            navw.stdin.write(cmd + "\n")
            navw.stdin.flush()
        except (BrokenPipeError, OSError):
            say("⛔ nav-worker stdin 断（写 go 失败）")
            if attempt <= args.nav_retries:
                continue
            return False
        t0 = time.time()
        line = pump_nav_until(lambda s: s.startswith("@@NAV_DONE"),
                              t0 + args.nav_timeout + 120,
                              f"@@NAV_DONE（导航 {label}）")
        el = time.time() - t0
        if line is None:
            if navw is not None and navw.poll() is None:
                navw.kill()
                say(f"⛔ 导航 worker 腿失联硬杀（{el:.0f}s）")
            if attempt <= args.nav_retries:
                continue
            return False
        try:
            rc = int(line.strip().split("rc=")[1].split()[0])
        except (IndexError, ValueError):
            rc = 1
        say(f"导航 → {label} rc={rc}（{el:.1f}s，含腿前置闸）")
        if rc == 0:
            return True
        if rc == 4 and attempt <= args.nav_retries:
            say("未确认到位（rc=4），重试（pns 从当前位重规划）")
            continue
        say(f"⛔ 导航失败 rc={rc}"
            "（2 初始化/定位 3 不可达 4 未到位 5 控制器 8 地图不一致 9 中断）")
        return False
    return False


def main():
    pt_dock = load_point("dock")
    pt_home = load_point("home")
    banner(f"S1 dock 任务编排 · {'干跑（无运动）' if DRY else '⚠ --exec 真实执行'}\n"
           f"循环 {args.cycles} 次 × 任务 {args.rounds} 推理块 | "
           f"dock ({pt_dock['x']:+.2f},{pt_dock['y']:+.2f}) ↔ "
           f"home ({pt_home['x']:+.2f},{pt_home['y']:+.2f}) | 速度 {args.speed} m/s | "
          f"导航={'常驻 worker' if args.nav_resident else 'pns_point 子进程'}"
           + ("\n急停确认！任务期 worker 输出逐行透传；编排器 Ctrl-C=优雅停"
              if not DRY else ""))
    if DRY:
        say("干跑口径：导航=pns 可达性查询（仍切控制器）｜任务=引擎装载+彩排不下发")

    say("① 启动常驻 worker（引擎装载中，进度透传）…")
    spawn_worker()
    if not wait_ready():
        kill_ladder("装载失败/超时")
        return 4
    say("worker 就绪（引擎常驻）")

    done = 0
    for c in range(1, args.cycles + 1):
        banner(f"cycle {c}/{args.cycles}")
        # 1) home → dock
        if not nav_to("dock"):
            kill_ladder("去程导航失败")
            return 2
        # 2) 任务
        rc = run_task(c)
        if rc is None:
            kill_ladder("任务超时/worker 失联")
            return 3
        if rc != 0:
            say(f"⛔ 任务 {c} 失败 rc={rc}（0 正常/1 异常/5 控制器/6 warm/"
                "9 SIGINT/10 护栏-星饿）——中止编排（详情见上方透传与 worker 日志）")
            quit_worker()
            return 3
        say(f"任务 {c} 完成（末状态保持，不归位）")
        # 3) dock → home
        if not nav_to("home"):
            say("⚠ 回程导航失败——机器人停在 dock↔home 途中；"
                "人工恢复：cd ~/holy/scripts/s1/nav && "
                f"{PY} -u pns_point.py go home --speed {args.speed}")
            kill_ladder("回程导航失败")
            return 2
        done = c
        say(f"cycle {c}/{args.cycles} 完成 ✓")

    banner(f"编排完成：{done}/{args.cycles} cycles ✓ → worker 收尾")
    quit_worker()
    stop_nav_worker("编排完成")
    return 0


_rc = 4
try:
    _rc = main()
except KeyboardInterrupt:
    print("\n[mission] ⛔ 人工中断（Ctrl-C）→ 优雅停 worker")
    kill_ladder("人工中断")
    _rc = 130
except Exception:
    import traceback
    traceback.print_exc()
    print("[mission] ⛔ 编排器内部异常 → 终止 worker")
    kill_ladder("编排器内部异常")
    _rc = 4
finally:
    try:
        if worker is not None and worker.poll() is None:
            kill_ladder("主流程退出仍有存活 worker")
    except Exception:
        pass
    try:
        if navw is not None and navw.poll() is None:
            stop_nav_worker("主流程退出仍有存活导航 worker")
    except Exception:
        pass
    # 最终 print 必须在 _log_fh.close() 之前：stdout/stderr 是 _Tee(真流, fh)，
    # 先关=退出打印半截 + lost sys.stderr（11-12 实录）
    print(f"[exit] rc={_rc}", flush=True)
    _log_fh.close()
sys.exit(_rc)
