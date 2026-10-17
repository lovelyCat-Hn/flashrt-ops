#!/usr/bin/env python
"""最简 chunk 回放测试——保存的推理 chunk 经 set_joint_commands 30Hz 直发驱动双臂。

目的（2026-10-12）：看模型推理出的 chunk 走 set_joint_commands 路线的真实效果，
不需要重新推理——数据源是 run_g1_loop_traj --dump-chunks 落盘的 chunk_*.npz
（原始 (50,16) 模型输出 + BASE_ARM，包络截断前落盘=纯模型输出）。

执行语义与 run_g1_loop_traj 在役配方同构（frames-per-row=1 纯 30Hz 直发口径）：
  绝对目标 = BASE_ARM + chunk 臂维 delta（维序 0-6 右臂 / 8-14 左臂，爪维不喂）
  链式限幅：每行从当前实际位向绝对目标最多挪 ±--delta-max rad（在役值 0.05）
  每行一拍一令：set_joint_commands(position×14, time_from_start_s=--tfs-ms)
  （SDK 契约：标准关节只消费 position，tfs=唯一配速输入）
  包络钳位（同任务盒自动配对 models/*/joint_envelope.json，臂 14 维）
  + 漂移护栏 + 导程≈0 免发
块间重锚到当前实际位（闭环换块同款）。换块弃尾默认 auto：按下块锚点行
（其 BASE_ARM 与本块绝对目标的逐行匹配）+ 其推理拍数复现真实消费行数——
块在真实闭环里只消费到换块点（水位+推理落地），走满 50 行会让下块锚在
身后，边界出现大回退（2026-10-12 实测 max 536 mrad vs 真实换块点 93）。

⚠ 加 --exec 会真实驱动双臂！安全设计（主脚本/回放台同款）：
  1. 干跑默认：零下发，只打印每块计划（|Δcmd| 谱 + 首行导程）
  2. 起始慢速就位：0.2 rad/s 官方低频路径挪到首块行 0 绝对目标
  3. q/Ctrl-C 随时退（SDK 阻塞 C++ 调用中也能退）；物理急停第一优先级
  4. 执行前回车确认
  更全的 A/B 台（stream 帧插值突发 / nav-suspend / EMA）见
  scripts/probes/replay_stream_bench.py——本脚本是它的最小直发子集。

用法:
  ~/holy/run.sh ~/holy/scripts/test/chunk_replay_test.py \
      --chunks-dir ~/holy/logs/chunks_traj_20261010_130403              # 干跑
  ~/holy/run.sh ~/holy/scripts/test/chunk_replay_test.py \
      --chunks-dir ~/holy/logs/chunks_traj_20261010_130403 \
      --rounds 2 --exec                                                 # 真机
  # --chunks-dir 缺省=logs/ 下最新的 chunks_traj_* 目录
"""
import argparse
import atexit
import glob
import json
import os
import pathlib
import select
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "FlashRT"))
sys.path.insert(0, str(_ROOT / "scripts" / "inference"))

from galbot_sdk.g1 import GalbotRobot, JointCommand  # noqa: E402

LEFT = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_arm_joint{i}" for i in range(1, 8)]
ARM_NAMES = RIGHT + LEFT            # 与 chunk[0:7]+chunk[8:15] 逐位配对（右臂在前）
GAP_EPS = 1e-4                      # 导程低于该值免发（伺服保持）

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--chunks-dir", default=None,
                help="--dump-chunks 落盘目录（缺省=logs/ 最新 chunks_traj_*）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真机下发（不带=干跑：打印计划即退）")
ap.add_argument("--rounds", type=int, default=2,
                help="回放块数（默认 2；0=目录内全部）")
ap.add_argument("--rows-per-block", type=int, default=None,
                help="每块消费行数：缺省=auto（按下块锚点行+其推理拍数复现真实"
                     "换块弃尾；走满 50 行=下块锚在身后，边界回退 max 536 mrad "
                     "vs 真实换块点 93 mrad，2026-10-12 本机实测）；"
                     "0=走满全块；N=固定 N 行/块")
ap.add_argument("--start", type=int, default=1,
                help="起始块序号（1 起，对应 chunk_XXX.npz 编号）")
ap.add_argument("--fps", type=int, default=30, help="行节拍 Hz（训练控制率）")
ap.add_argument("--delta-max", type=float, default=0.05,
                help="链式限幅 rad/行（在役配方值）")
ap.add_argument("--skip-retrace", dest="skip_retrace", nargs="?", const="delay",
                default=None, choices=["delay", "anchor"],
                help="换块接缝策略：delay=固定跳过前 ceil(infer_ms/tick) 行回溯段"
                     "（首版；滞后不释放→gap 持续偏大，抖动显形）；"
                     "anchor=从最接近臂当前位的行开始（argmin |target−cur| 取前进侧，"
                     "滞后每块释放且不倒车）。不带=在役口径（原样消费回溯段"
                     "=换块往复摆动根源）")
ap.add_argument("--merge", type=int, default=1,
                help="合步 N：每 N 行合一条指令（目标=组末行端点，指令间隔 N 拍，"
                     "块时长不变=同消费速率；1=逐行直发在役口径）。"
                     "A/B set_joint_commands 目标采样效应（10-12 探针 pursuit 律预言："
                     "均衡臂速=计划速度与采样无关，合步不提速只改滞后形态）")
ap.add_argument("--glide", type=int, default=0,
                help="滑入拍数 N（0=关，默认；仅 --skip-retrace anchor 生效）："
                     "跳块后目标链不从旧块末目标一步拽到入口点，而是分 N 拍定步长"
                     "滑过去（每拍步长=(入口−旧块末目标)/N）——滞后分期释放，"
                     "拍0 |Δcmd| 从 ≈lag 降到 ≈lag/N，臂减速平滑无猛拉。"
                     "⚠ 实测缺陷：定步长倒退快于臂减速，尾 1-2 拍目标穿到臂身后"
                     "=小回摆摊前几拍；已被 --blend 前向搭接取代，留作对照")
ap.add_argument("--blend", action="store_true",
                help="前向搭接（推荐单独用，勿与 --skip-retrace 叠加）：换块后前 M 拍"
                     "不发新块计划行，改发旧块末目标→rows[M−1] 直线路径"
                     "（M 自适应=计划首次追平旧链高度的行号+1）——参考链全程前向"
                     "移动，臂不停车不回摆直接追过块界；滞后不在块界释放，自然"
                     "并入块内正常滞后。M 拍替换前 M 行，块时长/消费行数守恒")
ap.add_argument("--tfs-ms", type=float, default=0.0,
                help="time_from_start_s 毫秒：0=尽快到达（官方流式示例同值）")
ap.add_argument("--ema-alpha", type=float, default=0.0,
                help="行目标 EMA 低通系数（0=关，默认；主脚本同义）："
                     "rows[k]←α·raw[k]+(1−α)·rows[k−1]，滤换块锚后小回退；"
                     "滤波态跨块携带（从上块末下发目标重播=主脚本 161933 修正同款）")
ap.add_argument("--install-offset", type=int, default=0,
                help="跳帧安装模拟（0=关，默认；与 --skip-retrace 互斥）：换块后"
                     "新块从第 N 行起取消费窗（rows[N:N+消费行数]）——复现在役"
                     "水位 0+skip-install 的墙钟槽位接入（seam_matrix 10-17 判定 "
                     "off≈消费行数接缝 p90 −62%）。0=帧 0 重锚（新块首行接入）")
ap.add_argument("--cmd-first-only", action="store_true",
                help="仅首组合步（在役 --cmd-every N --cmd-first-only 同义，"
                     "须配 --merge >1；与 --blend 互斥）：每块拍 0 发一条组末行"
                     "（rows[N−1]，占 N 拍）消块界，组后回逐行 30Hz 流式密随")
ap.add_argument("--entry-match", action="store_true",
                help="入口动态匹配（用户 10-17 提案；与 install-offset/skip-retrace/"
                     "cmd-first-only/blend/glide/merge>1 全互斥）：新块到来读臂"
                     "实际关节位，①只取顺运动方向前进侧（反向行=摆幅来源，剔除）；"
                     "②前进侧里选 gap 最接近稳态导程 g₀ 的行做接入行——"
                     "set_joint_commands 速度∝gap 引擎自算不可控，衔接首拍 gap "
                     "对齐上一块末尾=接缝两侧速度连续（固定合步 1/4 首拍 gap≈"
                     "4 行距恰落稳态带内=其手感好的机制解释，本模式是其动态精确化）")
ap.add_argument("--entry-dir-tol", type=float, default=0.002,
                help="方向容差 rad（默认 0.002=2 mrad）：关节 |末目标−实际| 低于此"
                     "值视为静止不约束方向；行反向超过此容差才判违约")
ap.add_argument("--entry-gap-min", type=float, default=0.008,
                help="g₀ 夹紧下限 rad（默认 0.008=8 mrad，--entry-match 用）")
ap.add_argument("--entry-gap-max", type=float, default=0.035,
                help="g₀ 夹紧上限 rad（默认 0.035=35 mrad）。⚠必须有：参考速度"
                     "常快于臂追踪速度→块内导程单调恶化→中位数被滞后污染→"
                     "入口越跳越深→g₀ 更大的正反馈（10-17 首跑实证 j=11→46、"
                     "gap 29→138 mrad 失控）；夹紧后入口回到 j≈1-4 小步档")
ap.add_argument("--entry-dir-frac", type=float, default=0.5,
                help="主导向关节阈值（默认 0.5）：只约束 |方向| ≥ frac×max|方向| "
                     "的关节；全关节门被模型抖动打死（任一关节转向即全块违约，"
                     "首跑 13/26 块退行 0），转向中的小关节豁免")
ap.add_argument("--hold-ms", type=float, default=330.0,
                help="块间静止窗 ms（默认 330=历史口径）。⚠接缝 A/B 建议 0：固定"
                     "消费行数模式下 hold 让臂每块从静止零起步=人为顿挫节奏，"
                     "且回放周期翻倍（27 块 17.4s vs 录制 9.6s）；0=连续追踪"
                     "（在役节奏：11 拍/块无休止，推理时延由消费窗覆盖）")
ap.add_argument("--max-excursion", type=float, default=3.0,
                help="偏离起始位护栏 rad（任一关节超限即停）")
ap.add_argument("--envelope", type=str, default=None,
                help="包络护栏 json（臂 14 维钳位）。缺省=按回放轨迹落盒率从 "
                     "models/*/joint_envelope.json 自动配对任务（dump 未记 ckpt，"
                     "盒必须同任务）；并列/全低配时拒绝猜测，要求显式给出")
ap.add_argument("--envelope-margin", type=float, default=0.15,
                help="包络余量 rad（主脚本同义，默认 0.15）")
args = ap.parse_args()

TICK = 1.0 / args.fps
MERGE = max(1, args.merge)
if args.cmd_first_only and MERGE <= 1:
    raise SystemExit("--cmd-first-only 须配 --merge > 1（否则无合步组可言）")
if args.cmd_first_only and args.blend:
    raise SystemExit("--cmd-first-only 与 --blend 互斥：首组并粗 vs 块首重铺细步，"
                     "方向相反（主脚本同款互斥）")
if args.install_offset > 0 and args.skip_retrace:
    raise SystemExit("--install-offset 与 --skip-retrace 互斥（同为换块跳行策略，"
                     "install-offset=在役跳帧安装口径，skip-retrace=回溯段研究口径）")
if args.entry_match:
    _bad = [n for n, on in (("--install-offset", args.install_offset > 0),
                            ("--skip-retrace", bool(args.skip_retrace)),
                            ("--cmd-first-only", args.cmd_first_only),
                            ("--blend", args.blend),
                            ("--glide", args.glide > 0)) if on]
    if _bad:
        raise SystemExit(f"--entry-match 与 {','.join(_bad)} 互斥：入口匹配统一了"
                         "跳帧/锚定/合步三机制（陈旧行要么方向违约要么 gap 不匹配）")
    if MERGE > 1:
        raise SystemExit("--entry-match 只配逐行直发（--merge 1）——合步量由入口行"
                         "动态承担，不再写死")

# ── 日志全文落盘（q 退出走 os._exit 不刷缓冲，逐行 flush）──
_log_path = str(_ROOT / "logs" / time.strftime("chunk_replay_test_%Y%m%d_%H%M%S.log"))
pathlib.Path(_log_path).parent.mkdir(parents=True, exist_ok=True)
_log_fh = open(_log_path, "a", buffering=1, encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except ValueError:
                pass    # teardown 已关日志文件后的残余 flush

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except ValueError:
                pass


sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)
print(f"[log] 全文日志 → {_log_path}")


# ── q 退出（主脚本同款：pause/resume 让确认提示符处回车可用）──
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
            r, w = os.pipe()                # 信号直通管道：C 层收信号即写字节，
            os.set_blocking(w, False)       # 主线程卡死在 SDK C++ 调用也能退出
            signal.set_wakeup_fd(w)
            signal.signal(signal.SIGINT, lambda *_: None)
            signal.signal(signal.SIGTERM, lambda *_: None)
            self._wake_r = r
            self.active.set()
            threading.Thread(target=self._loop, daemon=True).start()
            print("（循环期间随时按 q 或 Ctrl-C 退出；确认提示符处用回车；急停第一优先级）")

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
                    os._exit(130)
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出（已下发目标可能仍在限速执行）")
                    self.restore()
                    os._exit(2)
            except Exception:
                self.restore()
                os._exit(3)   # 监听线程死了比静默更危险：宁可误退不可失控


robot = None
WATCH = QuitWatcher()


def _teardown():
    rob = globals().get("robot")
    if rob is not None:
        try:
            rob.request_shutdown()
            rob.wait_for_shutdown()
            rob.destroy()
        except Exception:
            pass
    WATCH.restore()
    try:
        _log_fh.close()
    except Exception:
        pass


def _fatal_hook(t, v, tb):
    """任何未捕获异常：关 SDK + 还原终端 + 硬退（SDK 残留线程裸退会段错误）。"""
    print(f"\n⛔ 异常退出: {t.__name__}: {v}")
    _teardown()
    os._exit(1)


sys.excepthook = _fatal_hook

# ── 数据装载：chunk delta → 绝对目标（主脚本 BASE_ARM+delta 同一换算）──
if args.chunks_dir is None:
    _cands = sorted(glob.glob(str(_ROOT / "logs" / "chunks_traj_*")))
    if not _cands:
        raise SystemExit("logs/ 下无 chunks_traj_* 目录，--chunks-dir 必填")
    args.chunks_dir = _cands[-1]        # 最新
files = sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.npz")))
if not files:
    raise SystemExit(f"目录无 chunk_*.npz：{args.chunks_dir}")
i0 = max(0, args.start - 1)
files = files[i0:]
n_blocks = len(files) if args.rounds <= 0 else min(args.rounds, len(files))
files = files[:n_blocks]
BLOCKS = []          # (n,14) 绝对目标；BASES=各块锚点（原跑预测时刻臂位）
BASES = []
META = []
for f in files:
    d = np.load(f)
    ch = np.asarray(d["chunk"], dtype=np.float32)         # (50,16)
    base = np.asarray(d["base_arm"], dtype=np.float32)    # (14,)
    BLOCKS.append(np.concatenate([ch[:, :7], ch[:, 8:15]], axis=1)
                  + base[None, :])
    BASES.append(base)
    META.append({"file": os.path.basename(f),
                 "infer_ms": float(d["infer_ms"]),
                 "fps": int(d["fps"])})
    if int(d["fps"]) != args.fps:
        print(f"⚠ {os.path.basename(f)} 落盘 fps={int(d['fps'])} ≠ --fps {args.fps}"
              "（播放节奏与录制时不同）")
print(f"[data] {args.chunks_dir} | 回放 {n_blocks}/{len(glob.glob(os.path.join(args.chunks_dir, 'chunk_*.npz')))} 块"
      f"（start={args.start}）")

# ── 换块弃尾计划（复现闭环水位语义：块在真实闭环里不是走满的）──
# 真实时序：本块消费到水位时抓观测（≈下块 BASE_ARM 对应的本块行=锚点行），
# 推理落地拍数内旧块继续走，落地即换块弃尾。锚点行 = 下块 BASE_ARM 与本块
# 绝对目标逐行最大关节距的 argmin（130403 实测残差 3-30 mrad，正是观测时刻）。
def compute_swaps():
    if args.rows_per_block == 0:
        return [len(b) for b in BLOCKS], []
    if args.rows_per_block is not None:
        return [min(args.rows_per_block, len(b)) for b in BLOCKS], []
    out, info = [], []
    for i in range(len(BLOCKS)):
        if i == len(BLOCKS) - 1:                 # 末块无后继，走满
            out.append(len(BLOCKS[i]))
            continue
        d = np.max(np.abs(BLOCKS[i] - BASES[i + 1][None, :]), axis=1)
        k = int(np.argmin(d))
        delay = int(np.ceil(META[i + 1]["infer_ms"] / (TICK * 1000.0)))
        n = min(len(BLOCKS[i]), k + delay + 1)   # 消费到第 k+delay 行（含）
        out.append(n)
        info.append((i, k, delay, float(d[k]) * 1000))
    return out, info


SWAPS, SWAP_INFO = compute_swaps()
if args.rows_per_block is None:
    for i, k, dl, r in SWAP_INFO:
        print(f"[swap] 块{i + 1}: 消费 {SWAPS[i]}/{len(BLOCKS[i])} 行"
              f"（锚点行{k} + 推理 {dl} 拍，锚点残差 {r:.0f} mrad，"
              f"弃尾 {len(BLOCKS[i]) - SWAPS[i]}）")

# ── 包络（主脚本同源 json；直发口径钳行）──
# ⚠ 盒必须与 dump 同任务（pick 盒配 place 轨迹=护栏错位）。dump 本身没记 ckpt，
# 且同工位各任务盒高度重叠（实测 pick/place/onlypick 对 place 轨迹落入率全同），
# 轨迹反推判别力为零——只认两条权威来源，绝不猜：
#   ① dump 目录 sidecar task.json（{"envelope": 路径, ...}，由原跑日志反填）
#   ② 显式 --envelope
_env_margin = args.envelope_margin


def _env14(path):
    with open(path) as f:
        _e = json.load(f)
    _inf = float("inf")
    lo = [(-_inf if v is None else float(v)) for v in _e["action_lo"]]
    hi = [(_inf if v is None else float(v)) for v in _e["action_hi"]]
    return (np.array(lo[:7] + lo[8:15], dtype=np.float32) - _env_margin,
            np.array(hi[:7] + hi[8:15], dtype=np.float32) + _env_margin)


ENV_LO14 = ENV_HI14 = None
_sidecar_p = os.path.join(args.chunks_dir, "task.json")
if args.envelope:
    if not os.path.exists(args.envelope):
        raise SystemExit(f"包络文件不存在：{args.envelope}")
    ENV_LO14, ENV_HI14 = _env14(args.envelope)
    print(f"[护栏] 包络（显式）：{args.envelope}（臂 14 维钳位，余量 ±{_env_margin}）")
elif os.path.exists(_sidecar_p):
    with open(_sidecar_p) as f:
        _sc = json.load(f)
    if not _sc.get("envelope") or not os.path.exists(_sc["envelope"]):
        raise SystemExit(f"sidecar {_sidecar_p} 的 envelope 字段缺失或文件不存在")
    ENV_LO14, ENV_HI14 = _env14(_sc["envelope"])
    print(f"[护栏] 包络（sidecar task.json）：{_sc['envelope']}"
          f"（任务 {_sc.get('task', '?')}，余量 ±{_env_margin}）")
else:
    raise SystemExit(
        f"无法确定包络盒任务归属：{args.chunks_dir} 缺 task.json 且未给 --envelope\n"
        "  ① 在 dump 目录放 sidecar task.json（{\"envelope\": 同任务 json 路径}）\n"
        "  ② 或加 --envelope 显式指定\n"
        "  （盒配错任务=护栏形同虚设，宁停勿猜）")


def build_rows(cur, tgt_blk, ema_init=None):
    """链式限幅（主脚本 build_rows 同构：每行从当前位向绝对目标 ≤±delta-max）。

    ema_init：上块末下发目标（EMA 跨块携带=主脚本 161933 修正同款——滤波态
    不在块界重置，接缝阶跃被抹成从旧目标出发的斜坡；None=首块从 rows[0] 起）。
    """
    rows = np.empty_like(tgt_blk)
    prev = np.asarray(cur, dtype=np.float32)
    for kk in range(len(tgt_blk)):
        prev = prev + np.clip(tgt_blk[kk] - prev,
                              -args.delta_max, args.delta_max)
        rows[kk] = prev
    if args.ema_alpha > 0:
        a = float(args.ema_alpha)
        if ema_init is None:
            acc = rows[0]
        else:
            acc = a * rows[0] + (1.0 - a) * np.asarray(ema_init, np.float32)
            rows[0] = acc
        for kk in range(1, len(rows)):
            acc = a * rows[kk] + (1.0 - a) * acc
            rows[kk] = acc
    return rows


def read_joints():
    v = robot.get_joint_positions([], ARM_NAMES)
    if not v or len(v) != len(ARM_NAMES):
        raise SystemExit(f"关节读取失败（返回 {len(v) if v else 0} 维）")
    return np.asarray([float(x) for x in v], dtype=np.float32)


def send_row(fr):
    """一行目标 = 一条 set_joint_commands（标准关节只消费 position）。"""
    _cmds = []
    for _ji in range(14):
        _c = JointCommand()
        _c.position = float(fr[_ji])
        _cmds.append(_c)
    return robot.set_joint_commands(_cmds, joint_names=ARM_NAMES,
                                    time_from_start_s=args.tfs_ms / 1000.0)


_mode_send = ("每行一令直发" if MERGE == 1
              else f"合步 {MERGE}"
              + ("·仅首组，组后流式" if args.cmd_first_only
                 else f"（每 {MERGE} 行一令，间隔 {MERGE} 拍）")) \
    + (f" | 跳帧安装@{args.install_offset}" if args.install_offset > 0 else "") \
    + (" | 入口动态匹配" if args.entry_match else "") \
    + (" | 跳回溯段" if args.skip_retrace else "") \
    + (f" | 滑入{args.glide}拍" if args.glide > 0 else "") \
    + (" | 前向搭接" if args.blend else "")
print(f"[mode] {'⚠ 真机下发' if args.do_exec else '干跑（零下发）'}"
      f" | tick {TICK * 1000:.1f} ms（{args.fps} Hz）| {_mode_send}"
      f" | 链式限幅 ±{args.delta_max} | tfs {args.tfs_ms:g} ms"
      f" | 漂移护栏 ±{args.max_excursion} rad | 块间 hold {args.hold_ms:g} ms")

if not args.do_exec:
    print("\n== [干跑] 逐块计划（零锚=0 位，真实首行导程以真机重锚为准）==")
    if MERGE > 1:
        print(f"[合步] N={MERGE}：每 {MERGE} 行一条指令（目标=组末端点），"
              f"指令间隔 {MERGE} 拍，块时长不变（同消费速率 A/B）")
    cur = np.zeros(14, dtype=np.float32)
    for bi, tgt in enumerate(BLOCKS):
        rows = build_rows(cur, tgt)[:SWAPS[bi]]
        if ENV_LO14 is not None:
            rows = np.clip(rows, ENV_LO14, ENV_HI14)
        d = np.abs(np.diff(np.vstack([cur[None], rows]), axis=0))
        print(f"  块{bi + 1} {META[bi]['file']}（原推理 {META[bi]['infer_ms']:.0f} ms）: "
              f"消费 {len(rows)}/{len(tgt)} 行 |Δcmd| mean {d[1:].mean() * 1000:.1f} "
              f"p95 {np.percentile(d[1:], 95) * 1000:.1f} "
              f"max {d[1:].max() * 1000:.1f} mrad")
        cur = rows[-1]
    print("[干跑] 零下发。真机：加 --exec（先确认臂在预热工作位）。")
    sys.exit(0)

# ── 真机路径 ──
print("\n== SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
if not robot.init():
    raise SystemExit("robot.init 失败")
time.sleep(2)
_cst = robot.start_controller("all")
print(f"start_controller('all') → {_cst}")
if not str(_cst).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止（先跑 run_g1_execute.py 排查）")

start_pos = read_joints()
print(f"起始位: {[round(float(v), 3) for v in start_pos]}")
print("⚠ 若当前不是预热工作位，先跑 g1_pose_warmup.py 再来")

# 起始就位：慢速到首块行 0 绝对目标（官方低频路径 0.2 rad/s，包络钳位后）
first_abs = BLOCKS[0][0]
if ENV_LO14 is not None:
    first_abs = np.clip(first_abs, ENV_LO14, ENV_HI14)
_dist = float(np.max(np.abs(first_abs - start_pos)))
if _dist > 0.02:
    print(f"[就位] 距首块行 0 {_dist * 1000:.0f} mrad → 0.2 rad/s 慢速移动…")
    _gst = robot.set_joint_positions([float(v) for v in first_abs],
                                     [], ARM_NAMES, True, 0.2, 30.0)
    if not str(_gst).startswith("ControlStatus.SUCCESS"):
        raise SystemExit(f"⛔ 就位未 SUCCESS（{_gst}）——臂可能没到轨迹起点，终止"
                         "（先跑 g1_pose_warmup.py 对齐工作位再来）")
    time.sleep(0.5)
else:
    print(f"[就位] 已在首块行 0 附近（{_dist * 1000:.0f} mrad），免移动")

WATCH.pause()
input(f"\n⚠ 将回放保存轨迹 {n_blocks} 块 × 50 行（无推理，纯 set_joint_commands "
      f"直发" + (f"，合步 {MERGE}" + ("·仅首组" if args.cmd_first_only else "")
                if MERGE > 1 else "")
      + (f"，跳帧安装@{args.install_offset}" if args.install_offset > 0 else "")
      + "；急停就绪后回车开始）...")
WATCH.resume()

t0 = time.perf_counter()
n_skip = 0
n_drop = 0
errs = []
acts = []
entries = []        # 入口匹配选中的行号（--entry-match 汇总用）
_gap_hist = []      # 实发拍导程滚动历史（g₀ 稳态导程估计源；免发拍不入）
_prev_ach = None
last_tgt = None
for bi, tgt_blk in enumerate(BLOCKS):
    cur = read_joints()
    rows = build_rows(cur, tgt_blk,
                      ema_init=(last_tgt if (bi > 0 and args.ema_alpha > 0)
                                else None))
    if ENV_LO14 is not None:
        rows = np.clip(rows, ENV_LO14, ENV_HI14)    # 直发口径钳行（主脚本帧级同款）
    _skip_note = ""
    if args.entry_match and bi > 0 and last_tgt is not None:
        # ── 入口动态匹配：方向硬约束 + 导程匹配（用户 10-17 提案）──
        # 基准=臂当前实际位 cur（非上块计划位，指令与实际有追踪误差）；
        # 方向=sign(末目标−实际)，|差|≤容差的关节视为静止不约束；
        # 可行行=所有运动关节均不反向（反向行=关节回摆=摆幅来源）；
        # 入口=可行行中 max 关节 gap 最接近 g₀（上一块末尾稳态导程）的行。
        if _gap_hist:
            _g0_raw = float(np.median(_gap_hist[-args.entry_gap_win:]))
            _g0 = float(np.clip(_g0_raw, args.entry_gap_min, args.entry_gap_max))
            _dir = last_tgt - cur
            _mag = np.abs(_dir)
            _dom = _mag >= args.entry_dir_frac * _mag.max()   # 主导向关节掩码
            _ds = np.sign(_dir)
            _rel = rows - cur[None]                       # (50,14)
            _back = (_dom[None, :]
                     & ((_rel * _ds[None, :]) < -args.entry_dir_tol))
            _viol = _back.any(axis=1)                     # 主导向关节有反向=违约
            _cand = np.where(~_viol)[0]
            _gaps = np.max(np.abs(_rel), axis=1)
            if len(_cand):
                _entry = int(_cand[np.argmin(np.abs(_gaps[_cand] - _g0))])
                rows = rows[_entry:_entry + SWAPS[bi]]    # 消费窗整体右移到入口
                _skip_note = (f" | 入口匹配 j={_entry}"
                              f"（g₀={_g0 * 1000:.1f}"
                              + (f"←{_g0_raw * 1000:.1f}夹紧" if _g0 != _g0_raw else "")
                              + f" gap={_gaps[_entry] * 1000:.1f}"
                              f" mrad，可行行 {len(_cand)}/{len(_rel)}）")
                entries.append(_entry)
            else:
                _entry = int(np.argmin(_back.sum(axis=1)))  # 反向量最小的行（勿退行0）
                rows = rows[_entry:_entry + SWAPS[bi]]
                _skip_note = (f" | 入口匹配部分违约 j={_entry}"
                              f"（反向量最小 {_back.sum(axis=1)[_entry]} 关节·行）")
                entries.append(_entry)
        else:
            rows = rows[:SWAPS[bi]]
            _skip_note = " | 入口匹配缺 g₀（上块全免发）→行 0"
    elif args.install_offset > 0 and bi > 0:
        _off = min(args.install_offset, len(rows) - 1)
        rows = rows[_off:_off + SWAPS[bi]]           # 跳帧安装=消费窗整体右移
        _skip_note = (f" | 跳帧安装{SWAPS[bi]}行@偏移{_off}"
                      f"（行{_off}-{_off + SWAPS[bi] - 1}）")
    else:
        rows = rows[:SWAPS[bi]]                      # 换块弃尾（auto=复现真实消费行数）
    _n_raw = len(rows)
    if args.skip_retrace and bi > 0:
        if args.skip_retrace == "delay":
            _skip = int(np.ceil(META[bi]["infer_ms"] / (TICK * 1000.0)))
        else:    # anchor：目标轨迹上最接近臂当前位的行，取前进侧最后一处
            d = np.max(np.abs(rows - cur[None]), axis=1)
            _cand = np.where(d <= d.min() + 0.003)[0]
            _skip = int(_cand[-1])
        _skip = max(0, min(_skip, len(rows) - MERGE - 1))
        if _skip > 0:
            rows = rows[_skip:]
            _skip_note = f" | 跳{args.skip_retrace}{_skip}行({_n_raw}→{len(rows)})"
    d = np.abs(np.diff(np.vstack([cur[None], rows]), axis=0))
    print(f"\n── 块 {bi + 1}/{n_blocks} {META[bi]['file']}（原推理 "
          f"{META[bi]['infer_ms']:.0f} ms）| 消费 {len(rows)}/{len(tgt_blk)} 行"
          + (f" | 合步 {MERGE}" if MERGE > 1 else "")
          + _skip_note
          + f" |Δcmd| mean {d[1:].mean() * 1000:.1f} max {d[1:].max() * 1000:.1f} mrad ──")

    _next_t = time.perf_counter()
    # ── 前向搭接（--blend）：旧链末目标 → rows[M−1] 直连，参考链全程前向 ──
    # 几何：计划从臂前 ~50 mrad 起以行距上升，M 行内爬回旧链高度（=lag）——
    # 直连路径单调前向，臂不停车不回摆；滞后不释放，自然并入块内正常滞后。
    _n_blend = 0
    if args.blend and bi > 0 and last_tgt is not None:
        d_all = np.max(np.abs(rows - cur[None]), axis=1)
        d_old = float(np.max(np.abs(last_tgt - cur)))
        _rise = np.where(d_all >= d_old - 0.01)[0]
        _m = (int(_rise[0]) + 1) if len(_rise) else len(rows) - 1
        _m = max(2, min(_m, len(rows) - MERGE - 1))
        if (d_old > 0.06 and d_old - float(d_all.min()) > 0.06
                and _m < len(rows) - 1):
            _bend = rows[_m - 1].copy()
            _bstep = (_bend - last_tgt) / _m   # 定步长前向向量，M 拍正好走满
            print(f"  搭接：旧链末目标（离臂 {d_old * 1000:.0f} mrad）→ 行{_m - 1}"
                  f"（离臂 {float(d_all[_m - 1]) * 1000:.0f}）前向直连 {_m} 拍"
                  f"（每拍 ≤ {float(np.max(np.abs(_bstep))) * 1000:.1f} mrad）")
            for gk in range(1, _m + 1):
                now = time.perf_counter()
                if now < _next_t:
                    time.sleep(_next_t - now)
                _next_t += TICK
                cur = read_joints()
                tgt = _bend.copy() if gk == _m else last_tgt + _bstep
                d_mrad = float(np.max(np.abs(tgt - last_tgt))) * 1000
                gap0 = float(np.max(np.abs(tgt - cur)))
                drift = float(np.max(np.abs(tgt - start_pos)))
                if drift > args.max_excursion:
                    print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                          f"{args.max_excursion * 1000:.0f}，停止（机械臂留在原地）")
                    _teardown()
                    os._exit(5)
                if gap0 < GAP_EPS:
                    _fr_s = "· 免发 ·"
                else:
                    _st = send_row(tgt)
                    if not str(_st).startswith("ControlStatus.SUCCESS"):
                        print(f"⛔ 下发非 SUCCESS（{_st}），停止")
                        _teardown()
                        os._exit(4)
                    _fr_s = f"1令@{args.fps}Hz"
                last_tgt = tgt.copy()
                ach = read_joints()
                act = (float(np.max(np.abs(ach - _prev_ach))) * 1000
                       if _prev_ach is not None else 0.0)
                _prev_ach = ach
                err = float(np.max(np.abs(ach - tgt))) * 1000
                errs.append(err)
                acts.append(act)
                print(f"  混{gk:2d}/{_m}: |Δcmd| {d_mrad:5.1f} mrad"
                      f" | 导程 {gap0 * 1000:5.1f} → {_fr_s}"
                      f" | 动 {act:4.1f} 残差 {err:4.1f} mrad")
            _n_blend = _m
    # ── 滑入：anchor 跳块后滞后分期释放（--glide N）──
    # 目标链从旧块末目标分 N 拍定步长滑到入口点 rows[0]：每拍指令步长 = 总差/N，
    # 拍0 |Δcmd| 从 ≈lag 降到 ≈lag/N；臂以 k×gap 指数减速无猛拉，无反向拉扯。
    if (args.glide > 0 and args.skip_retrace == "anchor" and bi > 0
            and float(np.max(np.abs(rows[0] - last_tgt))) > 0.06):
        _gdist = float(np.max(np.abs(rows[0] - last_tgt)))
        _gstep = (rows[0] - last_tgt) / args.glide   # 定步长向量，N 拍正好走满
        print(f"  滑入：滞后 {_gdist * 1000:.0f} mrad 分 {args.glide} 拍释放"
              f"（每拍 ≤ {float(np.max(np.abs(_gstep))) * 1000:.1f} mrad）")
        for gk in range(1, args.glide + 1):
            now = time.perf_counter()
            if now < _next_t:
                time.sleep(_next_t - now)
            _next_t += TICK
            cur = read_joints()
            tgt = rows[0].copy() if gk == args.glide else last_tgt + _gstep
            d_mrad = float(np.max(np.abs(tgt - last_tgt))) * 1000
            gap0 = float(np.max(np.abs(tgt - cur)))
            drift = float(np.max(np.abs(tgt - start_pos)))
            if drift > args.max_excursion:
                print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                      f"{args.max_excursion * 1000:.0f}，停止（机械臂留在原地）")
                _teardown()
                os._exit(5)
            if gap0 < GAP_EPS:
                _fr_s = "· 免发 ·"
            else:
                _st = send_row(tgt)
                if not str(_st).startswith("ControlStatus.SUCCESS"):
                    print(f"⛔ 下发非 SUCCESS（{_st}），停止")
                    _teardown()
                    os._exit(4)
                _fr_s = f"1令@{args.fps}Hz"
            last_tgt = tgt.copy()
            ach = read_joints()
            act = (float(np.max(np.abs(ach - _prev_ach))) * 1000
                   if _prev_ach is not None else 0.0)
            _prev_ach = ach
            err = float(np.max(np.abs(ach - tgt))) * 1000
            errs.append(err)
            acts.append(act)
            print(f"  滑{gk:2d}/{args.glide}: |Δcmd| {d_mrad:5.1f} mrad"
                  f" | 导程 {gap0 * 1000:5.1f} → {_fr_s}"
                  f" | 动 {act:4.1f} 残差 {err:4.1f} mrad")
    _stride = 1 if args.cmd_first_only else MERGE
    for kk in range(_n_blend, len(rows), _stride):
        _merged = args.cmd_first_only and kk == _n_blend   # 仅首组合步：块内首拍并组
        _silent = args.cmd_first_only and not _merged \
            and (kk - _n_blend) < MERGE                    # 首组其余拍静默消费
        if _silent:
            continue            # 墙钟已被拍 0 的 MERGE 拍一次推进覆盖，勿再 sleep+推进
        # ── 节拍头：绝对截止；落后超 1 拍即重锚（绝不连发追赶，主脚本同款）──
        now = time.perf_counter()
        if now < _next_t:
            time.sleep(_next_t - now)
        elif now - _next_t > TICK:
            n_drop += 1
            _next_t = now
        _next_t += TICK * (MERGE if _merged else _stride)  # 合步=一令占 N 拍墙钟
        # （同消费速率；bug 修复：此前恒 +TICK，merge>1 实际变成 N 倍速消耗，
        # 131901/131929 两跑属此口径）
        cur = read_joints()                      # 拍首读实际
        tgt = rows[min(kk + MERGE - 1, len(rows) - 1)] \
            if (_merged or not args.cmd_first_only) else rows[kk]   # 合步=组末行端点
        cmd_delta = ((tgt - last_tgt) if last_tgt is not None
                     else np.zeros_like(tgt))
        d_mrad = float(np.max(np.abs(cmd_delta))) * 1000
        gap0 = float(np.max(np.abs(tgt - cur)))
        drift = float(np.max(np.abs(tgt - start_pos)))
        if drift > args.max_excursion:
            print(f"⛔ 漂移护栏：关节最大偏离 {drift * 1000:.0f} mrad > "
                  f"{args.max_excursion * 1000:.0f}，停止（机械臂留在原地）")
            _teardown()
            os._exit(5)
        if gap0 < GAP_EPS:
            n_skip += 1
            _fr_s = "· 免发 ·"
        else:
            _st = send_row(tgt)
            if not str(_st).startswith("ControlStatus.SUCCESS"):
                print(f"⛔ 下发非 SUCCESS（{_st}），停止")
                _teardown()
                os._exit(4)
            _fr_s = f"1令@{args.fps}Hz"
            _gap_hist.append(gap0)                   # g₀ 源：只记实发拍（免发=已追平）
            del _gap_hist[:-32]                      # 有界滚动
        last_tgt = tgt.copy()
        ach = read_joints()
        if _prev_ach is not None:
            act = float(np.max(np.abs(ach - _prev_ach))) * 1000  # 0=SUCCESS 但僵住
        else:
            act = 0.0
        _prev_ach = ach
        err = float(np.max(np.abs(ach - tgt))) * 1000
        errs.append(err)
        acts.append(act)
        print(f"  拍{kk:2d}"
              + (f"(行{max(0, kk - MERGE + 1)}-{kk})"
                 if (MERGE > 1 and not args.cmd_first_only) else "")
              + ("[合步]" if _merged else "")
              + f": |Δcmd| {d_mrad:5.1f} mrad"
              f" | 导程 {gap0 * 1000:5.1f} → {_fr_s}"
              f" | 动 {act:4.1f} 残差 {err:4.1f} mrad")
    if bi < n_blocks - 1 and args.rows_per_block is not None:
        time.sleep(args.hold_ms / 1000.0)   # 默认 330=历史口径；接缝 A/B 建议 0
        # （hold 让臂每块从静止零起步=人为顿挫+回放周期翻倍，10-17 实测）；auto
        # 模式的等待已建模在延迟行里，真实闭环换块瞬间衔接（落地即换，无窗）

fin = read_joints()
_entry_s = (f" | 入口行 min/p50/max {min(entries)}/{int(np.percentile(entries, 50))}"
            f"/{max(entries)}（匹配 {len(entries)}/{n_blocks - 1} 块）"
            if entries else "")
print(f"\n== 汇总 == 块 {n_blocks} | 合步 {MERGE} | 免发 {n_skip} | 丢拍重锚 {n_drop}"
      f" | 残差 mean {np.mean(errs):.1f} p95 {np.percentile(errs, 95):.1f}"
      f" max {np.max(errs):.1f} mrad"
      f" | 动/令 mean {np.mean(acts):.1f} mrad"
      + _entry_s
      + f" | 等效臂速 mean {np.mean(acts) / (MERGE * TICK):.0f}"
      f" p95 {np.percentile(acts, 95) / (MERGE * TICK):.0f} mrad/s"
      f" | 最终偏离起始位 {float(np.max(np.abs(fin - start_pos))) * 1000:.0f} mrad"
      f"（护栏 {args.max_excursion * 1000:.0f}）"
      f" | 总耗时 {time.perf_counter() - t0:.1f} s")
_teardown()
os._exit(0)   # SDK 残留线程，干净退出
