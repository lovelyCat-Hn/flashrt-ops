#!/usr/bin/env python
"""S1 双臂闭环轨迹流执行器（G1 run_g1_loop_traj.py 移植版）。

由 run_g1_loop_traj.py 整文件移植（2026-10-29 定案）：行为逻辑（水位预取/
换块弃尾/星饿停机/链式限幅/EMA/换块融合/RTC 前缀引导/护栏/退出监听/遥测）
与 G1 版同源，仅替换下述机体件。

机体件差异：
  SDK        galbot_sdk.g1 → galbot_sdk.s1；爪组名 "left/right_gripper"
  state 27   [右臂7, 右爪%, 左臂7, 左爪%, torso, head2, swerve×8]（数据集序；
             爪宽米→0-100%；swerve 恒 0；torso=torso_lift_joint1 高度米）
  取图       get_rgb_data(JPEG)，HEAD_RIGHT/LEFT_ARM/RIGHT_ARM；冻结守卫常开
  发令       set_joint_commands_batch 带时间轴批（tfs 逐拍累计=墙钟节拍；
             WBC 按 tfs 细插，tfs=0 参考跳变→力矩跳变；--rows-per-cmd R>1
             跨拍攒窗；--cmd-every>1 硬拦）
  取图线程   fresh_obs 随推理入 worker 线程（--no-obs-worker 回退主线程）
  爪标定     ckpt 无 manifest gripper 段 → 出厂行程 0.007/0.11 m
             （告警：未逐台标定）
  warm       内嵌 s1_warm：臂+头 @0.15 rad/s → torso @0.2；姿态由
             config/s1.toml [warm_pose] 覆盖，缺省 policies/base.py 烘档；
             --exec 确认后就位，自然结束归位；q/Ctrl-C 立即退出不复位
  导航/护栏  无 nav 栈（--nav-suspend 保留为 no-op 兼容档键）；ckpt 无
             joint_envelope.json 时包络护栏自动关闭

权重族：--policy pi05（默认）| fineart_vla_s1（16/27 维 delta-除爪同语义
链，2026-11-02 接入；predict 包装层族通用，RTC 前缀引导同生效）。

配置：config/s1.toml，三态 CLI > toml > s1_config.BUILTIN；键名对齐
config/g1.toml [lerobot]，术语对照 docs/lerobot-alignment.md。现役工作点
（10-30 定案）：execute_actions_per_chunk 0（换块即预取）/ frames_per_row 1 ×
rows_per_cmd 1（下发静音的必要条件=指令带 tfs 时间轴，窗内点数/窗长/合批
均非自变量）/ filter_alpha 0.1 / aggregate_fn weighted_average（crossfade
0.3 旧 + 0.7 新，双层融合；latest_only=单层对照口径）/
max_joint_delta_per_step 0.05 / fps 30。

trained-RTC：--execution-horizon N>0 启用前缀引导——旧块未消费尾段重锚进
新块坐标系作前缀，去噪步把新块开头拉向旧计划；推理延迟按实测 p50 动态补偿。
仅对 RTC 训练权重开启（train_config rtc_training_max_delay>0）。

安全（--exec 连续真实驱动双臂）：
  1. 仅下发双臂 14 关节与双爪；torso/head/swerve 维永不下发
  2. 行目标链式限幅 ±max_joint_delta_per_step rad/行 + 逐帧包络钳位；
     漂移护栏超 --max-excursion 立即停机
  3. q 键即时退出（后台监听线程，SDK 阻塞中亦可退）；物理急停第一优先级
  4. --exec 双重确认：warm 就位前一次、进入控制环前一次

用法：
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
  ~/holy/scripts/s1/inference/run_s1_loop_traj.py --exec
配置缺省即现役配方；ckpt 缺省 = S1 pi05 工作点（010000）。
"""
# ── 显式 bootstrap：holy 仓库根入 path（勿依赖 s1_config 的 import 副作用；
#    append 保持脚本目录优先=不遮蔽同名包，同 s1_config.py 口径）──
import pathlib as _pathlib
import sys as _sys
_HOLY_ROOT = str(_pathlib.Path(__file__).resolve().parents[3])
if _HOLY_ROOT not in _sys.path:
    _sys.path.append(_HOLY_ROOT)
del _pathlib, _sys, _HOLY_ROOT

import argparse
import atexit
from collections import deque
import functools
import s1_config  # noqa: E402  同目录共享配置（CLI > config/s1.toml > 内置默认）
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

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--prompt", default=None, help="任务指令（默认 config [run].prompt，须用训练原句）")
ap.add_argument("--policy", default=None,
                help="权重族：pi05（默认）| fineart_vla_s1（S1 微调版 fineart，"
                     "16/27 维 delta-除爪同语义链；装载/stats 分支见模型段）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实连续驱动双臂（默认干跑只打印首块 5 拍目标；不进配置，仅 CLI）")
ap.add_argument("--rounds", type=int, default=None,
                help="换块次数（每块消费到水位即换，弃尾；config [loop].rounds）")
ap.add_argument("--execute-actions-per-chunk", "--n-action-steps",
                dest="n_action_steps", type=int, default=None,
                help="预取水位阈值：本块消费到第 K 步即触发新推理（lerobot "
                     "chunk_size_threshold 语义；≤ --chunk-size；"
                     "config [loop].n_action_steps；v3 里它是消费配额，这里只定换块时机）。"
                     "0=换块即预取（本块第 0 步下发后立即起跑下一块推理，块长≈推理"
                     "时延；须配 --skip-frames（原 --skip-install））")
ap.add_argument("--chunk-size", type=int, default=50,
                help="chunk 长度（lerobot chunk_size），默认 50=训练原生长度"
                     "（env FLASH_RT_PI05_ACTION_CHUNK_SIZE 自动设）")
ap.add_argument("--max-joint-delta-per-step", "--delta-max",
                dest="delta_max", type=float, default=None,
                help="每行目标限幅 rad（链式：每行至多向 chunk 行绝对位挪这么多；"
                     "config [loop].delta_max）")
ap.add_argument("--fps", type=int, default=None,
                help="行节拍频率 Hz（默认 30=训练控制率，config 键 ctrl_hz）："
                     "全局节奏旋钮——tick、计划播放速度（模型 30Hz 语义，"
                     "40=1.33× 提速 OOD）、下发率三者同缩放。>40 超频区放行+警告"
                     "（2026-10-26 用户申请 50Hz 实验）：tick 低于取图+SDK 硬开销 "
                     "~25ms 会丢拍重锚；>100 硬拒。⚠ S1 降速=时间维 OOD（G1 15Hz "
                     "事故同款），勿用本旗标降速")
ap.add_argument("--frames-per-row", type=int, default=None,
                help="每行插值帧数：行目标样条插成 N 帧于 row-tick 内成一包"
                     "带时间轴批下发（S1 移植版批形态，见文件头④）。默认见 "
                     "config（S1 现役 1=单点直发但带 tfs=T 斜坡时间轴——"
                     "10-30 五角定案时间轴存在性=唯一必要条件；fr2~8=样条"
                     "弧线纹理备档。G1 traj 默认 1=tfs=0 单点流，S1 勿抄）")
ap.add_argument("--rows-per-cmd", type=int, default=None,
                help="跨行攒批行数 R（10-29 审查②）：R=1=每拍一包（行内 fr 帧，"
                     "窗口 33.3ms）【现役最简，10-30 定案】；R>1=攒 R 拍帧并一包 "
                     "set_joint_commands_batch，"
                     "tfs=fdt·(i+1) 铺满 R×tick 窗（R2=66.7ms）、每 R 拍一发——"
                     "命令窗口与行周期解耦，与适配器执行器/回放台真机验证过的 "
                     "R×fr 形状同款。换块=半窗攒帧随旧块弃尾；星饿/收尾=先冲账")
ap.add_argument("--no-obs-worker", dest="obs_worker", action="store_false",
                help="取图/读关节回主线程（10-29 审查①的回退门：默认取图搬进"
                     "推理工作线程，主线程同步取图 p50 ~86ms≈2.6 拍=每块预取拍"
                     "丢拍重锚；SDK 双线程并发未真机验证，异常时用本旗标一键"
                     "回退旧行为对照)")
ap.add_argument("--cmd-every", type=int, default=None,
                help="合步：每 N 拍并 1 条指令（拍 k 下发组末行 rows[k+N−1]，"
                     "中间行只消费不发送），指令沿 30→30/N Hz——阶跃≈N 行差"
                     "（链钳 ±delta-max 逐行封顶不变，合计 ≤N×50 mrad），臂以"
                     "指数追踪大导程平滑扫过=切块误差被导程淹没（A 线 spc 合步"
                     "同机制；native 七跑定案抖感∝指令沿频率）。默认 1=纯 30Hz"
                     "直发零变；与 --blend 互斥")
ap.add_argument("--cmd-first-only", action="store_true",
                help="合步只并每块首组（rows 0..N−1 一条指令消块界，该拍组差本"
                     "来就小 ~12-20 mrad），其余拍回 30Hz 逐拍流式密随——均匀合"
                     "步的组间大步 85-105 mrad 太肥（20261016 实录：块界治好但"
                     "误差搬进块内），流式 Pursuit 会把首组滞后自然吃回；须配 "
                     "--cmd-every>1，组后爪/臂恢复逐拍")
ap.add_argument("--cmd-every-ramp", type=str, default="",
                help="缝侧细笔画：换块后前几组的组宽序列，逗号分隔（如 '1,1,2'=组"
                     "0/1/2 各并 1/1/2 行、组 3 起回落 --cmd-every；默认空=关恒 N）。"
                     "动机（10-18 N 扫描）：缝上方向修正所有 N 同幅（重规划修正本身），"
                     "显形与否∝渲染笔画粗细（拍≥1 p50：N=2 11.5 vs N=4 29.4）——"
                     "缝侧 1/1/2 行细笔把修正角画成圆滑拐弯，组 3 起仍吃 N=4 导程"
                     "（干脆度/爪沿不动）；顺带修 entry-match 锚定被首组 tgi=k+N−1 "
                     "吃掉 N−1 行的精度损失（组宽 1 时下发=锚行 j 本身）。"
                     "与 --cmd-first-only 互斥")
ap.add_argument("--merge", type=int, default=1,
                help="合步（等时粗化，replay 台 3b40d0e 同构移植）：滤波后每 N 行"
                     "取 1 个宏步目标（末行必保），每宏步占 N 拍、宏步段 "
                     "frames-per-row 帧按拍均分 → 下发 ≈fps×frames-per-row/N Hz；"
                     "同轨迹同速同墙钟只稀疏化目标，EMA/链钳/参考连续保持行级。"
                     "与 --cmd-every/--cmd-first-only/--cmd-every-ramp/--entry-"
                     "match/--blend/--ref-offset/--seam-overlap-alpha 互斥；拍线"
                     " |Δcmd| 自动切帧流口径。默认 1=关")
ap.add_argument("--continuous-ref", action="store_true",
                help="换块参考链从 last_tgt 接续（链钳锚+EMA 不回臂位重锚；"
                     "replay 台同款语义，10-26 回放 A/B=块首 |Δcmd| 124→12、"
                     "残差 p50 −56%%）。与 --entry-match/--ref-offset 互斥（同管"
                     "换块链锚）；--blend/--skip-install 自带接续无需同开。默认关")
ap.add_argument("--entry-match", action="store_true",
                help="入口动态匹配换块（10-17 回放台 entry-match 移植，用户提案；"
                     "与 --skip-frames/--ref-offset/--blend 互斥——本模式取代时钟"
                     "跳帧，按位姿选行；10-18 起可与合步 --cmd-every 组合：入口管"
                     "从哪行接入，合步管怎么消费，正交）："
                     "新块到来读臂实际关节位，①顺运动方向（主导关节不倒行）；"
                     "②命令流单调（新目标须在上拍目标前进侧——追账段 gap 往小"
                     "匹配=反向阶跃=伺服急停反转，171925 实证）；③首拍导程对齐"
                     "稳态导程 g₀（近 N 次实发中位夹紧带 [min,max]）=接缝两侧"
                     "速度连续；④前跳超 --entry-j-cap=内容落后→退行 0（EMA "
                     "阻尼小步），防开环深跳棘轮；闭环新鲜重规划下内容落后罕见")
ap.add_argument("--entry-dir-tol", type=float, default=0.002,
                help="方向容差 rad（默认 2 mrad）：低于此值视为静止不约束方向")
ap.add_argument("--entry-gap-win", type=int, default=4,
                help="稳态导程 g₀ 中位窗口（默认 4 次实发；免发拍不计）")
ap.add_argument("--entry-gap-min", type=float, default=0.008,
                help="g₀ 夹紧下限 rad（默认 8 mrad）")
ap.add_argument("--entry-gap-max", type=float, default=0.035,
                help="g₀ 夹紧上限 rad（默认 35 mrad）：⚠必须——参考速度常快于臂"
                     "追踪，块内导程单调恶化会经中位窗污染 g₀ 形成正反馈"
                     "（回放台首跑实证 j=11→46、gap 29→138 失控）")
ap.add_argument("--entry-dir-frac", type=float, default=0.5,
                help="主导向关节阈值（默认 0.5）：只约束 |方向|≥frac×max 的关节，"
                     "转向中的小关节豁免（全关节门被模型抖动打死，回放台首跑 "
                     "13/26 块退行 0）")
ap.add_argument("--entry-dir-floor", type=float, default=0.005,
                help="显著运动地板 rad（默认 5 mrad）：|参考-臂差|低于地板的关节"
                     "不设方向门，全臂低于地板=门全免（平静段噪声关节当主导会把"
                     "可行行清零→假回退→块内呼吸，180940 实证）；0=关地板")
ap.add_argument("--entry-no-dir-gate", action="store_true",
                help="入口匹配全关方向门，纯 gap 匹配（argmin|gap−g₀| 恒有解）。"
                     "方向门两代各有「全窗不可行」死帧：符号门锁过冲帧（180940）、"
                     "绝对门锁拖曳帧（181946 参考钉臂前 40-98 mrad 脉冲）；闭环里"
                     "防反向阶跃交回 α 阶梯阻尼（≤0.12×步长）+cap 兜底")
ap.add_argument("--entry-fill-dip", action="store_true",
                help="凹陷填补（10-18 用户提案）：|gap−g₀| 绝对值匹配可选中参考后"
                     "方的行=首拍倒退，块内短促倒退段也照样入流=残摆源；按关节把"
                     "回补窗内的短倒退凹陷线性桥接填平（取 0 持位/镜像取正侧都是"
                     "桥的特例），须配 --entry-match")
ap.add_argument("--entry-fill-tol", type=float, default=0.002,
                help="凹陷判定的倒退步阈 rad（默认 2 mrad，与方向门 tol 同源）")
ap.add_argument("--entry-fill-max", type=int, default=15,
                help="凹陷回补搜索窗（行，默认 15）：窗内回到坑前水平=桥接填平；"
                     "超窗=持续后撤（真倒退）放行不改")
ap.add_argument("--entry-fwd-bias", type=float, default=0.0,
                help="前向偏好软代价 ε（默认 0=关，建议 0.5，10-18 用户提案）："
                     "argmin 目标改 |gap−g₀| + ε×后方越界量（主导关节 max(0,"
                     "last_tgt−plan) 最大值；151241 修正：全关节计罚会把臂落后"
                     "的导程常数抬进所有浅行=挑深不挑向）——绝对值匹配分不清参考"
                     "两侧同距行，后方侧=首拍倒退（EMA 阻尼≤0.12× 但非零）；"
                     "软偏好非硬门（无死帧），挑「方向相同又绝对误差最近」的行")
ap.add_argument("--entry-j-cap", type=int, default=12,
                help="入口前跳深度上限（行）：超限=内容落后→退行 0（EMA 阻尼"
                     "小步）；深跳消费块尾行→下块内容更落后→棘轮（回放台 "
                     "172802 实证 j 爬到 39、残差 25.5→41）")
ap.add_argument("--seam-alpha", type=float, default=0.0,
                help="接缝窗 EMA α 起点（0=关，默认=主 α）：换块后前 --seam-beats "
                     "拍 α 从此值线性爬回主 α（阶梯勿平切——硬切会把积压滞后"
                     "一次放出成新凸点）。削块首 EMA 衰减尾峰值，稳态零税")
ap.add_argument("--seam-beats", type=int, default=4,
                help="接缝窗拍数=α 爬坡长（默认 4）；--seam-alpha 0 时无效")
ap.add_argument("--seam-boost-alpha", type=float, default=0.0,
                help="接缝 α 助推起点（0=关，默认）：与 --seam-alpha 反向的块首"
                     "α 调度——换块 EMA 重播回 last_tgt 清零了速度状态，命令速"
                     "度块首=α×模型速度、τ=1/α≈整块（水位0 短块永远在瞬态，"
                     "160323 停顿案）；块首几拍用大 α（如 0.5）快速重建速度，"
                     "此后线性落回主 α。稳态纹理(主 α)与缝口速度两头兼得；"
                     "α=0.3 全程版对照=残差 p50 11.6→20.7 纹理粗 50%%（161429）"
                     "勿再走。与 --seam-alpha 互斥，须 --filter-alpha>0 且小于"
                     "本值")
ap.add_argument("--seam-boost-beats", type=int, default=4,
                help="接缝助推拍数=α 从 boost 落回主 α 的窗长（默认 4）；"
                     "--seam-boost-alpha 0 时无效")
ap.add_argument("--grip-rawrows", action="store_true",
                help="merge>1 时爪维默认随臂同管线（2026-10-26 用户定案：dim7/15 "
                     "与臂同 idx 宏步粗化+同款样条逐帧插值，EMA 上移到行准备、"
                     "换块从最后实发意图重播）；本旗标恢复旧行率流"
                     "（每发送拍发 chunk[k] 原值，A/B 用）。merge=1 下无差别")
ap.add_argument("--aggregate-fn", dest="aggregate_fn",
                choices=("weighted_average", "latest_only", "average",
                         "conservative"),
                help="换块融合函数名（上游 lerobot async_inference/configs.py "
                     "AGGREGATE_FUNCTIONS 注册表逐名对应，α=新计划权重）："
                     "weighted_average=0.7【官方默认，固定 λ 0.3 旧+0.7 新】/"
                     "latest_only=1.0（=混接关，新块全量）/average=0.5/"
                     "conservative=0.3（旧主导——灌爪冻 4mm 案配方，勿用）。"
                     "连续 α 覆盖=A/B 逃生门 --fusion-alpha（显式给则赢过本旗标）")
ap.add_argument("--fusion-alpha", "--seam-overlap-alpha",
                dest="seam_overlap_alpha", type=float, default=None,
                help="重叠混接权重（新计划权重，0=关；默认见 [lerobot] 档）：同事 lerobot 栈"
                     "weighted_average 移植——换块时保留旧块未执行行，按墙钟"
                     "对齐（skip-install 偏移）与新旧计划逐行 0.3·旧+0.7·新混合，"
                     "重叠区结束后纯新计划；每行执行值=两代计划凸组合，接缝不再"
                     "是事件。取代 --entry-match/--seam-alpha/--blend/--ref-offset"
                     "（互斥校验）；须配 --skip-frames（时间槽对齐）。爪维 dim7/15"
                     "同窗同权混合。2026-10-20 A/B 候审")
ap.add_argument("--burst-reserve-ms", type=float, default=None,
                help="row-tick 内留给读关节/打印/换块的毫秒数：帧突发配速预算="
                     "tick−该值（默认见 [lerobot] 档）")
ap.add_argument("--interp", choices=("linear", "cubic", "quintic"), default="cubic",
                help="行间样条（s1_traj_interp，移植自同事 infer_policy 真机在役"
                     "实现；默认 cubic natural 样条 C2 连续无冲击）")
ap.add_argument("--filter-alpha", "--ema-alpha",
                dest="ema_alpha", type=float, default=None,
                help="行目标 EMA 低通系数（0=关；默认见 [lerobot] 档；借自同事 action_filter "
                     "lowpass）：rows[k]←α·raw[k]+(1−α)·rows[k−1]，状态跨块连续"
                     "（接缝首行同样被滤，infer_policy 同款语义；10-12 判明块内"
                     "重启版对接缝无效）——接缝/深尾垃圾的执行侧兜底杠杆；首跑"
                     "先 0 判读样条本身")
ap.add_argument("--grip-filter-alpha", "--grip-ema-alpha",
                dest="grip_ema_alpha", type=float, default=None,
                help="爪指令 EMA 低通系数。默认 None=跟随 --filter-alpha（10-21 "
                     "定档：同事 action_filter=dim23 全维单滤镜 robot_controller"
                     ".py:110，臂+爪一个 α 一致性；joint 路爪也走 "
                     "set_joint_commands，分旋钮是分路时代遗产）。显式传值=爪单独"
                     "覆盖（A/B 用），0=爪不滤。grip%% 流逐拍平滑后再进发送判定"
                     "（治模型爪列翻转：125953 实证 EMA 滤反转即治本）")
ap.add_argument("--grip-ss", type=str, default="",
                help="爪状态机「开阈,合阈」如 50,35（空=关，默认）：平滑后爪%%"
                     "≥开阈→置开态发一次、≤合阈→置合态发一次、带内不发；无翻转"
                     "时平滑值漂离上次实发 ≥15 点且距上次下发 ≥2.0s 才补发一条"
                     "（中间宽度意图）。机制依据（111901 终审）："
                     "set_gripper_command 每条新指令把行程从头重启（起步死区 "
                     "~0.9s，单发标定台实证），循环流 p50 间隙 2-3 拍=死区永不"
                     "耗尽→110 条 send 全 SUCCESS 但实宽全程 0.5mm 冻结；q 停流"
                     "静→最后一条 1.5s 走满=「q 后突然张开」。治法=少发：状态机"
                     "每相 1-2 条，每条都拿到 ≥2s 静默行程窗")
ap.add_argument("--starve-limit", type=float, default=2.0,
                help="星饿安全停机阈值 s：块耗尽且推理未归超过该时长即停循环"
                     "（默认 2.0s=正常推理余量 ~6×；星饿期不下发=伺服保持）")
ap.add_argument("--blend-gate", type=float, default=0.06,
                help="搭接触发门（mrad→rad，默认 0.06=10-12 回放台定案值）："
                     "换块滞后/回撤深度超过此值即前向搭接。10-12 EMA+blend "
                     "真机发现块间瞬态（EMA 重锚回撤 α×(L−s) + 收敛慢段 + "
                     "重规划 bend）在 <60 mrad 换块上裸过门=块间顿挫节奏"
                     "（水位 10 下 ~1.4 Hz），降门（如 0.02）让搭接直线接管"
                     "全部有效换块——参考链全程等速无起步慢段")
ap.add_argument("--skip-frames", "--skip-install",
                dest="skip_install", action="store_true", default=None,
                help="换块跳帧安装（同事 infer_policy rtc_controller.add_chunk "
                     "skip_frames 墙钟对齐同款，2026-10-13 移植）：新块落地时丢掉"
                     "前「观测拍摄→落地」流逝拍数行，从当前墙钟槽位接续。行 0 重锚"
                     "口径下这段是模型对已走时隙的重绘（回溯段），消费它=块界"
                     "「减速→冲刺」脉动的构造性根源；跳帧后时间槽对齐，回溯段不进"
                     "消费链。110411 真机追补：光跳帧不够——链式限幅每块从臂位"
                     "重锚（拍0 导程钳回 50，锯齿 50→330→50）才是脉动残留源，"
                     "故本开关同时让换块链从 last_tgt 接续（参考连续，同同事"
                     "「无重锚」语义）。默认关=行 0 重锚现役口径。"
                     "与回放台 --skip-retrace 无关（那是离线概念）")
ap.add_argument("--ref-offset", action="store_true",
                help="偏移平移（配合 --skip-frames，10-13 120045 实机后续杠杆）："
                     "新计划锚=观测滞后臂，行 0 天然落后参考一个失配量（=块首回走"
                     "刹车的来源）。把整条新计划平移 offset=last_tgt−plan[0]（逐维"
                     "钳 ±delta-max）再接参考链——块首零缝步零回走，臂速无块界刹车，"
                     "导程呼吸趋平。钳幅 ≤delta-max 不吃穿包络余量（±0.15）；超出"
                     "部分仍按 ±delta-max 回走收敛。EMA 可叠加（块内滤抖）。"
                     "与 --merge/--continuous-ref 可叠加（offset 在宏步粗化"
                     "上游生效）；仍需 --skip-frames。默认关")
ap.add_argument("--blend", action="store_true",
                help="换块前向搭接（10-12 回放台定案：无回摆模式最优，残差 75.5 "
                     "vs anchor 97/基线 69）：新块前 M 行不发原目标，改发旧链末"
                     "目标→rows[M−1] 直连（M 自适应=计划首次爬回旧链高度行号+1；"
                     "门：滞后 >60 mrad 才触发）——消回溯俯冲与块首急停，参考链"
                     "全程前向，臂不停车不回摆；1:1 替换不加拍不占墙钟，末行=原 "
                     "rows[M−1] 与 EMA 状态链一致。与 --execution-horizon 叠加属双重"
                     "消缝（推理侧+执行侧），A/B 时分开评估")
ap.add_argument("--execution-horizon", "--rtc-horizon",
                dest="rtc_horizon", type=int, default=None,
                help="推理侧 RTC 前缀引导窗口（行数）：>0 启用（lerobot 默认 10），"
                     "0=关；默认见 [lerobot] 档。前缀=旧块未消费尾段重锚进新块坐标系")
ap.add_argument("--rtc-max-w", type=float, default=10.0,
                help="RTC 最大引导权重（lerobot max_guidance_weight 默认 10）")
ap.add_argument("--head-crop", action="store_true", default=None,
                help="头相机观测追加训练同款裁剪放大（letterbox224 → 裁 "
                     "[126:214,52:172] → Lanczos4 拉回 164×224 → 上下补 30px "
                     "黑条）。训练头图带此处理的 ckpt（sft_1012 系："
                     "sft1012_training_rtc / dropout20）推理必带，漏做="
                     "train/serve 图像分布失配（10-15 place 相 R 爪不放同款病）；"
                     "默认关=在役 letterbox224 行为零变")
ap.add_argument("--tfs-ms", type=float, default=0.0,
                help="每帧 time_from_start_s 毫秒数：0=尽快到达（官方语义 fastest "
                     "arrival，官方 100Hz 流式示例同值，默认）；>0=期望到达时限"
                     "（expected arrival time，galbot_robot.hpp L148）——≈帧间隔 "
                     "4.2ms@30fps 即限速到帧节奏；标准关节只消费 position，tfs 是"
                     "唯一配速输入")
ap.add_argument("--dump-chunks", action="store_true",
                help="每块推理落地时把原始 chunk（50,16）+BASE_ARM 落盘到 "
                     "logs/chunks_traj_<ts>/（供离线增量分析与回放台复喂；"
                     "包络截断前落盘=纯模型输出）")
ap.add_argument("--dump-frames", action="store_true",
                help="每次推理把三路 letterbox224 后的模型实际输入图落盘到 "
                     "logs/frames_traj_<ts>/（seq_编号_视图名.png；编号即推理序，"
                     "与日志块序一一对应，首块=0001）。排查黑帧/冻结/图像失配；"
                     "写盘在推理后台线程、不占控制节拍。注意磁盘：~0.4MB/推理")
ap.add_argument("--dump-frames-dir", default=None,
                help="帧落盘目录（默认 logs/frames_traj_<ts>/，与日志同时间戳）")
ap.add_argument("--cache-frames", type=int, default=None,
                help="K/V 时序复用周期：1=每帧全量（默认，无损）；config [loop].cache_frames")
ap.add_argument("--max-excursion", type=float, default=None,
                help="偏离起始位护栏 rad（任一关节超限即停；config [loop].max_excursion）")
ap.add_argument("--grip", action="store_true", default=None,
                help="启用夹爪下发（dim7/dim15 0~100%% → manifest 标定宽度；config [gripper].enabled）")
ap.add_argument("--no-grip", action="store_true",
                help="显式关闭夹爪下发（覆盖 config 的 gripper.enabled=true）")
ap.add_argument("--grip-speed", type=float, default=None, help="夹爪速度 m/s（config [gripper].speed）")
ap.add_argument("--grip-effort", type=float, default=None, help="夹爪力矩 N（config [gripper].effort）")
ap.add_argument("--grip-chg", type=float, default=None,
                help="夹爪下发变化阈值 %%（config [gripper].chg）")
ap.add_argument("--grip-send-every", type=int, default=0,
                help="爪定时静默制：每 N 个机会拍强制发 1 条当前 EMA 值，chg 死区门"
                     "整体旁路（0=关，走旧 chg 门）。2026-10-20 静默定律：爪只在指令"
                     "静默期执行、每条新指令打断在途行程从头重来——chg=0（无死区）"
                     "30Hz 密发实证完全冻结（143831），chg=2 密发滞留爬行（135828）。"
                     "N=4→≈7.5Hz/静默 75%%。R/L 同拍判定，同拍配对不变。⚠ 10-20 晚"
                     "定律终版补判决：无条件周期拍隙 N/机会拍率 <0.9s 死区时从静止"
                     "永起步不了（合步2 N=4→0.27s=永冻），要用须 N≥18（隙≥1.2s）。")
ap.add_argument("--grip-min-interval", type=float, default=0.0,
                help="爪事件门 s（2026-10-20 极限梯定律版）：距上次实发 ≥N 秒才发"
                     "下一条。固件起步延迟 onset=0.12s+行程/速度（0.15 m/s 全行程"
                     "119.5mm=0.91s），N=1.2>0.91 → 下一条来时上一条必已在途/完成，"
                     "吞令重摆结构性不可能；间隔未满时仅「在途(|v|>0.02)且意图差"
                     "≥3%%」豁免即时发（P2/P4 实锤：在途同向/反向均平滑零罚，"
                     "183232 反向掉头仅 ~0.2s 不碰零）。0=关（走旧 chg 门）。"
                     "推荐 1.2。与 --grip-send-every/--grip-ss 互斥。")
ap.add_argument("--grip-preclose", type=float, default=0.0,
                help="【已判死 185317 极限梯，勿启用】固件速度下限 0.03 m/s"
                     "（0.01 指令实测执行 0.03）且起步延迟 onset=0.12s+行程/速度"
                     "——爬速 0.03 全行程起步要 ~4.1s，爬速越慢起步越晚，"
                     "「低速吸收死区」前提整体不成立。旗标保留仅供复现对照。0=关。")
ap.add_argument("--grip-path", choices=["cmd", "joint"], default="joint",
                help="爪下发路径（10-21 P6 破案+用户定案默认 joint：onset=0.12s+"
                     "行程/速度＋吞令整条定律属 set_gripper_command 端点专属，"
                     "该路=训练外注入行为；joint=dim7/15 并回 set_joint_commands "
                     "关节流=lerobot 采集/同事 infer_policy 本体语义，P6 三格实证 "
                     "onset≈0.14s 纯 L 与行程无关、velocity=速度帽（0.15→0.1500 "
                     "被尊重）、密流不冻；双爪一包每发送拍一包，各门全旁路、"
                     "EMA 照带、effort=10 同事同款）。"
                     "cmd=set_gripper_command【降级=逃生门/复现对照】——爪各门"
                     "（chg/min-interval/ss/send-every/preclose）只在此路适用，"
                     "joint 路无该定律、门无对象（传了会被拒）。"
                     "速度下限 0.03 m/s 双路径共享（电机级），0.15 不受影响。"
                     "默认 joint。")
ap.add_argument("--grip-hold-on-stall", action="store_true", default=False,
                help="憋死保持（2026-10-21 力矩碰撞案；仅 --grip-path joint）："
                     "抓到物体后模型按训练语义持续发全闭指令，位置伺服以有限速度"
                     "反复撞进物体（每拍重发=斜坡重启≥0.03 m/s）。本机制检测"
                     "「指令显著深于实宽且实宽平台（物理被撑住）」→ 爪流改发"
                     "实宽−margin（钳位保持，位置误差≈margin=轻微恒定夹紧），"
                     "消反复撞击；模型指令上穿实宽+exit（想开爪）即恢复流式。"
                     "判据用每拍新鲜物理回读（p50 10ms）。默认关。")
ap.add_argument("--grip-hold-gap-mm", type=float, default=5.0,
                help="保持进入门 mm：指令宽度 ≤ 实宽−gap 才算「深指令」（"
                     "防 EMA 尾部微抖误触发）。默认 5.0")
ap.add_argument("--grip-hold-exit-mm", type=float, default=3.0,
                help="保持退出门 mm：指令宽度 ≥ 实宽+exit=模型想开爪，恢复流式"
                     "并清平台史（防近阈抖动闪断）。默认 3.0")
ap.add_argument("--grip-hold-margin-mm", type=float, default=1.0,
                help="保持段钳位余量 mm：下发目标=实宽−margin=保有的恒定"
                     "夹紧量=稳态位置误差（观测 093335：误差即伺服持续内顶"
                     "的驱动力，3mm 仍可闻声）。0=零夹紧物易松脱。默认 1.0")
ap.add_argument("--grip-hold-vel", type=float, default=0.05,
                help="保持段下发速度帽 m/s：每拍重武装的合爪斜坡速度，小=顶"
                     "得轻（电机下限 0.03，0.05 留余量；0=沿用 GRIP 速度帽"
                     "0.15）。默认 0.05")
ap.add_argument("--grip-hold-relax-mm", type=float, default=0.3,
                help="保持段 eff 闭环自动外松步长 mm（执行侧导纳仿真）。"
                     "095207 实证：1mm 稳态误差下伺服仍满驱（eff 恒贴饱和轨"
                     "100，156/159 拍）=「有差就追」是位置伺服天性，收小 "
                     "margin 治不了声。SDK 无阻抗模式可切（10-21 五路彻查"
                     "在案），只能在目标侧松：eff≥阈值持续 6 拍(0.2s)→冻结"
                     "目标外移一步，直到 eff 掉头或目标=进入时实宽（零误差"
                     "=伺服停追=静音）。单向只松不紧=无棘轮。0=关。默认 0.3")
ap.add_argument("--grip-hold-relax-eff", type=float, default=90.0,
                help="自动外松触发阈值（eff 回读是非牛顿负载/堵转轨指示，"
                     "勿按力判读）：保持段连续 6 拍 ≥ 此值才外松一步；40-49"
                     "级瞬态不触发。嫌物松脱=调高（≥100 等效关）；仍响="
                     "调低（更早松）。默认 90")
ap.add_argument("--grip-hold-quiet-n", type=int, default=9,
                help="保持段真静默制（101418 定案）：外松到零误差后 eff 仍"
                     "恒贴轨 100+响——每拍重发冻结目标虽逐位相同，但 10-20 "
                     "爪定律=每条新令打断在途从头重来，密发把爪永远钉在起步"
                     "段 settle 不了。本机制：进 hold/每次外松步后只发 N 拍"
                     "（0.3s，够走完 onset 0.14s+最后一程），之后该爪退出"
                     "指令包=真静默，让最后一条走完收工；读回/退出/外松评估"
                     "照常（不依赖发送），退出或外松步即重新武装 N 拍。"
                     "0=关（每拍重发）。默认 9")
ap.add_argument("--gripj-every", type=int, default=1,
                help="爪通道单独节流 N（臂不受影响，照旧 30Hz 直发；爪每 N "
                     "拍发 1 条）。10-22 合步 A/B 判读：合步对爪正向（发送"
                     "减半=「每令打断」减半→爪平稳）、对臂负向（参考阶梯化"
                     "×指数追踪 v≈2.6×gap=冲刺-滑停速度方波=阶梯式抖动，"
                     "合步线销案）——本旋钮把爪的好处拆出来臂不陪绑。hold "
                     "状态机照常每拍跑（窗口墙钟不变），仅物理下发被节流，"
                     "被跳拍打「跳」标签。1=每拍（默认）。与 --cmd-every 的"
                     "区别：那个臂爪一起并组（臂会阶梯），这个只动爪。")
ap.add_argument("--grip-hold-plateau-mm", type=float, default=0.8,
                help="平台判据 mm：连续 window 拍实宽 max−min ≤ 此值才算物理"
                     "被撑住（正在合爪途中不触发）。默认 0.8")
ap.add_argument("--grip-hold-window", type=int, default=12,
                help="憋死判定窗（发送拍数，30Hz 下 12 拍=0.4s）：须实宽平台"
                     "且深指令持续满窗（>爪 onset 0.14s+行程起步，150603 死锁"
                     "教训）。默认 12")
ap.add_argument("--grip-effort-hold", type=float, default=4.0,
                help="保持段 effort（N·m 级力矩上限）：官方文档明文爪关节 "
                     "effort 字段生效=最大夹持力限制（galbot_robot.hpp "
                     "set_gripper_command 语义）。0=沿用常规 10。默认 4.0")
ap.add_argument("--config", default=s1_config.DEFAULT_PATH,
                help="配置文件路径（优先级 CLI > config > 内置默认）")
ap.add_argument("--log-file", default=None,
                help="日志文件路径（默认 <仓库>/logs/loop_native_<时间戳>.log；终端照常显示，"
                     "全文同步写文件，判读免复制）")
ap.add_argument("--grip-state-cmd", action="store_true", default=None,
                help="state 夹爪维(dim7/15)改喂【指令值】而非 SDK 回读：数据集 state "
                     "爪维语义=跟随段记指令值（持物段=物理被撑开），喂物理回读=分布"
                     "漂移（「回读滞后 6.3-8s」旧说已被 10-13 标定翻案：新鲜 p50 "
                     "10ms；行为必要性 10-05/10-08 实证保留）。pick 闭环必带；"
                     "需 --grip；默认关=喂回读")
ap.add_argument("--envelope-margin", type=float, default=0.15,
                help="关节包络护栏余量，rad（默认 0.15）。"
                     "包络=<ckpt>/joint_envelope.json"
                     "（任务数据集臂关节逐维 [min,max]）：指令越界=模型在幻想训练分布"
                     "里不存在的构型——往桌下伸/顶桌即此类（刚性模式压桌→fault）")
ap.add_argument("--envelope-abort-n", type=int, default=5,
                help="连续 N 块指令被包络截断即停循环（默认 5）："
                     "持续越界=感知漂移，继续跑只会反复撞桌")
ap.add_argument("--no-envelope-guard", action="store_true",
                help="关闭关节包络护栏（排障对照用）")
ap.add_argument("--nav-suspend", action="store_true", default=None,
                help="【S1 no-op】S1 无导航栈可搁置；旗标与档键保留只为与 "
                     "G1 traj [lerobot] 档同形（config 键 nav_suspend），"
                     "生效时打印一行说明。默认关")
# [lerobot] 布尔档反关旗标（三态约定「布尔想显式关给 --no-X」，见 s1_config 头注；
# 档里 true 时 CLI 无从显式关——entry-match 路线/醒着导航 A/B/place 相都要用）
ap.add_argument("--no-skip-install", action="store_true",
                help="显式关跳帧安装（[lerobot] 档 skip_install=true 的反关；"
                     "entry-match 路线用）")
ap.add_argument("--no-grip-state-cmd", action="store_true",
                help="显式关 state 爪维喂指令值（[lerobot] 档 grip_state_cmd=true "
                     "的反关；place 相等回读语义任务用）")
ap.add_argument("--no-nav-suspend", action="store_true",
                help="【S1 no-op】显式关导航栈搁置（G1 [lerobot] 档反关旗保留；"
                     "S1 本就无导航栈）")
ap.add_argument("--no-head-crop", action="store_true",
                help="显式关头图裁剪（[lerobot] 档 head_crop=true 的反关；"
                     "place 相/非裁剪训练 ckpt 用）")
args = ap.parse_args()
# ── 全文日志落盘：q 退出走 os._exit 不刷缓冲（09-23 坑），故逐行强制 flush ──
_log_path = args.log_file or str(
    pathlib.Path(__file__).resolve().parents[3] / "logs" /
    time.strftime("loop_traj_%Y%m%d_%H%M%S.log"))
pathlib.Path(_log_path).parent.mkdir(parents=True, exist_ok=True)
# chunk 落盘目录（--dump-chunks）：与日志同时间戳，包络截断前的纯模型输出
_dump_dir = None
if args.dump_chunks:
    _lp = pathlib.Path(_log_path)
    _dump_dir = _lp.with_name(_lp.stem.replace("loop_traj", "chunks_traj"))
    _dump_dir.mkdir(parents=True, exist_ok=True)
# 帧落盘目录（--dump-frames）：同上约定；存的是 letterbox224 之后的图=模型看到的
_frames_dir = None
if args.dump_frames:
    _lp = pathlib.Path(_log_path)
    _frames_dir = pathlib.Path(
        args.dump_frames_dir
        or _lp.with_name(_lp.stem.replace("loop_traj", "frames_traj")))
    _frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[loop-traj] 逐推理帧落盘 → {_frames_dir}")


# ── 取图/块落盘件已收编 common/loop/traj_engine.py（步2b-2b，原文逐位一致；
#    _frame_seq 计数器随迁引擎模块；_frames_dir/_dump_dir 目录仍在上方 init，
#    经下方引擎回填块下发）──
from common.loop.runtime import _Tee   # common 收编步2a（原文=旧内联版逐位一致）

_log_fh = open(_log_path, "a", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)
print(f"[log] 全文日志 → {_log_path}")
# ── config/s1.toml（扁平顶层键=CLI dest 名，s1_config 同读这份档）：
#    三态 CLI > toml > s1_config.BUILTIN；先于交叉校验落位=校验看生效值 ──
# 布尔档冲突检测须在 apply 之前：apply 会把档值 true 回填进 args，
# 之后无法区分「CLI 显式给」与「config 回填」
for _b in ("skip_install", "grip_state_cmd", "nav_suspend", "head_crop"):
    if getattr(args, "no_" + _b) and getattr(args, _b):
        raise SystemExit(f"--{_b.replace('_', '-')} 与 "
                         f"--no-{_b.replace('_', '-')} 同传，意图冲突")
s1_config.apply(args, {
    "ckpt": "ckpt",
    "prompt": "prompt",
    "policy": "policy",
    "rounds": "rounds",
    "fps": "ctrl_hz",
    "n_action_steps": "execute_actions_per_chunk",
    "skip_install": "skip_frames",
    "aggregate_fn": "aggregate_fn",
    "cmd_every": "cmd_every",
    "ema_alpha": "filter_alpha",
    "grip_state_cmd": "grip_state_cmd",
    "delta_max": "max_joint_delta_per_step",
    "burst_reserve_ms": "burst_reserve_ms",
    "nav_suspend": "nav_suspend",
    "rtc_horizon": "execution_horizon",
    "head_crop": "head_crop",
    "frames_per_row": "frames_per_row",
    "rows_per_cmd": "rows_per_cmd",
    "cache_frames": "cache_frames",
    "max_excursion": "max_excursion",
    "grip": "grip_enabled",
    "grip_path": "grip_path",
    "grip_speed": "grip_velocity",
    "grip_effort": "grip_effort",
    "grip_chg": "grip_chg",
}, path=args.config)
if args.fps is None:
    args.fps = 30
if args.frames_per_row is None:
    args.frames_per_row = 1
if args.rows_per_cmd is None:
    args.rows_per_cmd = 1

# ── policy 族门禁（10-29 适配器入口同口径）：toml/BUILTIN 的 ckpt 属 pi05 族
#    （010000），policy≠pi05 不吃；CLI 显式给仍赢。prompt 两族训练 task 同为
#    "test"（同源数据集），toml 值恰好通用——只补 None 回落，不排他 ──
if args.policy is None:
    args.policy = "pi05"
if args.policy != "pi05":
    if not any(a == "--ckpt" or a.startswith("--ckpt=") for a in sys.argv):
        from policies.fineart_vla_s1 import CKPT_DIR as _FA_CKPT
        args.ckpt = _FA_CKPT
        print(f"[policy] {args.policy}：ckpt 未显式给 → 族默认"
              f"（…{pathlib.Path(_FA_CKPT).parent.name}）")
    if args.prompt is None:
        args.prompt = "test"

# None 哨兵回落（配方旗标 argparse 默认改 None 交给三态回填，此处补脚本旧默认）
# 换块融合：注册表名→α（上游 AGGREGATE_FUNCTIONS 逐名对应）；--fusion-alpha
# 显式给=连续覆盖赢过注册表名（打印可观测）
_AGG_FN_ALPHA = {"weighted_average": 0.7, "latest_only": 1.0,
                 "average": 0.5, "conservative": 0.3}
if args.seam_overlap_alpha is None:
    if args.aggregate_fn is not None:
        args.seam_overlap_alpha = _AGG_FN_ALPHA[args.aggregate_fn]
    else:
        args.seam_overlap_alpha = 0.0
elif args.aggregate_fn is not None:
    print(f"[config] --fusion-alpha {args.seam_overlap_alpha:g} 显式覆盖 "
          f"--aggregate-fn {args.aggregate_fn}"
          f"（注册表 α={_AGG_FN_ALPHA[args.aggregate_fn]:g}）")
if args.cmd_every is None:
    args.cmd_every = 1
if args.ema_alpha is None:
    args.ema_alpha = 0.0
if args.rtc_horizon is None:
    args.rtc_horizon = 0
if args.burst_reserve_ms is None:
    args.burst_reserve_ms = 6.0
# 布尔档反关覆盖：--no-X 压过档值 true（显式关）；X/--no-X 全未给=False
for _b in ("skip_install", "grip_state_cmd", "nav_suspend", "head_crop"):
    if getattr(args, "no_" + _b):
        if getattr(args, _b):
            print(f"[config] --no-{_b.replace('_', '-')} 压过档值（生效 False）")
        setattr(args, _b, False)
    elif getattr(args, _b) is None:
        setattr(args, _b, False)
if args.rtc_horizon > 0 and "rtc" not in pathlib.Path(str(args.ckpt)).name.lower():
    print("⚠ rtc_horizon>0 但 ckpt 名不含 rtc：前缀引导只对条件训练权重合法"
          "（未训练 ckpt 开引导=拉弯 chunk/伸歪，10-16 定罪案）；确认这是有意的")


if args.ref_offset and not args.skip_install:
    raise SystemExit("--ref-offset 须与 --skip-frames 同用（偏移定义在新计划行 0 上）")
if args.entry_fill_dip and not args.entry_match:
    raise SystemExit("--entry-fill-dip 需要 --entry-match（填补在入口匹配流内执行）")
if args.entry_match:
    if args.skip_install or args.ref_offset:
        raise SystemExit("--entry-match 与 --skip-frames/--ref-offset 互斥"
                         "（入口匹配按位姿选行，取代时钟跳帧）")
    if args.blend:
        raise SystemExit("--entry-match 与 --blend 互斥（同回放台口径）")
    if args.ema_alpha <= 0:
        raise SystemExit("--entry-match 须配 --filter-alpha>0：方向/导程判据作用在"
                         "滤波后的行序列上，裸行序列的模型抖动会打死方向门"
                         "（回放台 v1 教训）")
if 0 < args.seam_overlap_alpha < 1:
    # 真混接（0<α<1）才需要独占缝口机制；latest_only（α=1.0=新块全量，
    # 旧流权重 0）惯性项恒等、运行时 _ovp 另有 merge<=1 旁路——d21f155
    # 同类修正：勿把「混接关」当「混接开」拒掉（3057676 放行 ref-offset
    # 后在此首次踩中）。
    _ovp_bad = [n for n, _on in (
        ("--entry-match", args.entry_match),
        ("--seam-alpha", args.seam_alpha > 0),
        ("--blend", args.blend),
        ("--ref-offset", args.ref_offset)) if _on]
    if _ovp_bad:
        raise SystemExit(f"--aggregate-fn/--fusion-alpha 与 {_ovp_bad} 互斥：重叠混接以"
                         "「旧流惯性项」取代全部缝口机制（同事栈语义），叠加会"
                         "双重滤波（latest_only=α1.0 不受此限）")
    if not args.skip_install:
        raise SystemExit("--fusion-alpha 须配 --skip-frames：混接对齐键="
                         "「取图→落地流逝拍数」，即跳帧安装的偏移；无对齐则新旧"
                         "计划错位混接（时间维 OOD）")
    if args.ema_alpha > 0:
        print("⚠ --fusion-alpha 与 --filter-alpha 同开：EMA 再滤一遍混接行，"
              "收敛新计划更慢（同事栈臂维无 EMA）；A/B 判读时注意此差异")
if args.seam_boost_alpha > 0:
    if args.seam_alpha > 0:
        raise SystemExit("--seam-boost-alpha 与 --seam-alpha 互斥：同为块首 α "
                         "调度但方向相反（boost=大 α 起步落回主 α 补速度，"
                         "seam-ramp=小 α 起步爬回主 α 削脉冲），叠加无定义")
    if args.ema_alpha <= 0:
        raise SystemExit("--seam-boost-alpha 须配 --filter-alpha>0：助推作用在"
                         "EMA 调度上，无滤波则无对象")
    if args.seam_boost_alpha <= args.ema_alpha:
        raise SystemExit(f"--seam-boost-alpha {args.seam_boost_alpha:g} 须大于 "
                         f"--filter-alpha {args.ema_alpha:g}（助推=块首比稳态更"
                         "快地采纳新计划；≤主 α 无意义）")
    if 0 < args.seam_overlap_alpha < 1:
        raise SystemExit("--seam-boost-alpha 与重叠混接（weighted_average/average）"
                         "互斥：同管块首 EMA 调度；latest_only（α=1.0=混接关）"
                         "不受限")
if args.fps > 100:
    raise SystemExit(f"--fps {args.fps} > 100 拒绝：tick {1000 / args.fps:.1f}ms 已进"
                     "SDK 单调用时延量级，非实验范畴")
if args.fps > 40:
    # 2026-10-26 用户申请 50Hz 超频实验，40+ 从硬拒改放行+大字警告：
    # tick <25ms 下超预算拍=丢拍重锚（绝对截止配速不追发，滑点自查）。
    # 50Hz 工程建议：每拍 1 帧（per/merge≈2，如 per2/m2=50Hz 下发）——
    # 2 帧/拍在 reserve 0 下帧距 ~10ms，读关节+爪+打印挤不进大概率串拍。
    print(f"⚠ --fps {args.fps} 超频放行（>40 安全区）：tick {1000 / args.fps:.1f}ms "
          "低于取图+SDK 硬开销 ~25ms，超预算拍会丢拍重锚；建议每拍 1 帧"
          "（per/merge≈2）并盯块时长是否 >1/fps×拍数（串拍签名）")
if args.frames_per_row < 1:
    raise SystemExit("--frames-per-row 须 ≥ 1")
if args.rows_per_cmd < 1:
    raise SystemExit("--rows-per-cmd 须 ≥ 1（1=每拍一包零变）")
if args.rtc_horizon < 0:
    raise SystemExit("--execution-horizon 须 ≥ 0（0=关）")
if args.cmd_every < 1:
    raise SystemExit("--cmd-every 须 ≥ 1（1=纯 30Hz 直发零变）")
if args.cmd_every > 1:
    raise SystemExit(
        "[S1] --cmd-every>1（含 --cmd-first-only/--cmd-every-ramp）已拦截"
        "（10-29 审查④）：合步=组末行目标一次下发跨 N 行、窗口仍 33.3ms——"
        "G1 靠伺服指数追踪大导程吞掉 N 行阶跃，S1 singorix WBC 直接看到计划"
        "断点（断点→LP 削顶→力矩跳变=卡卡，③定案同因）。跨行平滑在 S1 走 "
        "--rows-per-cmd R：攒 R 拍帧一包、tfs 铺满 R×tick 窗，无阶跃无丢行")
if args.cmd_first_only and args.cmd_every <= 1:
    raise SystemExit("--cmd-first-only 须配 --cmd-every > 1（否则无合步组可言）")
args.cmd_every_ramp = ([int(x) for x in args.cmd_every_ramp.split(",")
                        if x.strip()] if args.cmd_every_ramp else [])
if args.cmd_every_ramp:
    if args.cmd_every <= 1:
        raise SystemExit("--cmd-every-ramp 须配 --cmd-every > 1（无合步组则无组宽可言）")
    if any(g < 1 for g in args.cmd_every_ramp):
        raise SystemExit("--cmd-every-ramp 组宽须 ≥1")
    if len(args.cmd_every_ramp) > 16:
        raise SystemExit("--cmd-every-ramp 序列过长（≤16 组）")
    if args.cmd_first_only:
        raise SystemExit("--cmd-every-ramp 与 --cmd-first-only 互斥（缝侧细笔取代"
                         "「首组合步+组后流式」口径）")
if args.cmd_every > 1 and args.blend:
    raise SystemExit("--cmd-every > 1 与 --blend 互斥：blend 把块首重铺细步"
                     "（消块首尖峰），合步把 N 拍并 1 条大步（消指令沿密度）"
                     "——块界两套相反策略，勿混用")
if args.merge > 1:
    if args.cmd_every > 1 or args.cmd_first_only or args.cmd_every_ramp:
        raise SystemExit("--merge 与 --cmd-every/--cmd-first-only/--cmd-every-"
                         "ramp 互斥（同为合步语义，叠加无定义）")
    if (args.entry_match or args.blend
            or 0 < args.seam_overlap_alpha < 1):
        raise SystemExit("--merge 与换块重构族（--entry-match/--blend/"
                         "重叠混接 weighted_average|average）互斥："
                         "它们重建 rows/frames，不认识宏步分帧结构。"
                         "关混接请 --aggregate-fn latest_only（α=1.0=新块全量，"
                         "勿用 --fusion-alpha 0=全旧块卡死）；合并模式下 _ovp "
                         "重建帧流已旁路。--ref-offset 已放行：offset 在 "
                         "build_rows 内、宏步粗化上游生效，结构兼容"
                         "（5697eef 缝口回走遥测定案后解锁）")
    if args.merge > args.chunk_size:
        raise SystemExit(f"--merge {args.merge} > chunk 行数 {args.chunk_size}")
else:
    if args.grip_rawrows:
        print("[config] --grip-rawrows 在 merge=1 下无差别（爪本来就走行率流）")
if args.continuous_ref and args.entry_match:
    raise SystemExit("--continuous-ref 与 --entry-match 互斥"
                     "（同管换块链锚：入口匹配锚臂位 vs 链从 last_tgt "
                     "接续，二选一）。--ref-offset 已放行：其锚同为 last_tgt"
                     "（_prev0 由 skip-install/continuous-ref 提供），"
                     "平移只加在链钳上游，二者叠加不冲突")
TICK = 1.0 / args.fps
GAP_EPS = 1e-4        # 行导程低于该值免突发（伺服保持）


if args.no_grip:
    args.grip = False
if args.grip_ema_alpha is None:
    # 10-21 定档：同事 action_filter=dim23 全维单滤镜（robot_controller.py:110）
    # ——爪跟随臂 α，一个旋钮；显式传 --grip-ema-alpha 才单独覆盖（A/B 用）
    args.grip_ema_alpha = args.ema_alpha
GRIP_SS = None      # --grip-ss 解析结果（开阈, 合阈）；None=关（旧 chg 门行为）。
                    # ⚠ 初值必须在解析前且此后勿再赋 None：模块后段再置 None 会
                    # 盖掉解析值（125953 教训：banner 先打印、循环时已被抹）
if args.grip_ss:
    try:
        _o, _c = args.grip_ss.split(",")
        GRIP_SS = (float(_o), float(_c))
    except ValueError:
        raise SystemExit('--grip-ss 格式：「开阈,合阈」，如 --grip-ss 50,35')
    if not (0.0 <= GRIP_SS[1] < GRIP_SS[0] <= 100.0):
        raise SystemExit("--grip-ss 需 0 ≤ 合阈 < 开阈 ≤ 100（带内=迟滞防抖）")
if args.n_action_steps < 0:
    raise SystemExit("--execute-actions-per-chunk 须 ≥ 0（0=换块即预取）")
if args.n_action_steps == 0 and not (args.skip_install or args.entry_match):
    raise SystemExit("水位 0（换块即预取）须配 --skip-frames 或 --entry-match："
                     "推理落地时新块行 0 的墙钟槽位必已流逝 ~10 拍，不跳帧安装"
                     "=每块首拍系统性回走（110411 锯齿同源）；entry-match 按"
                     "位姿选行达到同一目的（不消费历史槽位）")
print(f"[loop-traj] tick {TICK * 1000:.1f} ms（{args.fps} Hz）| 轨迹流 "
      f"{args.frames_per_row} 帧/行（≈{args.fps * args.frames_per_row}Hz {args.interp}"
      f"）| 突发预算 {TICK * 1000 - args.burst_reserve_ms:.1f} ms/行"
      f" | EMA α={args.ema_alpha:g}"
      + (f" | 爪EMA α={args.grip_ema_alpha:g}"
         + ("（随臂）" if args.grip_ema_alpha == args.ema_alpha else "（单独）")
         if args.grip_ema_alpha > 0 else "")
      + (f" | 合步 1/{args.cmd_every}"
         + ("·仅首组，组后流式" if args.cmd_first_only else "")
         + (f"，缝侧细笔 {'/'.join(map(str, args.cmd_every_ramp))}→{args.cmd_every}"
            if args.cmd_every_ramp else "")
         if args.cmd_every > 1 else "")
      + (f" | 爪状态机 {GRIP_SS[0]:g}/{GRIP_SS[1]:g}" if GRIP_SS is not None else "")
      + (f" | 帧落盘 {_frames_dir.name}" if _frames_dir is not None else "")
      + f" | 水位 {args.n_action_steps} 步换块弃尾"
      + ("（换块即预取，块长≈推理时延）" if args.n_action_steps == 0 else "")
      + f" | 星饿上限 {args.starve_limit}s"
      + (" | 跳帧安装" if args.skip_install else "")
      + (f" | 入口动态匹配(锚=参考,跳深≤{args.entry_j_cap}"
         + (",无方向门" if args.entry_no_dir_gate else "")
         + (",凹陷填补" if args.entry_fill_dip else "")
         + (f",前向偏好ε{args.entry_fwd_bias:g}" if args.entry_fwd_bias > 0 else "")
         + ")" if args.entry_match else "")
      + (f" | 接缝α{args.seam_alpha}→{args.ema_alpha}/{args.seam_beats}拍"
         if args.seam_alpha > 0 and args.ema_alpha > 0 else "")
      + (f" | 接缝助推α{args.seam_boost_alpha}→{args.ema_alpha}/{args.seam_boost_beats}拍"
         if args.seam_boost_alpha > 0 and args.ema_alpha > 0 else "")
      + (f" | 重叠混接 旧{1 - args.seam_overlap_alpha:g}/新{args.seam_overlap_alpha:g}"
         if 0 < args.seam_overlap_alpha < 1 else "")
      + (" | 偏移平移" if args.ref_offset else "")
      + (f" | RTC 引导 horizon {args.rtc_horizon}（max_w {args.rtc_max_w}）"
         if args.rtc_horizon > 0 else " | RTC 关")
      + (" | 头图训练裁剪放大" if args.head_crop else "")
      + (f" | 帧到达时限 {args.tfs_ms:g} ms" if args.tfs_ms > 0 else ""))


# ── 导航栈搁置：S1 无导航栈，NavSuspend 整类不移植 ──
#（--nav-suspend 旗标/档键保留只为与 G1 traj 配置同形；生效 no-op）
NAV = None   # G1 版此处指向 NavSuspend 实例；S1 恒 None，各退出路径 None 守卫照旧


import common.loop.runtime as RT   # common 收编步2a：QuitWatcher/_fatal_hook/NAV 回填
from common.loop.runtime import QuitWatcher, _fatal_hook

WATCH = QuitWatcher()
RT.WATCH = WATCH
RT.NAV = NAV   # S1 恒 None（G1=NavSuspend 实例；runtime 各退出路径 None 守卫照旧）

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
print(f"[ckpt] {CKPT}")  # 日志自证权重（160917 盲区：日志无法判 A/B 跑的哪个 ckpt）

# ── 关节包络护栏（压桌防护，2026-10-06）：任务数据集臂关节合法构型盒。SDK 侧
# 臂端 fault/堵钻不回传（指令恒 SUCCESS，10-06 实录），刚性伺服撞桌只会硬压到
# 进 fault——唯一的软防线是在指令侧拦住"训练分布外的构型"。Motion.init 脱机
# 挂死（实测 >120s），FK 路线不可用，故用数据集包络做几何代理。
# 比对对象=BASE_ARM+delta 绝对构型（chunk_abs_block，2026-10-07 修范畴错误）。
ENV_LO = ENV_HI = None
ENV_LO14 = ENV_HI14 = None   # 无条件先置 None（帧级钳位只看 ENV_LO 门；2b-1 回填块
                             # 无条件引用这两个名字——护栏关档时旧码不定义也不读）
_env_file = CKPT / "joint_envelope.json"
if args.no_envelope_guard:
    print("[护栏] 关节包络护栏已 --no-envelope-guard 关闭")
elif _env_file.exists():
    _env = json.loads(_env_file.read_text())
    _inf = float("inf")
    ENV_LO = [(-_inf if v is None else float(v)) for v in _env["action_lo"]]
    ENV_HI = [(_inf if v is None else float(v)) for v in _env["action_hi"]]
    print(f"[护栏] 关节包络开：{_env_file.name}（来源 {_env.get('source', '?')}；"
          f"余量 ±{args.envelope_margin} rad；越界=截断 chunk，连续 {args.envelope_abort_n} "
          f"块→停）")
else:
    print(f"[护栏] 未找到 {_env_file}——关节包络护栏关闭（当前 ckpt 无任务包络）")

# ── 夹爪下发配置（--grip；宽度来自 manifest 标定，无标定拒绝开启）──
GRIP = None
if args.grip:
    g = mf.get("gripper", {})
    wmin, wmax = g.get("width_min"), g.get("width_max")
    if wmin is None or wmax is None:
        # S1 移植：ckpt 无 manifest 标定段 → 出厂行程缺省（与 GRIP_WMIN/MAX
        # 同源；标定后把实测写回 manifest 或这里）
        wmin, wmax = 0.007, 0.11
        print(f"[grip] ⚠ 无 manifest 爪标定，用出厂行程缺省 {wmin}/{wmax} m")
    GRIP = {"names": (("right_gripper", 7), ("left_gripper", 15)),
            "wmin": float(wmin), "wmax": float(wmax),
            "speed": args.grip_speed, "effort": args.grip_effort,
            "chg": args.grip_chg, "sent": {}, "sent_peak": {}}
    print(f"夹爪下发开启: 0%→{wmin} m | 100%→{wmax} m | 速度 {args.grip_speed} m/s | "
          f"力矩 {args.grip_effort} N | 变化阈值 {args.grip_chg}%")
    if args.grip_path == "joint":
        print("爪路径=关节流（--grip-path joint，10-21 起默认）：双爪作关节组并入 "
              "set_joint_commands（tfs 与臂同参、velocity=速度帽 0.15 被尊重、"
              "effort=10 同事同款）；chg/min-interval/send-every/ss/preclose "
              "各门全旁路（10-21 P6 三格实证：onset 0.12+D/v 律属 "
              "set_gripper_command 端点专属），EMA 照带")
        _gp_bad = [n for n, _v in (
            ("--grip-min-interval", args.grip_min_interval),
            ("--grip-send-every", args.grip_send_every),
            ("--grip-ss", args.grip_ss),
            ("--grip-preclose", args.grip_preclose)) if _v]
        if _gp_bad:
            raise SystemExit(f"--grip-path joint 与 {'/'.join(_gp_bad)} 互斥"
                             "（joint 路径无 onset/吞令定律，各门无对象；"
                             "10-21 P6 三格实证）")
    else:
        if args.grip_hold_on_stall:
            raise SystemExit("--grip-hold-on-stall 仅 --grip-path joint 支持"
                             "（憋死保持改写关节流目标；cmd 路是降级逃生门，"
                             "不加新行为）")
        print("爪路径=行程控制器（--grip-path cmd【逃生门/复现对照】）："
              "set_gripper_command 老路，各门适用——该路 onset=0.12s+行程/速度"
              "+0.03 下限+吞令=训练外注入行为（10-21 P6 破案），默认已不走")
    if args.grip_hold_on_stall:
        print(f"[grip-hold] 憋死保持开：指令≤实宽−{args.grip_hold_gap_mm}mm "
              f"持续满窗({args.grip_hold_window}拍)且实宽平台"
              f"(≤{args.grip_hold_plateau_mm}mm)"
              f"→钳 进入时实宽−{args.grip_hold_margin_mm}mm（冻结不跟随）"
              f"@vel{args.grip_hold_vel:g}/effort"
              f"{'沿用' if args.grip_effort_hold <= 0 else args.grip_effort_hold}"
              f"；指令≥实宽+{args.grip_hold_exit_mm}mm 退出恢复流式")
        if args.grip_hold_relax_mm > 0:
            print(f"[grip-hold] 自动外松开：保持中 eff≥{args.grip_hold_relax_eff:g} "
                  f"持续 6 拍→目标外松 {args.grip_hold_relax_mm:g}mm"
                  f"（上限=进入时实宽，零误差伺服停追；单向不回紧）")
        if args.grip_hold_quiet_n > 0:
            print(f"[grip-hold] 真静默制开：进 hold/外松步后发 {args.grip_hold_quiet_n} "
                  f"拍即停发该爪（静默期固件走完最后一条收工；读回评估照常，"
                  f"退出/外松自动重新武装）")
    if args.gripj_every > 1:
        print(f"[grip] 爪通道节流: 每 {args.gripj_every} 拍发 1 条"
              f"（{30.0 / args.gripj_every:.1f}Hz；臂照旧 30Hz 直发，"
              f"hold 状态机照常 30Hz 跑、窗口墙钟不变）")
    if args.grip_send_every > 0:
        print(f"爪定时静默制: 每 {args.grip_send_every} 机会拍 1 条"
              f"（≈{30.0 / args.grip_send_every:.1f}Hz、静默 "
              f"{100.0 - 100.0 / args.grip_send_every:.0f}%），chg 死区门旁路")
    if args.grip_min_interval > 0:
        if args.grip_send_every > 0 or GRIP_SS is not None:
            raise SystemExit("--grip-min-interval 与 --grip-send-every/--grip-ss "
                             "互斥（三选一的门：事件门/周期拍/状态机）")
        print(f"爪事件门: 距上次实发 ≥{args.grip_min_interval}s 才发；间隔未满仅"
              f"在途(|v|>0.02)且意图差≥3% 豁免（chg 门旁路）——静默保证起步、"
              f"在途更新免费（P2/P4：同向反向均平滑）")
    if args.grip_preclose > 0:
        print(f"预合爪爬行: 意图下行未平稳时下发改用 {args.grip_preclose} m/s"
              f"（死区吸收进意图早期，在途同向免费跟随；平稳后回 "
              f"{args.grip_speed} m/s 收敛；变速生效性待探针 P5）")
    if args.grip_state_cmd:
        print("⚠ state 夹爪维(7/15)喂【指令值】（--grip-state-cmd）：对齐数据集 state "
              "爪维语义（「回读滞后 6.3-8s」旧说已被 10-13 标定翻案，回读新鲜 "
              "p50 10ms；行为必要性 10-05/10-08 实证保留）")

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
    # 14 维臂包络（行/帧钳位用；16 维数据集布局挖掉夹爪两维）
    ENV_LO14 = np.concatenate([ENV_LO[:7], ENV_LO[8:15]])
    ENV_HI14 = np.concatenate([ENV_HI[:7], ENV_HI[8:15]])
import flash_rt.frontends.torch.pi05_rtx as _fe  # noqa: E402

# 回退 eager 时才封图（PI05_NO_GRAPH=1）；默认开图（scripts/g1/test/bench_pi05.py 同款）
if os.environ.get("PI05_NO_GRAPH", "0") == "1":
    _orig_init = _fe.Pi05TorchFrontendRtx.__init__

    @functools.wraps(_orig_init)
    def _no_graph_init(self, *a, **kw):
        kw["use_cuda_graph"] = False
        _orig_init(self, *a, **kw)

    _fe.Pi05TorchFrontendRtx.__init__ = _no_graph_init
    print("[loop-traj] CUDA graph 已禁用（PI05_NO_GRAPH=1），eager 模式")

import flash_rt  # noqa: E402
from flash_rt.core.utils.actions import normalize_state  # noqa: E402
from galbot_sdk.s1 import (  # noqa: E402
    GalbotRobot, JointCommand, RgbOutputFormat, SensorType, Trajectory,
    TrajectoryPoint)
from s1_traj_interp import create_interpolator  # noqa: E402  同目录（轨迹流插值）
# ── 机体专属件已收编 common/bodies/s1.py（步2b-2，原文逐位一致：关节表/
#    state 拼装/取图/爪下发/就位 warm；关节名常量改从机体模块取）──
import common.bodies.s1 as B
from common.bodies.s1 import (  # noqa: E402
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

# 当前 chunk 的预测时刻臂位——chunk 臂维是 delta（见 build_rows），换块时更新
BASE_ARM = None
_gcal = mf.get("gripper", {})
# S1 ckpt 无 manifest gripper 段 → 出厂行程缺省（policies/base.py 同源，pyi 同值；
# 未逐台标定——标定后把实测写回 manifest 或改这里）
GRIP_WMIN = _gcal.get("width_min", 0.007)
GRIP_WMAX = _gcal.get("width_max", 0.11)
if "width_min" not in _gcal or "width_max" not in _gcal:
    print(f"[grip] ⚠ 无 manifest 爪标定，用出厂行程缺省 {GRIP_WMIN}/{GRIP_WMAX} m"
          "（未逐台标定；偏了只会平移爪百分比零点，不影响臂）")
B.args = args
B.GRIP, B.GRIP_SS = GRIP, GRIP_SS   # dict 引用回填（send_grip* 原地改写共享）
B.GRIP_WMIN, B.GRIP_WMAX = GRIP_WMIN, GRIP_WMAX
B.CAM_MAP, B.VIEWS = CAM_MAP, VIEWS

# ── warm 就位/归位姿态：config/s1.toml [warm_pose] 覆盖（工作人员自配；
#    scripts/tools/warm_pose_from_dataset.py 按数据集重生成；缺省=烘档零改）
#    ——重绑 B.S1_HOME_POSE 模块属性（s1_warm 读自身模块 global，引擎经
#    TE 回填的函体同源），先于 TE.run()=引擎 gate-15 前 ──
_wp = s1_config.warm_pose(args.config)
if _wp:
    B.S1_HOME_POSE = {**B.S1_HOME_POSE, **_wp}
    print(f"[warm] 姿态覆盖 config/s1.toml [warm_pose]: {sorted(_wp)}"
          f"（未列组沿用 policies/base.py 烘档）")


# ── state 拼装/关节读取/相机冻结守卫已收编 common/bodies/s1.py（步2b-2）──


# ── 取图/冻结守卫已收编 common/bodies/s1.py（步2b-2，grab_views）──
# ── 遥测件已收编 common/loop/telemetry.py（步2a，原文逐位一致）──
from common.loop.telemetry import LatencyTracker, pstats  # noqa: E402


# ── 安全护栏纯件已收编 common/control/guards.py（步2b-1，原文逐位一致；
#    BASE_ARM/ENV_LO/ENV_HI 走模块属性回填，见下方 init 区回填块）──
import common.control.guards as GD
from common.control.guards import (  # noqa: E402
    chunk_abs_block,
    env_first_violation,
)


# ── 爪下发两路已收编 common/bodies/s1.py（步2b-2，原文逐位一致；
#    _GRIP_* 状态 dict/计数器以机体模块为唯一持有地）──


# ── ① SDK 初始化 + 模型 ──
print("== ① SDK 初始化（确认急停可及！）==")
robot = GalbotRobot()
RT.robot = robot   # _fatal_hook 的 SDK 优雅三连用（common/loop/runtime）
sensors = {getattr(SensorType, c) for c in list(CAM_MAP.values())[:VIEWS]}
if not robot.init(sensors):
    raise SystemExit("robot.init 失败")
time.sleep(5)
st = robot.start_controller("all")
print(f"start_controller('all') → {st}")
if not str(st).startswith("ControlStatus.SUCCESS"):
    raise SystemExit("控制器未 SUCCESS，终止（排查 SDK/急停/控制器状态）")

if args.nav_suspend:
    print("[nav-suspend] S1 无导航栈——旗标生效为 no-op（兼容档键保留）")


# ── 就位/归位 warm 已收编 common/bodies/s1.py（步2b-2，s1_warm）──
if args.do_exec:
    input("\n⚠ --exec 就位 warm：臂+头 @0.15 rad/s 慢速 → torso @0.2 → 双爪"
          "全开（目标 S1_HOME_POSE）。确认急停可及后回车开始…")
    if not s1_warm(robot, "就位"):
        raise SystemExit("warm 未 SUCCESS，终止（排查 SDK/急停/控制器状态）")
else:
    print("[warm] 干跑跳过就位（--exec 才动）")

HOME = read_joints(robot, ARM_NAMES)   # 起始位 = 漂移护栏基准（warm 后读）
print(f"起始位（漂移护栏基准）: {np.round(HOME, 3).tolist()}")

t0 = time.time()
if args.policy == "pi05":
    model = flash_rt.load_model(
        str(CKPT), config="pi05", num_views=VIEWS,
        cache_frames=args.cache_frames, action_dim=ACTION_DIM,
        rtc=args.rtc_horizon > 0,
        rtc_horizon=args.rtc_horizon or 10,
        rtc_max_weight=args.rtc_max_w)
    ns = model._pipe.norm_stats
else:  # fineart_vla_s1：config/stats 来源不同；predict 包装层通用（set_prompt
       # 带 state 特征探测 / calibrate 自动闩 / rtc_prefix 签名门控全族适用，
       # 11-02 traj 调用形探针 P1-P4 全绿；无 cache_frames——前端未声明该参）
    from policies.fineart_vla_s1 import state_stats_ns
    model = flash_rt.load_model(
        str(CKPT), config="fineart_vla", num_views=VIEWS, action_dim=ACTION_DIM,
        rtc=args.rtc_horizon > 0,
        rtc_horizon=args.rtc_horizon or 10,
        rtc_max_weight=args.rtc_max_w)
    ns = state_stats_ns(CKPT)   # preprocessor q01/q99 → ns dict（(27,) 硬闸）
print(f"tier=int8_full views={VIEWS} | load {time.time() - t0:.1f}s")

obs = grab_views(robot)
state0 = state_from_joints(read_joints(robot, STATE_NAMES))
state_n0 = normalize_state(state0, ns)
BASE_ARM = state0[ARM_SLICE]   # 首块 delta 基准 = 预测时刻臂位
chunk = np.asarray(model.predict(obs, prompt=args.prompt, state=state_n0))  # 建管线
# 首用引擎构建吸收：真实控制首拍曾撞 ~800ms 惰性构建（autotune 中途重现，
# 2026-09-22 tegrastats 已排除热/内存）。连做 3 次新鲜取图推理，把构建成本
# 烧在计时区外，避免首拍尖峰污染遥测
for _ in range(3):
    model.predict(grab_views(robot), prompt=args.prompt,
                  state=normalize_state(read_joints(robot, STATE_NAMES), ns))
print("预热推理 ×3 完成（吸收惰性引擎构建）")
WMARK = max(0, min(args.n_action_steps, args.chunk_size))   # 预取水位阈值（0=换块即预取）
if args.rtc_horizon > 0:
    print(f"[RTC] 前缀引导开：换块时旧块尾段重锚进新块坐标系喂引擎"
          f"（horizon {args.rtc_horizon}；前缀不可用时该块自动免引导）")


INTERP = create_interpolator(args.interp, dim=14,
                             input_hz=args.fps,
                             output_hz=args.fps * args.frames_per_row)
GRIP_INTERP = create_interpolator(args.interp, dim=2,
                                  input_hz=args.fps,
                                  output_hz=args.fps * args.frames_per_row)

_EMA_STATE = None   # 跨块连续 EMA 状态（上块末行滤波输出；None=尚无历史不滤波）
# ⚠ GRIP_SS 的初值 None 放 argparse 段（≈300 行）——若在此处再
# 置 None 会把上面解析好的 --grip-ss 盖掉（125953 实录：banner
# 打了 50/35 但循环里 GRIP_SS 已被本行旧赋值抹成 None，静默退
# 回 chg 门=状态机整跑未生效）
# （_GRIP_* 状态 dict/计数器已收编 common/bodies/s1.py，步2b-2）


# ── 换块融合纯件已收编 common/control/fusion.py（步2b-1，原文逐位一致；
#    args/插值器/包络盒/GRIP 一次回填，chunk/BASE_ARM/_EMA_STATE 活状态
#    全链只 FU 一份：入口每处重绑后回填同名属性，见下方回填块）──
import common.control.fusion as FU
from common.control.fusion import (  # noqa: E402
    build_block_stream,
    build_rows,
)

# ── common 模块回填（步2b-1）：此时 init 全就位（首块 chunk/BASE_ARM/插值器/
#    包络盒/爪标定）；其后入口重绑处逐一回填（首块落地/护栏截断/跳帧安装/
#    入口匹配/_EMA_STATE 复位与重播）──
FU.args = args
FU.chunk = chunk
FU.BASE_ARM = BASE_ARM
FU.INTERP = INTERP
FU.GRIP_INTERP = GRIP_INTERP
FU.GRIP = GRIP
FU.ENV_LO, FU.ENV_LO14, FU.ENV_HI14 = ENV_LO, ENV_LO14, ENV_HI14
FU._EMA_STATE = _EMA_STATE
GD.BASE_ARM = BASE_ARM
GD.ENV_LO, GD.ENV_HI = ENV_LO, ENV_HI




# ── ② 预取 worker/换块主循环/③执行确认/④汇总已收编 common/loop/traj_engine.py
#    （步2b-2b，原文逐位一致整体搬入 TE.run()；活状态 chunk/BASE_ARM/job/k/
#    counters 经 run() 的 global 名单保住引擎模块属性语义，见引擎文件头）──
# ── 引擎回填（一次，TE.run() 前；init 全就位）。TE.run() 不返回（内部
#    归位 + SDK 优雅三连 + os._exit）──
import common.loop.traj_engine as TE

TE.args = args
TE.robot = robot
TE.model = model
TE.ns = ns
TE.TICK = TICK
TE.WATCH = WATCH
TE.HOME = HOME
TE.WMARK = WMARK
TE.GAP_EPS = GAP_EPS
TE.GRIP = GRIP
TE.ENV_LO, TE.ENV_LO14, TE.ENV_HI14 = ENV_LO, ENV_LO14, ENV_HI14
TE.INTERP = INTERP
TE.NAV = NAV
TE._frames_dir = _frames_dir
TE._dump_dir = _dump_dir
TE.CKPT = CKPT
TE._env_file = _env_file
TE.run()
