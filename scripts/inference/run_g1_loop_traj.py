#!/usr/bin/env python
"""G1 双臂闭环——轨迹流执行器（set_joint_commands 高频流路线）。

与 run_g1_loop_native.py（母本）同一条推理链路、同一套护栏与水位换块调度，
只换发令侧：native 每拍发一条位置指令（30Hz 阶跃，10-09 真机七跑定案
抖感∝指令沿能量=阶跃幅值×沿频率）；本脚本走同事 infer_policy 的真机在役
路线——
  换块时一次性：行目标（链式 ±--delta-max rad/行 限幅）→ [当前实际位]+
  行目标 样条插值成每行 --frames-per-row 帧（默认 1=纯 30Hz 直发；10-10
  用户定案退役 240Hz 微流，N>1 仅 A/B 显式选用）→ 每 row-tick
  按 (tick−--burst-reserve-ms) 预算匀速突发下发 set_joint_commands
  （time_from_start=0，标准关节只消费 position）。N=1 时指令阶跃=整行差
  （链钳 ±delta-max 封顶）；N>1 时阶跃≈行差/N——消指令沿激励而非压速度。
SDK 契约出处：g1/__init__.pyi set_joint_commands "suitable for
  high-frequency control usage; For standard joints only the position
  field will be effective"；插值算法移植自同事 /home/galbot/infer_policy/
  joint_interpolator.py（linear/cubic/quintic → g1_traj_interp.py）。
调度约定出处（机制参照，非逐行移植；Apache-2.0）：
  lerobot async_inference/robot_client.py L477-489 control_loop 固定节拍
  （environment_dt 截止，sleep(max(0, dt−elapsed))）；L410-413
  _ready_to_send_observation 水位换块（qsize/chunk_size ≤ 0.5）。
时序铁律：绝对截止（next_t += TICK），落后超 1 拍即重锚——绝不连发追赶；
  星饿（块耗尽且推理未归）=不下发（刚性伺服保持最后目标原地等待），超
  --starve-limit 判挂死停机。帧突发配速预算=tick−--burst-reserve-ms，
  SDK 单帧调用若超预算只会体现为丢拍重锚（遥测可见），绝不追发。
RTC：--rtc-horizon N > 0 时启用推理侧前缀引导（同 native，FlashRT 引擎
  guided RTC）：旧块未消费尾段重锚进新块坐标系作为前缀，每个去噪步把
  新块开头拉向旧计划，消换块接缝跳变。0=关（默认）。
搭接：--blend 执行侧前向搭接（10-12 回放台定案移植）——换块后新块前 M 行
  目标替换为"旧链末目标→rows[M−1]"直连，消回溯俯冲/块首急停；与 RTC
  正交（推理侧 vs 执行侧），叠加属双重消缝，A/B 分开评估。0=关（默认）。
合步：--cmd-every N>1 每 N 拍并成 1 条指令（拍 k 下发组末行 rows[k+N−1]，
  中间行只消费不发送），指令沿 30→30/N Hz、阶跃≈N 行差——臂以指数追踪
  大导程平滑扫过，切块误差被导程淹没（A 线 spc 合步同机制；native 七跑
  定案抖感∝指令沿频率）。默认 1=纯 30Hz 直发零变。与 --blend 互斥
  （blend 把块首重铺细步，方向相反）。
  ⚠ 接口铁律（10-16 用户定案）：本脚本只走 set_joint_commands 纯流式，与
  lerobot/同事 infer_policy 执行侧保持同源——不得引入 set_joint_positions/
  配速（那是要配速时直接用 run_g1_loop 的理由）。代价已知：流式底层=指数
  追踪 v≈2.6×gap，合步大步每条带一个初速尖峰（20261016「一震一震」实录），
  属接口固有，压抖靠缩小 N 或 --cmd-first-only 减少合步条数。
  折中口径 --cmd-first-only：合步只并每块首组（rows 0..N−1 一条指令消块界，
  该拍组差本来就小 ~12-20 mrad），其余拍回 30Hz 逐拍流式密随——均匀合步的
  组间大步 85-105 mrad 太肥（20261016 真机实录：块界治好但误差搬进块内），
  且流式 Pursuit 会把首组滞后自然吃回。
参数命名对 lerobot 术语的对照：docs/lerobot-alignment.md。

⚠ 加 --exec 会连续真实驱动双臂！安全设计（与 native 同款）：
  1. 只动双臂 14 关节；腿/头维度永不下发；夹爪仅 --grip 显式开启时下发
     （0~100%% → manifest 标定宽度，变化超 --grip-chg 才发，非阻塞不等反馈）
  2. 换块时行目标链式限幅 ±--delta-max rad/行 + 逐帧包络钳位；漂移护栏：
     行目标偏离起始位超 --max-excursion → 立即停止循环
  3. q 键即时退出（后台监听线程，SDK 阻塞中也可退）；物理急停第一优先级
  4. 执行前回车确认

每块遥测：推理 ms / 取图 ms / tick 服务时长与抖动 / 星饿窗 /
接缝尖峰（换块首行 |Δcmd| vs 块内均值——RTC 收益主指标）/
汇总分位数（LatencyTracker，lerobot 同款 API）。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/inference/run_g1_loop_traj.py \
      [--ckpt ~/holy/models/pi05_g1_onlypick_deploy] [--exec] \
      [--rounds 60] [--rtc-horizon 10]
  # 240Hz 微流（--frames-per-row 8）已于 10-10 退役，默认 1=纯 30Hz 直发
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

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=None, help="部署目录（默认 config [run].ckpt）")
ap.add_argument("--prompt", default=None, help="任务指令（默认 config [run].prompt，须用训练原句）")
ap.add_argument("--exec", dest="do_exec", action="store_true",
                help="真实连续驱动双臂（默认干跑只打印首块 5 拍目标；不进配置，仅 CLI）")
ap.add_argument("--rounds", type=int, default=None,
                help="换块次数（每块消费到水位即换，弃尾；config [loop].rounds）")
ap.add_argument("--n-action-steps", type=int, default=None,
                help="预取水位阈值：本块消费到第 K 步即触发新推理（lerobot "
                     "chunk_size_threshold 语义；≤ --chunk-size；"
                     "config [loop].n_action_steps；v3 里它是消费配额，这里只定换块时机）。"
                     "0=换块即预取（本块第 0 步下发后立即起跑下一块推理，块长≈推理"
                     "时延；须配 --skip-install）")
ap.add_argument("--chunk-size", type=int, default=50,
                help="chunk 长度（lerobot chunk_size），默认 50=训练原生长度"
                     "（env FLASH_RT_PI05_ACTION_CHUNK_SIZE 自动设）")
ap.add_argument("--delta-max", type=float, default=None,
                help="每行目标限幅 rad（链式：每行至多向 chunk 行绝对位挪这么多；"
                     "config [loop].delta_max）")
ap.add_argument("--fps", type=int, default=30,
                help="行节拍频率 Hz（默认 30=训练控制率；>40 拒绝：tick 低于"
                     "取图+SDK 硬开销 ~25ms 必丢拍）")
ap.add_argument("--frames-per-row", type=int, default=1,
                help="每行插值帧数：行目标样条插成 N 帧于 row-tick 内按预算匀速"
                     "突发下发 set_joint_commands。默认 1=纯 30Hz 直发（每 tick"
                     " 一条命令，原生 lerobot 语义；10-10 用户定案退役 240Hz 微"
                     "流）；N>1 仅留作 A/B 显式选用（8≈240Hz@30fps）")
ap.add_argument("--cmd-every", type=int, default=1,
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
ap.add_argument("--entry-match", action="store_true",
                help="入口动态匹配换块（10-17 回放台 entry-match 移植，用户提案；"
                     "与 --skip-install/--ref-offset/--blend 互斥——本模式取代时钟"
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
ap.add_argument("--burst-reserve-ms", type=float, default=6.0,
                help="row-tick 内留给读关节/打印/换块的毫秒数：帧突发配速预算="
                     "tick−该值（默认 6）")
ap.add_argument("--interp", choices=("linear", "cubic", "quintic"), default="cubic",
                help="行间样条（g1_traj_interp，移植自同事 infer_policy 真机在役"
                     "实现；默认 cubic natural 样条 C2 连续无冲击）")
ap.add_argument("--ema-alpha", type=float, default=0.0,
                help="行目标 EMA 低通系数（0=关，默认；借自同事 action_filter "
                     "lowpass）：rows[k]←α·raw[k]+(1−α)·rows[k−1]，状态跨块连续"
                     "（接缝首行同样被滤，infer_policy 同款语义；10-12 判明块内"
                     "重启版对接缝无效）——接缝/深尾垃圾的执行侧兜底杠杆；首跑"
                     "先 0 判读样条本身")
ap.add_argument("--grip-ema-alpha", type=float, default=0.0,
                help="爪指令 EMA 低通系数（0=关，默认）：grip%% 流逐拍平滑后再进"
                     "发送判定。臂维 EMA（--ema-alpha）管不了爪：build_rows 只滤"
                     " [:7]+[8:15]，爪维 dim7/15 直通 send_grip。⚠单独用治不了"
                     "冻结（111901 终审：病根=每条新指令重启行程，细流仍是流）；"
                     "与 --grip-ss 配套做去抖穿越判定，候选 0.1")
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
ap.add_argument("--skip-install", action="store_true",
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
                help="偏移平移（配合 --skip-install，10-13 120045 实机后续杠杆）："
                     "新计划锚=观测滞后臂，行 0 天然落后参考一个失配量（=块首回走"
                     "刹车的来源）。把整条新计划平移 offset=last_tgt−plan[0]（逐维"
                     "钳 ±delta-max）再接参考链——块首零缝步零回走，臂速无块界刹车，"
                     "导程呼吸趋平。钳幅 ≤delta-max 不吃穿包络余量（±0.15）；超出"
                     "部分仍按 ±delta-max 回走收敛。EMA 可叠加（块内滤抖）。默认关")
ap.add_argument("--blend", action="store_true",
                help="换块前向搭接（10-12 回放台定案：无回摆模式最优，残差 75.5 "
                     "vs anchor 97/基线 69）：新块前 M 行不发原目标，改发旧链末"
                     "目标→rows[M−1] 直连（M 自适应=计划首次爬回旧链高度行号+1；"
                     "门：滞后 >60 mrad 才触发）——消回溯俯冲与块首急停，参考链"
                     "全程前向，臂不停车不回摆；1:1 替换不加拍不占墙钟，末行=原 "
                     "rows[M−1] 与 EMA 状态链一致。与 --rtc-horizon 叠加属双重"
                     "消缝（推理侧+执行侧），A/B 时分开评估")
ap.add_argument("--rtc-horizon", type=int, default=0,
                help="推理侧 RTC 前缀引导窗口（行数）：>0 启用（lerobot 默认 10），"
                     "0=关（默认）。前缀=旧块未消费尾段重锚进新块坐标系")
ap.add_argument("--rtc-max-w", type=float, default=10.0,
                help="RTC 最大引导权重（lerobot max_guidance_weight 默认 10）")
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
ap.add_argument("--config", default=g1_config.DEFAULT_PATH,
                help="配置文件路径（优先级 CLI > config > 内置默认）")
ap.add_argument("--log-file", default=None,
                help="日志文件路径（默认 <仓库>/logs/loop_native_<时间戳>.log；终端照常显示，"
                     "全文同步写文件，判读免复制）")
ap.add_argument("--grip-state-cmd", action="store_true",
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
ap.add_argument("--nav-suspend", action="store_true",
                help="闭环期间 SIGSTOP 搁置导航栈 7 进程（localization/fusion/"
                     "navigation_plan/vtn/surround/swallows/lidar_capture）："
                     "2026-10-07 实测推理 375.3→320.5ms（−54.8ms）、GR3D 50→7。"
                     "退出自动 CONT（含 kill -9，守护网兜底）。"
                     "⚠ 只用于臂上任务+底盘不动窗口：冻结期 localization 位姿停更，"
                     "底盘先恢复导航。默认关")
args = ap.parse_args()

if args.ref_offset and not args.skip_install:
    raise SystemExit("--ref-offset 须与 --skip-install 同用（偏移定义在新计划行 0 上）")
if args.entry_fill_dip and not args.entry_match:
    raise SystemExit("--entry-fill-dip 需要 --entry-match（填补在入口匹配流内执行）")
if args.entry_match:
    if args.skip_install or args.ref_offset:
        raise SystemExit("--entry-match 与 --skip-install/--ref-offset 互斥"
                         "（入口匹配按位姿选行，取代时钟跳帧）")
    if args.blend:
        raise SystemExit("--entry-match 与 --blend 互斥（同回放台口径）")
    if args.ema_alpha <= 0:
        raise SystemExit("--entry-match 须配 --ema-alpha>0：方向/导程判据作用在"
                         "滤波后的行序列上，裸行序列的模型抖动会打死方向门"
                         "（回放台 v1 教训）")
if args.fps > 40:
    raise SystemExit(f"--fps {args.fps} > 40 拒绝：tick {1000 / args.fps:.1f}ms 低于"
                     "取图+SDK 硬开销 ~25ms，必然连续丢拍")
if args.frames_per_row < 1:
    raise SystemExit("--frames-per-row 须 ≥ 1")
if args.rtc_horizon < 0:
    raise SystemExit("--rtc-horizon 须 ≥ 0（0=关）")
if args.cmd_every < 1:
    raise SystemExit("--cmd-every 须 ≥ 1（1=纯 30Hz 直发零变）")
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
TICK = 1.0 / args.fps
GAP_EPS = 1e-4        # 行导程低于该值免突发（伺服保持）

# ── 全文日志落盘：q 退出走 os._exit 不刷缓冲（09-23 坑），故逐行强制 flush ──
_log_path = args.log_file or str(
    pathlib.Path(__file__).resolve().parents[2] / "logs" /
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
_frame_seq = 0


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
    "delta_max": ("loop", "delta_max"),
    "cache_frames": ("loop", "cache_frames"),
    "max_excursion": ("loop", "max_excursion"),
    "grip": ("gripper", "enabled"),
    "grip_speed": ("gripper", "speed"),
    "grip_effort": ("gripper", "effort"),
    "grip_chg": ("gripper", "chg"),
})
if args.no_grip:
    args.grip = False
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
    raise SystemExit("--n-action-steps 须 ≥ 0（0=换块即预取）")
if args.n_action_steps == 0 and not (args.skip_install or args.entry_match):
    raise SystemExit("水位 0（换块即预取）须配 --skip-install 或 --entry-match："
                     "推理落地时新块行 0 的墙钟槽位必已流逝 ~10 拍，不跳帧安装"
                     "=每块首拍系统性回走（110411 锯齿同源）；entry-match 按"
                     "位姿选行达到同一目的（不消费历史槽位）")
print(f"[loop-traj] tick {TICK * 1000:.1f} ms（{args.fps} Hz）| 轨迹流 "
      f"{args.frames_per_row} 帧/行（≈{args.fps * args.frames_per_row}Hz {args.interp}"
      f"）| 突发预算 {TICK * 1000 - args.burst_reserve_ms:.1f} ms/行"
      f" | EMA α={args.ema_alpha:g}"
      + (f" | 爪EMA α={args.grip_ema_alpha:g}" if args.grip_ema_alpha > 0 else "")
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
      + (" | 偏移平移" if args.ref_offset else "")
      + (f" | RTC 引导 horizon {args.rtc_horizon}（max_w {args.rtc_max_w}）"
         if args.rtc_horizon > 0 else " | RTC 关")
      + (f" | 帧到达时限 {args.tfs_ms:g} ms" if args.tfs_ms > 0 else ""))


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
            signal.signal(signal.SIGINT, lambda *_: None)     # 退出走管道，别靠
            signal.signal(signal.SIGTERM, lambda *_: None)    # 会被推迟的异常
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
                    if NAV is not None:
                        NAV.resume()
                    os._exit(130)
                if sys.stdin.read(1) in ("q", "Q"):
                    print("\n⛔ 按下 q —— 立即退出（已下发目标可能仍在限速执行）")
                    self.restore()
                    if NAV is not None:
                        NAV.resume()
                    os._exit(2)
            except Exception:
                self.restore()
                if NAV is not None:
                    NAV.resume()
                os._exit(3)   # 监听线程死了比静默更危险：宁可误退不可失控


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
# 比对对象=BASE_ARM+delta 绝对构型（chunk_abs_block，2026-10-07 修范畴错误）。
ENV_LO = ENV_HI = None
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
        raise SystemExit("--grip 需要 manifest 夹爪标定；先重跑 "
                         "g1_ckpt_prep.py --grip-wmin/--grip-wmax（2026-09-23 实测 "
                         "0.0005/0.1200 m）")
    GRIP = {"names": (("right_gripper", 7), ("left_gripper", 15)),
            "wmin": float(wmin), "wmax": float(wmax),
            "speed": args.grip_speed, "effort": args.grip_effort,
            "chg": args.grip_chg, "sent": {}, "sent_peak": {}}
    print(f"夹爪下发开启: 0%→{wmin} m | 100%→{wmax} m | 速度 {args.grip_speed} m/s | "
          f"力矩 {args.grip_effort} N | 变化阈值 {args.grip_chg}%")
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

# 回退 eager 时才封图（PI05_NO_GRAPH=1）；默认开图（scripts/test/bench_pi05.py 同款）
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
from galbot_sdk.g1 import GalbotRobot, SensorType, G1JointGroup, JointCommand  # noqa: E402
from g1_traj_interp import create_interpolator  # noqa: E402  同目录（轨迹流插值）

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
# 当前 chunk 的预测时刻臂位——chunk 臂维是 delta（见 build_rows），换块时更新
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

    数据集 state 爪维语义=跟随段记指令值——喂物理回读=分布漂移（模型全程
    看到滞后/爬坡中的爪宽）。⚠「SDK 回读滞后 6.3-8s」旧说已被 10-13 标定
    翻案（grip_speed_bench 四程：回读时戳新鲜 p50 10ms，伺服守 0.15 m/s）；
    本旗标的行为必要性（10-05/10-08 四跑实证）不受影响。

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

    2026-10-12 实锤（本机 pick_place_balence 抽帧判读）：三视角训练帧全部为
    letterbox——224×224、上下各 28px 纯黑条、内容带 168px=224×960/1280（源帧
    1280×960 等比缩放，非拉伸非裁剪）。此前 grab_views 直拉 resize 满幅无条
    = train/serve 图像分布失配（10-10 视觉-臂接地微弱的第一嫌疑，推翻当日
    "喂图净"结论——该验证未覆盖预处理变换维度）。
    """
    h, w = img.shape[:2]
    _s = 224.0 / max(h, w)
    _nh, _nw = round(h * _s), round(w * _s)
    _cv = np.zeros((224, 224, 3), dtype=np.uint8)
    _y, _x = (224 - _nh) // 2, (224 - _nw) // 2
    _cv[_y:_y + _nh, _x:_x + _nw] = cv2.resize(img, (_nw, _nh))
    return _cv


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
        out[key] = np.ascontiguousarray(
            cv2.cvtColor(letterbox224(img), cv2.COLOR_BGR2RGB))
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
    BASE_ARM）、夹爪维=绝对 0-100。
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


def send_grip(robot, chunk_row):
    """chunk 行 dim7/dim15（0~100%）→ 标定宽度下发（超阈值才发，非阻塞）。

    只发不查不等待（拖死控制环）；终态核对放循环结束后。回读新鲜（10-13
    标定 p50 10ms），实际下发拍顺带采样一次物理实宽进日志（[实 X.Xmm]），
    爪的物理时序证据逐 send 可见。已知物理动态（同标定）：起步死区 ~0.9s +
    0.15 m/s 帽（满程 ~0.8s）。
    ⚠ 111901 终审：每条新指令把行程从头重启（死区重吃）——逐拍补发的指令流
    p50 间隙 2-3 拍，死区永不耗尽 → 110 send 全 SUCCESS 实宽全程 0.5mm 冻结；
    q 停流静→最后一条 1.5s 走满（「q 后突然张开」全案闭合）。故
    --grip-ss 状态机=少发（每相 1-2 条、每条 ≥2s 静默窗）；无 ss 时退回旧
    chg 门（2% 变化即发，111901 证明该模式爪会冻结——保留只为对照）。
    """
    if GRIP is None:
        return None
    if not args.do_exec:
        return "dry（不下发）"
    parts = []
    now = time.monotonic()
    for name, dim in GRIP["names"]:
        p = float(np.clip(chunk_row[dim], 0.0, 100.0))
        if args.grip_ema_alpha > 0:
            prev = _GRIP_EMA.get(name)
            p = p if prev is None else \
                args.grip_ema_alpha * p + (1.0 - args.grip_ema_alpha) * prev
            _GRIP_EMA[name] = p
        last = GRIP["sent"].get(name)
        why = ""
        if GRIP_SS is not None:
            st0 = _GRIP_SS_STATE.get(name)
            t0, v0 = _GRIP_SS_SENT.get(name, (None, None))
            if st0 is None:                      # 首拍定态并发一次
                st0 = "hi" if p >= GRIP_SS[0] else "lo"
                _GRIP_SS_STATE[name] = st0
                why = "首拍"
            elif p >= GRIP_SS[0] and st0 != "hi":
                _GRIP_SS_STATE[name] = "hi"
                why = "开阈"
            elif p <= GRIP_SS[1] and st0 != "lo":
                _GRIP_SS_STATE[name] = "lo"
                why = "合阈"
            elif (t0 is not None and abs(p - v0) >= 15.0
                  and now - t0 >= 2.0):
                why = "漂移补发"                 # 中间宽度意图，限频 0.5Hz
            if not why:                          # 无翻转未到补发条件=一律不发
                parts.append(f"{name[0].upper()} {p:.1f}%·hold")
                continue
        elif last is not None and abs(p - last) < GRIP["chg"]:
            parts.append(f"{name[0].upper()} {p:.1f}%·hold")
            continue
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        st = robot.set_gripper_command(getattr(G1JointGroup, name),
                                       w, GRIP["speed"], GRIP["effort"], False)
        GRIP["sent"][name] = p
        if GRIP_SS is not None:
            _GRIP_SS_SENT[name] = (now, p)
        _gs = robot.get_gripper_state(getattr(G1JointGroup, name))
        _phys = f" 实{_gs.width * 1000:.1f}mm" if _gs is not None else " 实?"
        parts.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm{_phys} "
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
model = flash_rt.load_model(
    str(CKPT), config="pi05", num_views=VIEWS,
    cache_frames=args.cache_frames, action_dim=ACTION_DIM,
    rtc=args.rtc_horizon > 0,
    rtc_horizon=args.rtc_horizon or 10,
    rtc_max_weight=args.rtc_max_w)
ns = model._pipe.norm_stats
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
env_hits = 0                         # 包络护栏：连续越界块数（干净块清零）
env_trunc = 0                        # 包络护栏：累计截断块数（汇总用）
if args.rtc_horizon > 0:
    print(f"[RTC] 前缀引导开：换块时旧块尾段重锚进新块坐标系喂引擎"
          f"（horizon {args.rtc_horizon}；前缀不可用时该块自动免引导）")


INTERP = create_interpolator(args.interp, dim=14,
                             input_hz=args.fps,
                             output_hz=args.fps * args.frames_per_row)

_EMA_STATE = None   # 跨块连续 EMA 状态（上块末行滤波输出；None=尚无历史不滤波）
_GRIP_EMA = {}      # 爪指令 EMA 状态（--grip-ema-alpha>0 时启用；跨块连续，爪流
                    # 换块不弃尾——爪相位是低频意图，逐拍直通=物理包络被抖动打断）
                    # ⚠ GRIP_SS 的初值 None 放 argparse 段（≈300 行）——若在此处再
                    # 置 None 会把上面解析好的 --grip-ss 盖掉（125953 实录：banner
                    # 打了 50/35 但循环里 GRIP_SS 已被本行旧赋值抹成 None，静默退
                    # 回 chg 门=状态机整跑未生效）
_GRIP_SS_STATE = {}  # 状态机每爪当前态 'hi'/'lo'
_GRIP_SS_SENT = {}   # 状态机每爪 (上次实发时刻, 上次实发值)


def build_rows(cur, prev0=None, offset=None, seam=False):
    """chunk delta 块 → 链式限幅行目标（14 维，右臂在前）。

    ⚠ 2026-09-23 语义修正（同母本 plan_step）：训练管线
    relative_actions_processor 把臂维动作转成 delta（相对本块预测时刻臂位
    BASE_ARM）——行目标不是 chunk 绝对值，须加回 BASE_ARM（换块时随 job 更新）。
    链式：rows[0]=cur 向行 0 绝对位限幅挪一步，rows[k]=rows[k−1] 再向行 k 挪——
    每行位移 ≤ --delta-max，且逐行夹在 cur 与 chunk 绝对位之间（包络盒凸性 →
    行目标自动在包络内；样条过冲由 build_block_stream 的帧级钳位兜底）。
    EMA：--ema-alpha>0 时状态跨块连续（infer_policy action_filter 同款语义，
    2026-10-12 对照判明块内重启版对接缝无效）：新块 rows[0] 先与上块末行滤波
    输出混合再起链，接缝首行同样被滤；首块/循环起点无历史即以 rows[0] 播种。
    滤波对象=链钳后的绝对行目标（与同事"钳位→滤波"同序），凸组合仍在包络内。
    seam=True：接缝窗 α 阶梯（10-17 回放台移植）——换块后前 --seam-beats 拍
    α 从 --seam-alpha 线性爬回主 α，削块首衰减尾峰值、稳态零税；仅跨块
    有历史时生效（首块无接缝）。缺省 False=原口径逐位不变。
    """
    global _EMA_STATE
    rows = np.empty((len(chunk), 14), dtype=np.float32)
    # prev0=链起点（skip-install 参考连续用）：缺省从 cur 出发（原口径）；
    # 传入 last_tgt 时链从上一拍下发目标接续——首行=last_tgt+clip(新首行−
    # last_tgt, ±delta_max)，参考不重锚（110411 判明：换块重锚=脉动残留源）。
    prev = (np.asarray(cur, dtype=np.float32) if prev0 is None
            else np.asarray(prev0, dtype=np.float32))
    for kk in range(len(chunk)):
        arm_tgt = BASE_ARM + np.concatenate([chunk[kk][:7], chunk[kk][8:15]])
        if offset is not None:
            arm_tgt = arm_tgt + offset      # --ref-offset：整条计划平移（块首零回走）
        prev = prev + np.clip(arm_tgt - prev, -args.delta_max, args.delta_max)
        rows[kk] = prev
    if args.ema_alpha > 0:
        a = float(args.ema_alpha)
        # 接缝窗 α 阶梯（seam=True 且有跨块历史时启用）：α 从 seam_alpha
        # 线性爬回主 α；硬切会把积压滞后一次放出成新凸点（回放台离线实证）
        _ramp = seam and args.seam_alpha > 0 and _EMA_STATE is not None
        a_seam = float(args.seam_alpha)
        n_sb = max(1, int(args.seam_beats))
        acc = None
        for kk in range(len(rows)):
            if _ramp:
                a_k = a_seam + (a - a_seam) * min(kk, n_sb) / n_sb
            else:
                a_k = a
            if kk == 0:
                acc = rows[0] if _EMA_STATE is None else \
                    a_k * rows[0] + (1.0 - a_k) * _EMA_STATE
            else:
                acc = a_k * rows[kk] + (1.0 - a_k) * acc
            rows[kk] = acc
        _EMA_STATE = acc.copy()
    return rows


def build_block_stream(cur, prev0=None, offset=None, seam=False):
    """换块时一次性：行目标 → [链起点]+行目标 样条插值 → 逐行分桶帧。

    返回 (rows_abs, frames_all, bursts)：frames_all (M,14) float32；
    bursts[k]=第 k 拍下发的帧号列表（恰好 --frames-per-row 帧/行）。
    帧级包络钳位同 infer_policy robot_controller._clip_to_joint_limits
    （样条可能过冲）。帧时间网格 = 行节奏 1/args.fps → 帧率
    args.fps×frames_per_row：帧按 wall-clock 匀速铺满整行（--fps<30 时整块
    放慢播放，节奏语义同 native 的慢速彩排）。prev0 透传 build_rows
    （skip-install 参考连续：链起点=last_tgt，首帧恰=last_tgt=零缝步）。
    """
    rows = build_rows(cur, prev0, offset, seam=seam)
    _start = (np.asarray(cur, dtype=np.float32) if prev0 is None
              else np.asarray(prev0, dtype=np.float32))
    waypoints = np.vstack([_start[None, :], rows])
    frames_all = INTERP.interpolate(waypoints)
    if ENV_LO is not None:
        np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :], out=frames_all)
    per = args.frames_per_row
    bursts = [list(range(kk * per, (kk + 1) * per)) for kk in range(len(rows))]
    return rows, frames_all, bursts


def _rtc_prefix(new_base):
    """旧块未消费尾段 → 新块 delta 坐标系（RTC 前缀，推理侧引导输入）。

    臂维重锚：绝对构型 (OLD_BASE+delta) − NEW_BASE（新块自己的 delta 基准）
    夹爪维（7/15）数据集语义即绝对 0-100，原值携带。
    尾段不足 horizon 时按 lerobot rollout 语义 hold-pad（重复末行——零填充
    会解码到数据集均值）；块耗尽/RTC 关时返回 None（引擎侧 enable=0 短路）。
    须在 job.start 前调用（chunk/BASE_ARM 还属旧块，new_base=本次取关节读数，
    与 job 即将存下的 state_arm 同源）。

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

    只有 predict 放后台；SDK 取图/读关节/下发全部留在主线程，不碰 SDK
    线程安全。同一时刻只有一个 predict 在飞（到水位才启动下一个）。
    """

    def __init__(self, model, prompt):
        self._model, self._prompt = model, prompt
        self._done = threading.Event()
        self._res = None
        self._err = None
        self.dur_ms = 0.0

    def start(self, obs, state_n, state_arm, rtc_prefix=None):
        self._done.clear()
        self._err = None
        self.state_arm = state_arm   # 本块 delta 基准（预测时刻臂位）

        def _run():
            t = time.perf_counter()
            try:
                self._res = np.asarray(self._model.predict(
                    obs, prompt=self._prompt, state=state_n,
                    rtc_prefix=rtc_prefix))
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


# ── ② 干跑：只打印首块前 5 拍计划，绝不下发 ──
infer_ms = LatencyTracker()
grab_ms = LatencyTracker()
job = _PredictJob(model, args.prompt)
job.start(*fresh_obs())              # 首块
chunk = job.result()
_dur0 = job.dur_ms
infer_ms.add(_dur0)
BASE_ARM = job.state_arm
job = None
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
    return False


if not args.do_exec:
    print(f"\n== [干跑] 首块轨迹流预览（tick {TICK * 1000:.1f} ms，"
          f"{args.frames_per_row} 帧/行，链式限幅 ±{args.delta_max} rad）"
          "未下发任何命令 ==")
    _cur0 = read_joints(robot, ARM_NAMES)
    _rows, _frames, _bursts = build_block_stream(_cur0)
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
          + (f"指令沿 合步 1/{args.cmd_every}={args.fps / args.cmd_every:.1f}Hz"
             f"（每 {args.cmd_every} 拍 1 条）" if args.cmd_every > 1 else
             f"指令沿 {args.fps:g}Hz 逐拍 1 条")
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
_EMA_STATE = None                    # 循环起点重置：干跑预览那次 build 不作块 1 的跨块历史
t_loop0 = time.perf_counter()        # 任务秒表
t_block0 = t_loop0                   # 块秒表（换块节拍）
seam_first = None                    # 换块后首拍 |Δcmd|（接缝尖峰，主指标）
seam_rest = []                       # 换块后其余拍 |Δcmd|（块内均值基准）
seam_first_all = []                  # 跨块累计：每块首拍 |Δcmd|
seam_intra_all = []                  # 跨块累计：每块块内均值 |Δcmd|


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
next_t = time.perf_counter()
swap_envelope_check()
_cur0 = read_joints(robot, ARM_NAMES)
rows_abs, frames_all, bursts = build_block_stream(_cur0)
print(f"\n── 块 1 | t+0.0s | 推理 {infer_ms.percentile(1.0) if len(infer_ms) else float('nan'):.0f} ms ──")
print(f"[轨迹流] 行 {len(rows_abs)} → 帧 {len(frames_all)}（{args.interp} "
      f"{args.frames_per_row} 帧/行，≈{args.fps * args.frames_per_row}Hz）")

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
        chunk = job.result()
        _dur = job.dur_ms
        infer_ms.add(_dur)
        BASE_ARM = job.state_arm
        job = None
        n_swapped += 1
        _dump_chunk(chunk, BASE_ARM, n_swapped, _dur)   # n_swapped 初值=1（首块计入）→ 块2 起 idx 连号不覆盖
        prev_block_ticks = k          # 上块消费拍数（重置前捕获，表头用）
        k = 0
        _grp_k0 = _grp_idx = 0        # ramp 组游标随块复位
        seam_flush()
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
            _EMA_STATE = last_tgt.copy()
        # ── 参考链连续（skip-install 配套，10-13 真机 110411 判明）：链式限幅
        # 从当前臂位重建链会把参考钳回 cur+delta_max（拍0 导程 50），换块锯齿
        # 50→330→50 = 脉动的残留源（跳帧安装只对了时间槽，没动链锚）。
        # 跳帧安装后新块行 0 本就 ≈ 墙钟正确位（≈last_tgt±模型修正），链改从
        # last_tgt 接续：首帧恰=last_tgt（零缝步），参考全程前向不重锚——同
        # 同事方案「执行原始计划行、无重锚」的语义。entry-match 同需参考
        # 连续（首行=last_tgt+clip 修正，与入口判据同基准）。非二者口径不变。
        _prev0 = last_tgt if ((args.skip_install or args.entry_match)
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
        rows_abs, frames_all, bursts = build_block_stream(
            _cur_sw, prev0=_prev0, offset=_roff,
            seam=(args.entry_match and n_swapped > 1))
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
            cmd_delta = (tgt - last_tgt) if last_tgt is not None else np.zeros_like(tgt)
            d_mrad = float(np.max(np.abs(cmd_delta))) * 1000
            if seam_first is None:
                seam_first = d_mrad          # 接缝尖峰：换块首发 |Δcmd|
            else:
                seam_rest.append(d_mrad)
            gp = send_grip(robot, chunk[tgi])
            gap0 = float(np.max(np.abs(tgt - cur)))
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
                    # 帧突发：绝对截止配速（预算 tick−reserve 匀速铺满整行），
                    # SDK 单帧调用超预算只体现为丢拍重锚，绝不追发
                    fdt = (TICK - args.burst_reserve_ms / 1000.0) / len(b)
                    _t_f = time.perf_counter()
                    st = None
                    for _bi, _fi in enumerate(b):
                        _fr = frames_all[_fi]
                        _cmds = []
                        for _ji in range(14):
                            _c = JointCommand()
                            _c.position = float(_fr[_ji])   # 标准关节只消费 position
                            _cmds.append(_c)
                        _st = robot.set_joint_commands(_cmds, joint_names=ARM_NAMES,
                                                       time_from_start_s=args.tfs_ms / 1000.0)
                        if not str(_st).startswith("ControlStatus.SUCCESS"):
                            st = _st
                            break
                        _t_f += fdt
                        if _bi < len(b) - 1:
                            _sl = _t_f - time.perf_counter()
                            if _sl > 0:
                                time.sleep(_sl)
                    if st is not None:
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
            last_tgt = tgt.copy()
            k += 1
            ach = read_joints(robot, ARM_NAMES)
            err = float(np.max(np.abs(ach - tgt))) * 1000
            track_err.append(err)
            fr_s = (f"{n_fr}帧@{args.fps * args.frames_per_row:.0f}Hz" if n_fr
                    else "· 免发 ·")
            print(f"  拍{k - 1}: |Δcmd| {d_mrad:5.1f} mrad"
                  f" | 导程 {gap0 * 1000:5.1f} → {fr_s}"
                  f" | 残差 {err:4.1f} mrad"
                  + (f" | 合步[{tgi}]×{_grp_n}" if _is_merged else "")
                  + f" | {gp if gp else st}")

    # ── 出口：最后一块消费到水位（或包络截短后耗尽）──
    _w = min(WMARK, len(chunk))
    if n_swapped >= args.rounds and (k >= _w or k >= len(chunk)):
        break

    # ── 预取：到水位且无在飞推理 → 带前缀启动下块 ──
    if job is None and n_swapped < args.rounds and k >= _w:
        _obs_k = k               # 观测拍摄时点（跳帧安装的流逝基准）
        _obs, _st_n, _st_arm = fresh_obs()
        job = _PredictJob(model, args.prompt)
        job.start(_obs, _st_n, _st_arm, rtc_prefix=_rtc_prefix(_st_arm))

    tick_ms.add((time.perf_counter() - t_tick) * 1000)

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
        gs = robot.get_gripper_state(getattr(G1JointGroup, name))
        if gs is not None:
            print(f"夹爪终态 {name}: {gs.width * 1000:.1f} mm (moving={gs.is_moving})")
fin = read_joints(robot, ARM_NAMES)
exc = float(np.max(np.abs(fin - HOME))) * 1000
print(f"\n== 汇总（{n_swapped} 块 / {len(tick_ms)} 拍 / tick {TICK * 1000:.1f} ms / "
      f"轨迹流 {args.frames_per_row} 帧/行 {args.interp}）==")
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
if args.entry_match and entries:
    _e = np.asarray(entries, float)              # 仅块 2+ 追加（首块无入口）
    print(f"入口匹配: 块 {len(_e)} | j min/p50/max "
          f"{_e.min():.0f}/{np.median(_e):.0f}/{_e.max():.0f}"
          f"（行0 {int((_e == 0).sum())} 块；j 是内容/位姿失配的深读数，"
          "闭环应显著浅于回放台）")
if starve_windows or starve_ticks:
    print(f"星饿: {starve_windows} 段 / {starve_ticks} 拍 / 共 {starve_total:.1f} s"
          f"（最长 {starve_worst:.1f}s，上限 {args.starve_limit}s 未触发即健康）")
if dropped_ticks:
    print(f"⚠ 丢拍重锚 {dropped_ticks} 次（tick 服务超 {TICK * 1000:.1f} ms——"
          "取图拍撞推理落地拍等；持续丢拍说明 --fps 偏高）")
print(f"最终偏离起始位: {exc:.0f} mrad（护栏 {args.max_excursion * 1000:.0f} mrad）"
      + (" ⛔ 护栏触发过" if aborted else ""))

WATCH.restore()
if NAV is not None:
    NAV.resume()
print("SDK 关闭中...")
robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
os._exit(0)   # SDK 残留线程，干净退出
