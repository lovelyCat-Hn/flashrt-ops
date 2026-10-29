#!/usr/bin/env python
"""G1 双臂闭环——轨迹流执行器（set_joint_commands 高频流路线）。

【标准版配方（2026-10-21 用户定档=lerobot 对齐版）】
  标准跑一条命令：run_g1_loop_traj.py --exec
  配方固化在 config/g1.toml [lerobot] 档（三态 CLI > [lerobot] > 内置：
  逐项覆盖=CLI 显式传参；布尔档显式关=--no-X 反关旗标；日志 [config]
  覆盖行可核对生效来源）。旗标/档键名已对齐同事栈词汇
  （execute_actions_per_chunk/skip_frames/aggregate_fn〔上游
  AGGREGATE_FUNCTIONS 注册表；--fusion-alpha=连续 α 逃生门〕/filter_alpha/
  max_joint_delta_per_step/execution_horizon；旧旗标名保留为 CLI 别名）。
  等价 CLI 展开：
    --rounds 200 --execute-actions-per-chunk 0 --skip-frames
    --aggregate-fn weighted_average
    --cmd-every 1 --filter-alpha 0.3 --grip-state-cmd --max-joint-delta-per-step 0.05
    --burst-reserve-ms 0 --nav-suspend --execution-horizon 10 --head-crop
    --ckpt models/pi05_g1_training_rtc_d10_deploy
  语义=同事 lerobot 栈忠实复刻：爪=关节流（--grip-path joint 默认，
  dim7/15 并回 set_joint_commands，P6 破案后 10-21 定档）、tfs=0、
  30Hz 直发（cmd-every 1）、skip-install=同事 skip_frames、
  overlap 0.7=同事 weighted_average（0.3 旧+0.7 新）、EMA=同事
  action_filter（dim23 全维：臂+爪同 α，--grip-filter-alpha 默认跟随、
  仅 A/B 单独覆盖）、水位 0=换块即预取；+trained-RTC（--execution-horizon 10
  须配 sft1012_training_rtc_d10 权重，--head-crop pick 相必带——
  place 相/非此 ckpt 勿带）。逐项依据见 USAGE §3.8。

与 run_g1_loop_native.py（母本）同一条推理链路、同一套护栏与水位换块调度，
只换发令侧：native 每拍发一条位置指令（30Hz 阶跃，10-09 真机七跑定案
抖感∝指令沿能量=阶跃幅值×沿频率）；本脚本走同事 infer_policy 的真机在役
路线——
  换块时一次性：行目标（链式 ±--max-joint-delta-per-step rad/行 限幅）→ [当前实际位]+
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
RTC：--execution-horizon N > 0 时启用推理侧前缀引导（同 native，FlashRT 引擎
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
  --merge N（10-26 replay 台移植）：另一种合步=等时粗化——滤波后每 N 行取
  1 个宏步、宏步段帧按拍均分，指令沿=fps×frames-per-row/N 且发令**均匀无洞**
  （cmd-every 是组末单发+静默拍）；与 cmd-every 族互斥，二者取一。
  --continuous-ref：换块链从 last_tgt 接续不重锚（replay 6f52423 移植）。
参数命名对 lerobot 术语的对照：docs/lerobot-alignment.md。

⚠ 加 --exec 会连续真实驱动双臂！安全设计（与 native 同款）：
  1. 只动双臂 14 关节；腿/头维度永不下发；夹爪仅 --grip 显式开启时下发
     （0~100%% → manifest 标定宽度，变化超 --grip-chg 才发，非阻塞不等反馈）
  2. 换块时行目标链式限幅 ±--max-joint-delta-per-step rad/行 + 逐帧包络钳位；漂移护栏：
     行目标偏离起始位超 --max-excursion → 立即停止循环
  3. q 键即时退出（后台监听线程，SDK 阻塞中也可退）；物理急停第一优先级
  4. 执行前回车确认

每块遥测：推理 ms / 取图 ms / tick 服务时长与抖动 / 星饿窗 /
接缝尖峰（换块首行 |Δcmd| vs 块内均值——RTC 收益主指标）/
汇总分位数（LatencyTracker，lerobot 同款 API）。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/g1/inference/run_g1_loop_traj.py \
      [--ckpt ~/holy/models/pi05_g1_onlypick_deploy] [--exec] \
      [--rounds 60] [--execution-horizon 10]
  # 240Hz 微流（--frames-per-row 8）已于 10-10 退役，默认 1=纯 30Hz 直发
"""
#!/usr/bin/env python
"""S1 双臂闭环——轨迹流执行器（run_g1_loop_traj.py 直移植版，S1 机体件）。

【移植原则（2026-10-29 用户定案）】G1 traj 就是行为标准，写 traj 就是为了
S1 直接用——本文件 = run_g1_loop_traj.py 整文件复制，只换「机体件」：
  ① SDK 面：galbot_sdk.g1 → galbot_sdk.s1（G1JointGroup 枚举 → S1 组名字符串
    "left_gripper"/"right_gripper"，写路径 set_joint_commands 签名同形）
  ② 关节表/state：27 维 = [右臂7, 右爪%, 左臂7, 左爪%, torso, head2, swerve×8]
    （pi05 S1 数据集序；爪=get_gripper_state 宽度米→0~100%；swerve 恒 0=
    训练口径；torso=torso_lift_joint1 高度米）
  ③ 取图：get_rgb_data(JPEG)（G1 同名相机 HEAD_RIGHT/LEFT_ARM/RIGHT_ARM）
  ④ 发令批形态：G1 traj 逐帧单点 tfs=0 流 → S1 带时间轴批 set_joint_commands_batch
    （帧=tick 内插值帧，tfs 全批累计=墙钟节拍；--rows-per-cmd R>1 再跨拍攒窗=
    R×tick 窗每 R 拍一发）。S1 ③定案（10-29 晚真机）：singorix WBC 按 tfs
    时间轴细插，tfs=0 ASAP=参考跳变→LP 削顶→力矩跳变=卡卡；时间轴是解药
    ——故移植版把「每帧一条单点指令」换成「带时间轴批」（同回放台/适配器
    执行器真机验证过的下发形状，四角闭环④）。--cmd-every>1 S1 硬拦（审查④：
    S1 WBC 直视计划断点，G1 伺服吞阶跃语义不成立）
  ④b 取图在 worker：fresh_obs（三相机 ~86ms）随推理搬进工作线程（10-29 审查①，
    主线程同步取图≈2.6 拍=每块预取拍丢拍重锚）；--no-obs-worker 回退主线程
  ⑤ 爪标定：S1 ckpt 无 manifest gripper 段 → 出厂行程缺省 0.007/0.11 m
    （policies/base.py GRIP_WIDTH_*，pyi 同值）+ 告警（未逐台标定）
  ⑥ warm：G1 手跑 g1_pose_warmup.py → 移植版内嵌同款机制（set_joint_positions
    is_blocking 限速 0.15 rad/s 臂+头 → torso @0.2，G1 send_warm 节奏），
    --exec 确认后就位到 S1_HOME_POSE、自然跑完归位；q/Ctrl-C 照 traj=立即
    退出不复位（已下发目标可能仍在限速执行）
  ⑦ 导航栈：S1 无 nav 栈，NavSuspend 整类删除（--nav-suspend 旗标保留兼容
    档键，生效 no-op）；包络护栏=S1 ckpt 无 joint_envelope.json 时自动关闭
行为逻辑（水位预取/换块弃尾/星饿限/链钳/EMA/融合/入口匹配/合步/合并/
护栏/QuitWatcher/日志/汇总）逐行未动。参数命名对照 docs/lerobot-alignment.md。

【标准版配方】固化在 config/s1.toml（三态 CLI > toml > s1_config.BUILTIN），
等价 CLI 展开：--rounds 200 --execute-actions-per-chunk 0 --skip-frames
--aggregate-fn weighted_average --cmd-every 1 --filter-alpha 0.1
--max-joint-delta-per-step 0.05 --burst-reserve-ms 0 --frames-per-row 8
--rows-per-cmd 2 --grip-path joint --grip-velocity 0.05 --grip-effort 10
--fps 30
（真机跑必带 --exec；10-29 首跑教训：S1 版无合成图路径，取图即真实相机，
无 --live-frames 概念——相机冻结守卫 _FREEZE_STREAK 常开）

⚠ 加 --exec 会连续真实驱动双臂！安全设计（与 G1 traj 同款）：
  1. 只动双臂 14 关节；torso/head/swerve 维永不下发；夹爪随 --grip 并入
     关节流（0~100%% → 行程宽度，每发送拍无条件发）
  2. 换块时行目标链式限幅 ±--max-joint-delta-per-step rad/行 + 逐帧包络钳位；漂移护栏：
     行目标偏离起始位超 --max-excursion → 立即停止循环
  3. q 键即时退出（后台监听线程，SDK 阻塞中也可退）；物理急停第一优先级
  4. 执行前回车确认（--exec 就位 warm 前确认一次，进环前再确认一次）

每块遥测：推理 ms / 取图 ms / tick 服务时长与抖动 / 星饿窗 /
接缝尖峰（换块首行 |Δcmd| vs 块内均值）/ 汇总分位数（LatencyTracker）。

用法:
  LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
  ~/miniforge3/envs/flash_pyrt311/bin/python \
      ~/holy/scripts/s1/inference/run_s1_loop_traj.py --exec
  # 配方默认全在 config/s1.toml；ckpt 缺省=S1 pi05 工作点
"""
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
                     "config（S1 现役 8=4.17ms 点距≈wbc_dt 4ms 采样周期；G1 "
                     "traj 默认 1=逐帧单点直发——S1 ③定案 tfs=0 ASAP=卡卡，"
                     "勿抄）")
ap.add_argument("--rows-per-cmd", type=int, default=None,
                help="跨行攒批行数 R（10-29 审查②）：R=1=每拍一包（行内 fr 帧，"
                     "窗口 33.3ms）；R>1=攒 R 拍帧并一包 set_joint_commands_batch，"
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
         if args.seam_overlap_alpha > 0 else "")
      + (" | 偏移平移" if args.ref_offset else "")
      + (f" | RTC 引导 horizon {args.rtc_horizon}（max_w {args.rtc_max_w}）"
         if args.rtc_horizon > 0 else " | RTC 关")
      + (" | 头图训练裁剪放大" if args.head_crop else "")
      + (f" | 帧到达时限 {args.tfs_ms:g} ms" if args.tfs_ms > 0 else ""))


# ── 导航栈搁置：S1 无导航栈，NavSuspend 整类不移植 ──
#（--nav-suspend 旗标/档键保留只为与 G1 traj 配置同形；生效 no-op）
NAV = None   # G1 版此处指向 NavSuspend 实例；S1 恒 None，各退出路径 None 守卫照旧


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
print(f"[ckpt] {CKPT}")  # 日志自证权重（160917 盲区：日志无法判 A/B 跑的哪个 ckpt）

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
from policies.base import (  # noqa: E402  S1 机体常量（run_s1_loop 同源）
    HEAD_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, S1_HOME_POSE,
    TORSO_JOINTS, gripper_pct_to_width_m, gripper_width_m_to_pct)

# ── 关节表（S1 数据集维序：右臂在前；27 维 state）──
LEFT = LEFT_ARM_JOINTS
RIGHT = RIGHT_ARM_JOINTS
ARM_NAMES = RIGHT + LEFT            # 与 chunk[0:7]+chunk[8:15] 逐位配对
SWERVE_NAMES = [f"swerve_joint{i}" for i in range(1, 9)]   # 占位名（恒喂 0，
                                    # 不参与读——训练里该 8 维无信息，见记忆）
STATE_NAMES = (RIGHT + ["right_gripper"]      # 爪列名=S1 组名（get_gripper_state 键）
               + LEFT + ["left_gripper"]
               + TORSO_JOINTS                 # torso_lift_joint1（高度米）
               + HEAD_JOINTS + SWERVE_NAMES)  # 7+1+7+1+1+2+8 = 27
GRIP_IDX = (7, 15)                  # state 里的夹爪维：SDK 米 → 数据集 0~100%
GRIP_NAMES = ("right_gripper", "left_gripper")   # S1 爪组实名（get/set 端点）
ARM_SLICE = list(range(0, 7)) + list(range(8, 15))   # state 里的臂维（0-6 右 / 8-14 左）
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


def state_from_joints(vals) -> np.ndarray:
    """原始关节读数 → 模型 state：夹爪两维按行程换算 0~100%。

    数据集 state dim7/15 是百分比；SDK 读数是米——直接喂会被归一化成
    恒 ≈闭合（0.11m 当 0.11% 算），模型看到的夹爪状态永远错。
    行程缺省=出厂 0.007/0.11 m（S1 ckpt 无 manifest 标定段，见上）。
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
    """S1 关节读取拼装。ARM_NAMES(14)=一次读臂（get_joint_positions 返回请求
    序，按请求序配对——⚠ 序坑在 get_joint_states，此路不踩）；
    STATE_NAMES(27)=臂 + torso/head 真值 + 双爪 get_gripper_state 宽度米
    （state_from_joints 再换算 %）+ swerve 8 维恒 0（训练口径：该 8 维
    常转无信息，喂 0）。
    """
    if len(names) == len(ARM_NAMES):
        vals = robot.get_joint_positions(joint_names=list(names))
        if not vals or len(vals) != len(names):
            raise SystemExit(f"关节读取失败（返回 {len(vals) if vals else 0} 维）")
        return np.array(vals, dtype=np.float32)
    arms = robot.get_joint_positions(joint_names=ARM_NAMES)
    if not arms or len(arms) != len(ARM_NAMES):
        raise SystemExit(f"臂读取失败（返回 {len(arms) if arms else 0} 维）")
    tb_h = robot.get_joint_positions(joint_names=TORSO_JOINTS + HEAD_JOINTS)
    if not tb_h or len(tb_h) != len(TORSO_JOINTS) + len(HEAD_JOINTS):
        raise SystemExit("torso/head 读取失败")
    out = np.zeros(len(STATE_NAMES), dtype=np.float32)
    out[0:7] = arms[0:7]                                   # 右臂
    _g = robot.get_gripper_state(GRIP_NAMES[0])
    out[7] = _g.width if _g is not None else GRIP_WMIN     # 右爪 米
    out[8:15] = arms[7:14]                                 # 左臂
    _g = robot.get_gripper_state(GRIP_NAMES[1])
    out[15] = _g.width if _g is not None else GRIP_WMIN    # 左爪 米
    out[16:19] = tb_h                                      # torso + head2
    return out


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


# ── 头图训练裁剪放大（--head-crop，2026-10-19 trained-RTC 权重试跑）────────
# 复现同事数据管线对头相机的处理：/home/galbot/image_preprocessor.py
# ImagePreprocessor._apply_head_training_crop 原样内联（其顶层 import
# emulate_data_collection 在 /home/galbot/infer_policy/，跨目录 import 脆弱，
# 故不 import 只内联，常量逐一对齐源文件；与 ab_test/1015 B 线内联逐位一致）。
# 训练视频头图 = letterbox224 帧上裁 [126:214, 52:172] → Lanczos4 拉回
# 164×224（≈1.87× 放大）→ 上下各补 30px 黑条回 224。仅作用头相机键 "image"
# （双腕图训练无此处理）；裁剪/缩放/补边均为逐通道空间算子，BGR/RGB 通道序无关。
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
        d = robot.get_rgb_data(getattr(SensorType, cam),
                               RgbOutputFormat.JPEG, True)
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
            _lb = head_training_crop(_lb)   # 头图训练裁剪放大（仅头相机）
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

    ⚠ 10-21 起默认路径是 send_grip_joint（--grip-path joint=lerobot 采集
    语义）；本函数=set_gripper_command 行程控制器路【逃生门/复现对照】，
    下列各门只在此路适用（该路 onset=0.12s+行程/速度+吞令=训练外注入行为，
    P6 三格破案在案）。

    只发不查不等待（拖死控制环）；终态核对放循环结束后。回读新鲜（10-13
    标定 p50 10ms），实际下发拍顺带采样一次物理实宽进日志（[实 X.Xmm]），
    爪的物理时序证据逐 send 可见。已知物理动态（同标定）：起步死区 ~0.9s +
    0.15 m/s 帽（满程 ~0.8s）。
    ⚠ 111901 终审：每条新指令把行程从头重启（死区重吃）——逐拍补发的指令流
    p50 间隙 2-3 拍，死区永不耗尽 → 110 send 全 SUCCESS 实宽全程 0.5mm 冻结；
    q 停流静→最后一条 1.5s 走满（「q 后突然张开」全案闭合）。故
    --grip-ss 状态机=少发（每相 1-2 条、每条 ≥2s 静默窗）；无 ss 时退回旧
    chg 门（2% 变化即发，111901 证明该模式爪会冻结——保留只为对照）。
    --grip-send-every N=定时静默制（2026-10-20 定稿）：chg 死区门整体旁路，
    每 N 个机会拍无条件发当前 EMA 值一条、其余拍纯静默 → 爪在静默窗内跑完
    行程；N=4→≈7.5Hz/静默 75%。纯去掉死区（≡chg=0）=143831 实证完全冻结，
    勿用。⚠ 10-20 晚定律终版（174428）补判决：拍隙 <0.9s 死区的周期拍从静止
    永起步不了（合步2 N=4→0.27s=永冻），此路要真静默须 N≥18。
    --grip-min-interval N=事件门（2026-10-20 定律终版，三选一门之一）：距上次
    实发 ≥N 秒才发（固件 onset=0.12s+行程/速度，0.15 全行程 0.91s < N=1.2 →
    下一条来时上一条必已在途，吞令结构性不可能）；间隔未满仅「在途(|v|>0.02)
    且意图差≥3%」豁免——在途同向/反向均平滑零罚（P2 174428 满速零凹陷滑到底；
    P4 183232 反向掉头 ~0.2s 不碰零；10-16「反转重启律」随之改写：杀手=死区
    密发，方向只是伴随特征）。
    --grip-preclose V=预合爪爬行【185317 极限梯判死勿启用】：固件速度下限
    0.03 m/s（0.01 指令实测执行 0.03）且 onset=0.12s+行程/速度——爬速 0.03
    全行程起步要 ~4.1s，慢速反把起步拖晚，「低速吸收死区」前提不成立。
    """
    if GRIP is None:
        return None
    if not args.do_exec:
        return "dry（不下发）"
    parts = []
    now = time.monotonic()
    # --grip-send-every：节流判定每调用一次（=一个机会拍），R/L 共用同一判定
    # 保持同拍配对；合步模式只组末拍调用，节奏自然跟随。chg 门在节流模式下
    # 整体旁路（死区逻辑去掉，见函数 docstring 的剂量-响应证据）。
    global _GRIP_BEAT
    _pace = int(args.grip_send_every) > 0
    _send_beat = False
    if _pace:
        _send_beat = (_GRIP_BEAT % int(args.grip_send_every)) == 0
        _GRIP_BEAT += 1
    _minint = float(args.grip_min_interval or 0.0)
    _pre = float(args.grip_preclose or 0.0)
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
        elif _minint > 0:
            # 事件门（定律终版）：间隔窗=静默保证起步；在途同向豁免=免费微调。
            # 意图没动（<max(chg,3%)）连间隔窗也不发——零travel重发只会白白
            # 重置静默钟、推迟下一条真令。
            lt = _GRIP_SENT_AT.get(name)
            moved = last is not None and \
                abs(p - last) >= max(float(GRIP["chg"] or 2.0), 3.0)
            if lt is None:
                why = "首拍"
            elif not moved:
                parts.append(f"{name[0].upper()} {p:.1f}%·hold")
                continue
            elif now - lt >= _minint:
                why = "间隔窗"
            else:
                _gs = robot.get_gripper_state(name)
                if _gs is not None and abs(_gs.velocity) > 0.02:
                    why = "在途"                # P2/P4 双实锤：在途同向/反向均
                                                # 平滑零罚（183232 掉头仅 ~0.2s
                                                # 减速-再加速，不碰零）
            if not why:
                parts.append(f"{name[0].upper()} {p:.1f}%·hold")
                continue
        elif not _pace and last is not None and abs(p - last) < GRIP["chg"]:
            parts.append(f"{name[0].upper()} {p:.1f}%·hold")
            continue
        if _pace and not _send_beat:         # 节流窗内的拍一律不发（ss 放行也压住）
            parts.append(f"{name[0].upper()} {p:.1f}%·hold")
            continue
        # --grip-preclose：意图下行（合爪）且未平稳 → 爬速下发；平稳回正常速收敛
        spd = GRIP["speed"]
        _tag = ""
        if _pre > 0:
            h = _GRIP_PRE_HIST.setdefault(name, [])
            h.append((now, p))
            while h and now - h[0][0] > 0.75:
                h.pop(0)
            if len(h) >= 2 and p < h[0][1] - 3.0:   # 净降 >3 点=真下行（±2 平台
                spd = _pre                           # 摆动不重燃爬速，EMA 流常见）
                _tag = " 爬"
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        st = robot.set_gripper_command(name,
                                       w, spd, GRIP["effort"], False)
        GRIP["sent"][name] = p
        _GRIP_SENT_AT[name] = now
        if GRIP_SS is not None:
            _GRIP_SS_SENT[name] = (now, p)
        _gs = robot.get_gripper_state(name)
        # eff 回读单位未知（144325 实证：饱和轨 100、全开 114mm 也 100，
        # 头文件 "Newtons" 被推翻）——只当负载/堵转指示，勿按牛顿判读
        _phys = (f" 实{_gs.width * 1000:.1f}mm eff{_gs.effort:.1f}"
                 if _gs is not None else " 实?")
        parts.append(f"{name[0].upper()} {p:.1f}%→{w * 1000:.0f}mm@{spd:.2f}{_tag}"
                     f"{_phys} {str(st).replace('ControlStatus.', '')}"
                     + (f"({why})" if why else ""))
    return "  ".join(parts)


# --grip-path joint：爪维走关节固件路径的独立 EMA 状态（与 cmd 路径 _GRIP_EMA
# 分开，A/B 切路径不串状态）
_GRIPJ_EMA = {}
# --grip-hold-on-stall 憋死保持状态（每爪独立）：on=保持中；hist=实宽平台窗；
# enters=累计进入次数（汇总打印）；tgt=冻结目标；floor=进入时实宽（外松上限）；
# rail=eff 贴轨连续拍数（自动外松节拍器）；relax_n=累计外松步数（汇总打印）；
# quiet=距上次武装的已发拍数（静默制节拍器）；silent_n=累计静默拍数（汇总打印）
_GRIPJ_HOLD = {}
# --gripj-every 爪通道节拍器（全局连续拍，不随换块复位）
_GRIPJ_TX = 0


def send_grip_joint(robot, chunk_row, tfs_s, frame=None):
    """--grip-path joint：夹爪 dim7/15 并回 set_joint_commands 流（10-21 P6 破案）。

    同事 infer_policy / lerobot 采集语义原样：position=width_m、
    velocity=速度帽（P6j 实测 0.15→0.1500 被尊重，非 pursuit 追踪）、
    effort=10（同事默认；非 set_gripper_command 路的 30）、tfs 与臂同参。
    「onset=0.12s+行程/速度＋吞令」整条定律属 set_gripper_command 端点
    （P6 三格实证：关节路径 onset≈0.14s 纯 L 与行程无关、密流照跟不冻），
    故 chg/min-interval/send-every/ss/preclose 各门在此全部旁路——每发送拍
    无条件发当前 EMA 值（密流在该路径免费，P6j_dense 实证）。EMA 照带
    （独立状态 _GRIPJ_EMA）。双爪一包：groups=[right_gripper,left_gripper]
    顺序=命令序（P6 单组已验组写安全；group 模式读的顺序坑不适用于写）。

    GRIP["sent"] 照常更新 → --grip-state-cmd / 遥测 / sent_peak 持物签名
    无感知路径切换。

    --grip-hold-on-stall（2026-10-21 力矩碰撞案）：模型按训练语义持物段持续
    发全闭指令，位置伺服每拍重发=斜坡重启反复撞进物体。检测「指令≤实宽−gap
    且实宽平台」→ 改发 进入时实宽−margin 冻结目标@effort_hold（钳位保持；
    ⚠勿逐拍跟随实宽=棘轮碾进，144325 首跑实证）；指令上穿实宽+exit
    （想开爪）→ 恢复流式并清平台史。只改物理下发，不改指令记账。
    自动外松（--grip-hold-relax-mm，095207 定案）：钳位后稳态误差=margin 仍
    使伺服满驱（eff 恒贴饱和轨 100，「有差就追」=位置伺服天性，收 margin 无效
    ——1mm 时 156/159 拍仍贴轨可闻声）。SDK 无阻抗模式，执行侧导纳仿真：eff
    贴轨持续 6 拍→目标外松一步，直到 eff 掉头或目标=进入时实宽（零误差停追）。
    单向只松不紧=无棘轮。
    真静默制（--grip-hold-quiet-n，101418 定案）：外松到零误差后 eff 仍恒贴轨
    100+响（112 拍无衰减）——每拍重发冻结目标虽逐位相同，但 10-20 爪定律=每条
    新令打断在途从头重来，密发把爪永远钉在起步段 settle 不了。进 hold/外松步后
    只发 N 拍即停发该爪（固件执行最后一条收工），读回/退出/外松评估照常。
    --gripj-every 爪通道节流（10-22 合步 A/B 判读）：仅物理下发被节流，
    状态机照常每拍跑（窗口墙钟不变）。
    """
    if GRIP is None:
        return None
    if not args.do_exec:
        return "dry（不下发）"
    global _GRIPJ_TX
    _tx = True
    if args.gripj_every > 1:
        _tx = _GRIPJ_TX % args.gripj_every == 0
        _GRIPJ_TX += 1
    parts = []
    cmds, groups = [], []
    for _ni, (name, dim) in enumerate(GRIP["names"]):
        if frame is not None:
            # 爪随臂宏步管线（2026-10-26）：帧值已在上游行准备中 EMA+插帧，
            # 此处只钳位——二次 EMA 会双滤波
            p = float(np.clip(frame[_ni], 0.0, 100.0))
        else:
            p = float(np.clip(chunk_row[dim], 0.0, 100.0))
            if args.grip_ema_alpha > 0:
                prev = _GRIPJ_EMA.get(name)
                p = p if prev is None else \
                    args.grip_ema_alpha * p + (1.0 - args.grip_ema_alpha) * prev
                _GRIPJ_EMA[name] = p
        GRIP["sent"][name] = p   # 记模型意图（state-cmd/sent_peak 语义不变；
                                 # hold 只改物理下发，不改指令记账）
        w = GRIP["wmin"] + p / 100.0 * (GRIP["wmax"] - GRIP["wmin"])
        _gs = robot.get_gripper_state(name)
        w_sent, eff_sent, vel_sent, _tag = w, 10.0, GRIP["speed"], ""
        _skip = False
        if args.grip_hold_on_stall and _gs is not None:
            phys = _gs.width
            h = _GRIPJ_HOLD.setdefault(name, {"on": False, "hist": [],
                                              "enters": 0, "tgt": None,
                                              "floor": None,
                                              "rail": 0, "relax_n": 0,
                                              "quiet": 0, "silent_n": 0})
            # 双史窗口：(实宽, 本拍指令是否够深)。进入须「实宽平台」∧「深指令
            # 持续满窗」——150603 死锁教训：爪静停开位时模型刚下合令，只看
            # 当拍深指令会误进（onset 未到爪还没动）→ 冻结在开位=意图永不上
            # 穿 exit 门、爪全程不闭合。满窗深指令=「给了 onset+行程时间仍到
            # 不了」才算真憋死；自由爪满窗前早动了，平台破即不进。
            h["hist"].append((phys,
                              w <= phys - args.grip_hold_gap_mm / 1000.0))
            _win = max(2, int(args.grip_hold_window))
            while len(h["hist"]) > _win:
                del h["hist"][:-_win]
            _plateau = (len(h["hist"]) >= _win and
                        (max(p for p, _ in h["hist"])
                         - min(p for p, _ in h["hist"])) * 1000.0
                        <= args.grip_hold_plateau_mm)
            _deep_all = all(d for _, d in h["hist"])
            if h["on"]:
                if w >= phys + args.grip_hold_exit_mm / 1000.0:
                    h["on"] = False
                    h["hist"].clear()   # 退出清史：再进须新平台（防近阈闪断）
                    h["tgt"] = None
                    h["floor"] = None
                    h["rail"] = 0
                    h["quiet"] = 0
                    _tag = " HOLD退出"
            elif _plateau and _deep_all:
                h["on"] = True
                h["enters"] += 1
                # 进入时冻结目标（144325 教训：逐拍跟随 实宽−margin=棘轮
                # 碾进；冻结后刚性物同样钉死、软物最多压进 margin 一次）
                h["tgt"] = max(0.0,
                               phys - args.grip_hold_margin_mm / 1000.0)
                h["floor"] = phys   # 外松上限=进入时实宽（零误差=停追静音）
                h["quiet"] = 0      # 重新武装：进 hold 先发满 quiet-n 拍
                _tag = " HOLD进入"
            if h["on"]:
                if args.grip_hold_relax_mm > 0 and h["floor"] is not None:
                    # eff 闭环自动外松（执行侧导纳仿真，095207 实证：1mm 误差
                    # 仍伺服满驱 eff 恒贴轨=「有差就追」天性，收 margin 治不了
                    # 声）。贴轨满 6 拍→目标外移一步；eff 掉头或到进入时实宽
                    # （零误差）即停。单向只松不紧=无棘轮回旋余地。先于
                    # w_sent 赋值：步进当拍即按新目标下发。
                    if _gs.effort >= args.grip_hold_relax_eff:
                        h["rail"] += 1
                    else:
                        h["rail"] = 0
                    if h["rail"] >= 6 and h["tgt"] < h["floor"] - 1e-6:
                        h["tgt"] = min(h["floor"],
                                       h["tgt"] + args.grip_hold_relax_mm
                                       / 1000.0)
                        h["rail"] = 0
                        h["relax_n"] += 1
                        h["quiet"] = 0   # 外松步重新武装：发满 quiet-n 拍
                        _tag = f" HOLD松{h['tgt'] * 1000:.1f}"
                # 目标恒定=稳态误差恒为 margin（观测 093335：误差即伺服持续
                # 内顶驱动力，故 margin 1mm+保持段速度帽放软双管压声）
                w_sent = h["tgt"]
                if args.grip_effort_hold > 0:
                    eff_sent = args.grip_effort_hold
                if args.grip_hold_vel > 0:
                    vel_sent = min(args.grip_hold_vel, GRIP["speed"])
                _tag = _tag or " HOLD"
                if args.grip_hold_quiet_n > 0:
                    # 静默制（101418 定案）：外松到零误差 eff 仍恒贴轨+响——
                    # 每拍重发虽逐位相同，但 10-20 爪定律=每条新令打断在途
                    # 从头重来，密发把爪永远钉在起步段 settle 不了。发满
                    # quiet-n 拍后本爪退出指令包=真静默，让最后一条走完收工；
                    # 读回/退出/外松评估照常（不依赖发送）。
                    if h["quiet"] >= args.grip_hold_quiet_n:
                        _skip = True
                        h["silent_n"] += 1
                        _tag = " HOLD静"
                    else:
                        h["quiet"] += 1
        c = JointCommand()
        c.position = float(w_sent)
        c.velocity = float(vel_sent)        # 速度帽语义：恒速段被钳在此值
        c.effort = float(eff_sent)
        if not _skip and _tx:   # 静默制/节流：该爪本拍退出指令包（读回照常）
            cmds.append(c)
            groups.append(name)
        # eff 回读单位未知（144325 实证：饱和轨 100、全开 114mm 也 100，
        # 头文件 "Newtons" 被推翻）——只当负载/堵转指示，勿按牛顿判读
        _phys = (f" 实{_gs.width * 1000:.1f}mm eff{_gs.effort:.1f}"
                 if _gs is not None else " 实?")
        if not _tx and not _skip:
            _tag = f"{_tag} 跳" if _tag else " 跳"
        parts.append(f"{name[0].upper()} {p:.1f}%→{w_sent * 1000:.0f}mm@j"
                     f"{_phys}{_tag}")
    if not cmds:    # 双爪皆静默：本拍无指令包
        return "  ".join(parts) + " j静"
    st = robot.set_joint_commands(cmds, joint_groups=groups,
                                  time_from_start_s=tfs_s)
    ok = str(st).startswith("ControlStatus.SUCCESS")
    return "  ".join(parts) + (" jOK" if ok else
                               f" j{str(st).replace('ControlStatus.', '')}")


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
    raise SystemExit("控制器未 SUCCESS，终止（排查 SDK/急停/控制器状态）")

if args.nav_suspend:
    print("[nav-suspend] S1 无导航栈——旗标生效为 no-op（兼容档键保留）")


# ── 就位/归位 warm（G1 g1_pose_warmup·send_warm 机制内嵌，2026-10-29 移植）──
# G1 流程=手跑 warmup 脚本；S1 移植版内嵌同款：set_joint_positions is_blocking
# 限速——臂+头 @0.15 rad/s → 段间 1s → torso @0.2（升降柱，高度米语义）→
# 双爪关节流一次全开。⚠ 严禁快照/链钳快移复位（10-29 run_s1_loop 版
# 1.5 rad/s 链钳被用户判「一瞬间复位」，10× 超速；本函数=治愈版）。
# 目标=S1_HOME_POSE（policies/base.py 烘档=clean_ready_224_1028 全 47 集
# 首帧逐关节中位，in-distribution 起点）。
def s1_warm(robot, tag: str) -> bool:
    tgt = ([float(v) for v in S1_HOME_POSE["right_arm"]]
           + [float(v) for v in S1_HOME_POSE["left_arm"]]
           + [float(v) for v in S1_HOME_POSE["head"]])
    st = robot.set_joint_positions(
        tgt, joint_names=RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS + HEAD_JOINTS,
        is_blocking=True, speed_rad_s=0.15, timeout_s=45.0)
    print(f"[warm] {tag} 臂+头 @0.15 rad/s → {st}")
    if not str(st).startswith("ControlStatus.SUCCESS"):
        return False
    time.sleep(1.0)                      # G1 send_warm 节奏：段间 1s
    st = robot.set_joint_positions(
        [float(S1_HOME_POSE["torso"])], joint_names=list(TORSO_JOINTS),
        is_blocking=True, speed_rad_s=0.2, timeout_s=45.0)
    print(f"[warm] {tag} torso @0.2 → {st}")
    if not str(st).startswith("ControlStatus.SUCCESS"):
        return False
    _w = gripper_pct_to_width_m(S1_HOME_POSE["grip_pct"])
    _cmds = []
    for _g in GRIP_NAMES:                # 双爪一包（关节流，速度帽/effort 生效）
        _c = JointCommand()
        _c.position = float(_w)
        _c.velocity = float(args.grip_speed)
        _c.effort = float(args.grip_effort)
        _cmds.append(_c)
    st = robot.set_joint_commands(_cmds, joint_groups=list(GRIP_NAMES),
                                  time_from_start_s=0.0)
    print(f"[warm] {tag} 双爪全开 {_w * 1000:.0f} mm → {st}")
    return str(st).startswith("ControlStatus.SUCCESS")


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
GRIP_INTERP = create_interpolator(args.interp, dim=2,
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
_GRIP_BEAT = 0       # --grip-send-every 机会拍计数（send_grip 每调用 +1）
_GRIP_SENT_AT = {}   # 事件门每爪上次实发时刻（--grip-min-interval；send_grip 记账）
_GRIP_PRE_HIST = {}  # 预合爪每爪意图史 (t, p)，0.75s 窗（判「下行未平稳」用）


def build_rows(cur, prev0=None, offset=None, seam=False):
    """chunk delta 块 → 链式限幅行目标（14 维，右臂在前）。

    ⚠ 2026-09-23 语义修正（同母本 plan_step）：训练管线
    relative_actions_processor 把臂维动作转成 delta（相对本块预测时刻臂位
    BASE_ARM）——行目标不是 chunk 绝对值，须加回 BASE_ARM（换块时随 job 更新）。
    链式：rows[0]=cur 向行 0 绝对位限幅挪一步，rows[k]=rows[k−1] 再向行 k 挪——
    每行位移 ≤ --max-joint-delta-per-step，且逐行夹在 cur 与 chunk 绝对位之间（包络盒凸性 →
    行目标自动在包络内；样条过冲由 build_block_stream 的帧级钳位兜底）。
    EMA：--filter-alpha>0 时状态跨块连续（infer_policy action_filter 同款语义，
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
        # 接缝 α 助推（--seam-boost-alpha，2026-10-26 块间停顿案 160323）：
        # 与 seam-ramp 反向。换块 EMA 重播回 last_tgt（防瞬移修正）清零了
        # 速度状态 → 命令速度块首=α×模型速度、τ=1/α≈整块，水位0 短块(11拍)
        # 永在瞬态=目视停顿；块首 n 拍用大 α 快速重建速度再线性落回主 α，
        # 稳态纹理零税。α 全程 0.3 对照=纹理粗 50%（161429 残差 p50
        # 11.6→20.7），勿再走。与 _ramp 互斥（守卫把关，此处 boost 优先）。
        _boost = args.seam_boost_alpha > 0 and _EMA_STATE is not None
        a_boost = float(args.seam_boost_alpha)
        n_bb = max(1, int(args.seam_boost_beats))
        acc = None
        for kk in range(len(rows)):
            if _boost:
                a_k = a_boost + (a - a_boost) * min(kk, n_bb) / n_bb
            elif _ramp:
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
    per = args.frames_per_row
    if args.merge > 1:
        # 合步（等时粗化，replay 台 3b40d0e 同构）：滤波后每 N 行取 1 个宏步
        #（末行必保=块尾/链状态不动），宏步段 P 帧按拍均分 N 组——同轨迹同速
        # 同墙钟只稀疏化目标，下发 ≈fps×per/N Hz。样条跑在宏步序列上（声明
        # input_hz 仍 30=replay 同款时间折法），行级 EMA/链钳已在 build_rows
        # 完成。返回行=重复宏步行（k 逐拍推进与水位/星饿/RTC 语义不变）。
        idx = list(range(0, len(rows), args.merge))
        if idx[-1] != len(rows) - 1:
            idx[-1] = len(rows) - 1
        rows = np.repeat(rows[idx], args.merge, axis=0)
        n_macro = len(rows) // args.merge
        waypoints = np.vstack([_start[None, :], rows[::args.merge]])
        frames_all = INTERP.interpolate(waypoints)
        if ENV_LO is not None:
            np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :],
                    out=frames_all)
        bursts = []
        for _m in range(n_macro):
            for _grp in np.array_split(np.arange(_m * per, (_m + 1) * per),
                                       args.merge):
                bursts.append([int(i) for i in _grp])   # 空组=静默拍（per<N 时）
        # ── 爪维随臂同管线（2026-10-26 用户定案；--grip-rawrows 恢复旧行率
        # 流）：dim7/15 与臂同 idx 宏步粗化+同款样条逐帧插帧。EMA 上移到行
        # 准备（镜像臂 build_rows：滤波→粗化→插帧，勿在 send 端二次滤波），
        # 换块从 GRIP['sent']（最后实发意图）重播=臂 _EMA_STATE=last_tgt 的
        # 防瞬移镜像；首块/无历史以首行播种。起点=GRIP['sent']（prev0 镜像，
        # 块首帧恰=上块末发值=零缝步）。帧值钳 [0,100]（样条过冲兜底）。
        grip_frames = None
        if GRIP is not None and not args.grip_rawrows:
            def _sent_seed():
                _v = [GRIP["sent"].get(n) for n, _d in GRIP["names"]]
                return (None if any(x is None for x in _v)
                        else np.asarray(_v, dtype=np.float32))
            g = np.stack([chunk[:, d] for _, d in GRIP["names"]],
                         axis=1).astype(np.float32)
            if args.grip_ema_alpha > 0:
                _ab = float(args.grip_ema_alpha)
                _seed = _sent_seed()
                acc = g[0].copy() if _seed is None else _seed.copy()
                for _k in range(len(g)):
                    acc = _ab * g[_k] + (1.0 - _ab) * acc
                    g[_k] = acc
            _seed = _sent_seed()
            _g_start = g[0].copy() if (prev0 is None or _seed is None) else _seed
            _g_wp = np.vstack([_g_start[None, :], g[idx]])
            grip_frames = np.clip(GRIP_INTERP.interpolate(_g_wp), 0.0, 100.0)
        return rows, frames_all, bursts, grip_frames
    waypoints = np.vstack([_start[None, :], rows])
    frames_all = INTERP.interpolate(waypoints)
    if ENV_LO is not None:
        np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :], out=frames_all)
    bursts = [list(range(kk * per, (kk + 1) * per)) for kk in range(len(rows))]
    return rows, frames_all, bursts, None


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
_EMA_STATE = None                    # 循环起点重置：干跑预览那次 build 不作块 1 的跨块历史
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
        _ovp = (args.seam_overlap_alpha > 0 and args.merge <= 1
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
                  + f" | {gp if gp else st}")

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
