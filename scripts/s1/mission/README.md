# mission/ — S1 dock 任务编排（home↔dock↔推理任务 循环）

2026-11-11 场景编排测试：机器人从 `home` 导航到 `dock` → 跑 traj 推理任务
（模糊完成=跑满 `--rounds` 个推理块）→ 回 `home` → 再去 `dock` → …循环往复。
约束落法：**pi05/fineart 推理引擎常驻**（worker 进程全程存活，引擎只装载
一次）、**导航栈（pns/localization）不关**（编排器不碰服务树）。

## 文件

| 文件 | 角色 |
|---|---|
| `run_dock_mission.py` | 编排器（无 SDK 依赖，纯 stdlib）：点位/地图闸、导航子进程调度、worker stdin 协议、失败升级链 |
| `s1_mission_worker.py` | 推理 worker = `../inference/run_s1_loop_traj.py` 常驻复用副本（引擎路径逐位同源，仅 D0–D4 五处标记修改点，文件头有总注） |

## 用法（先干跑，后 --exec，两次确认纪律）

```bash
cd ~/holy/scripts/s1/mission
PY=~/miniforge3/envs/flash_pyrt311/bin/python

# ① 干跑（⚠ 非零真机：引擎真装载、导航真切底盘控制器，只是不下发运动——
#    前置条件与 --exec 完全相同：WORKING 模式/定位/点位/地图一样不能少）
$PY run_dock_mission.py --cycles 2 --rounds 6

# ② 真跑：急停在 手边！起点在 home 附近
$PY run_dock_mission.py --cycles 2 --rounds 6 --exec

# fineart 线（RTC 权重必带引导窗，经 --worker-extra 透传）
$PY run_dock_mission.py --cycles 2 --rounds 6 --exec \
    --policy fineart_vla_s1 --worker-extra "--execution-horizon 10"

# 常驻导航 worker（可选，2026-11-12）：SDK 会话跨腿保持，每腿省 pns_point
# 子进程冷启动 ~8-10s（用户实测 22.1s/腿=8 冷启动+12.1 行驶+2 收尾）；
# 切控制器/定位/地图闸/可达闸每腿照跑不省。默认关=子进程 proven 路径。
$PY run_dock_mission.py --cycles 2 --rounds 6 --nav-resident
```

前置：WORKING 模式（pns/localization 在跑、`is_localized=True`）；点位
`home`/`dock` 已用 `../nav/pns_point.py record` 录在当前地图（编排器做同源
地图一致性闸，换图拒发）。

## 每循环时序（exec）

```
[worker 启动] robot.init(传感器) → 模型装载 → 预热×3 → @@READY
每 cycle:
  pns_point go dock   （独立子进程：切 POSE_CTRL→可达闸→v2 导航→平台判到位；
                       --nav-resident 时=常驻 s1_nav_worker 吃 "go dock" 行，
                       同闸同逻辑，SDK 会话不拆）
  worker "task"+"go"  （"task"=serve 协议命令；"go" 不是协议命令，是预填给
                       TE.run() 内 exec 确认 input() 的应答行，仅 --exec 才写）
                       → 前奏 start_controller('all')→warm 就位
                       →HOME 重读→TE.run() 消费 --rounds 块→归位→汇总
                       →底盘切回 SWERVE_CHASSIS_POSE_CTRL（让位 pns）→@@TASK_DONE
  pns_point go home
[收尾] worker "quit" → 真三连退出
```

## 设计要点（为什么这样接）

- **引擎常驻**：原版 `TE.run()` 尾部是 SDK 三连+`os._exit`（单任务自灭）。
  worker 副本用两个接缝变可返回：D1 `TE.robot` 挂保活代理（三连空转，
  `RT.robot` 保持真身=致命钩子路径不变）；D2 TE 命名空间 `os` 窝换（仅
  `._exit` 改抛 SystemExit，归位/汇总照走）。引擎循环体零改动。
- **控制器占用**：worker 启动期绝不 `start_controller`（夺 pns→WBC 通路=
  导航下发成功位移≈0）；只在任务前奏拉起、任务后切回 POSE_CTRL 让位。
  导航永远走 pns_point 独立子进程（每次自切自退，proven 路径）。
- **q 键语义**：worker stdin=管道，QuitWatcher 不启用；中断走编排器
  SIGINT 升级链（SIGINT→10s→SIGTERM→5s→SIGKILL），任务中收到 SIGINT=
  臂留原地（同原版 q 语义）。**SIGTERM 起为硬终止**（Python 层不参与：
  无归位、无汇总、无 @@TASK_DONE）——臂停在半程任意位，事后需人工
  确认臂位/手动归位。

## 失败语义

| 信号 | 处置 |
|---|---|
| 导航 rc≠0（重试后） | 中止编排；回程失败时打印人工恢复命令 |
| 任务 @@TASK_DONE rc≠0 | 中止编排（0 正常/1 异常/5 控制器/6 warm/9 SIGINT/10 护栏-星饿停机） |
| 任务超时/worker 死亡 | SIGINT 升级链终止 worker，rc=3/4 |
| worker `TE.aborted`（rc=10） | 护栏/星饿停机，臂留原地——下一任务前奏的 warm 就位可自动恢复臂姿 |

日志：编排器全文 `~/holy/logs/mission_dock_<ts>.log`（worker 输出逐行透传；
worker 自身另有 `loop_traj_<ts>.log`）。
