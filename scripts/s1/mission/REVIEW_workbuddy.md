# mission/ 评审注记（WorkBuddy 侧，2026-11-12 09:40）

本文件由 WorkBuddy 侧 agent 撰写，供 vs code 侧参考。结论先行：**编排协议自洽，
未发现会导致卡死或失控的缺陷**，以下为核对依据与三条建议。

## 一、已核实的正确性（无需改动）

1. **exec 确认行数对齐**。`run_dock_mission.py run_task()` 非 DRY 时预填
   `"task\ngo\n"`，而 `common/loop/traj_engine.py:461` 的 `args.do_exec` 分支
   只有一次 `input("急停就绪后回车开始...")`，`common/bodies/s1.py` 的
   `s1_warm` 无 input。一行 `go` 恰好消费一次确认；前奏失败时残留的 `go`
   由 `s1_mission_worker.py serve()` 的杂音忽略分支兜住。无死锁风险。
2. **kill 升级链顺序合理**（SIGINT 10s → SIGTERM 5s → SIGKILL）。
3. **点位/地图闸与 pns_point.py 同源口径**（cur realpath 目录名 + cleaned
   点云 mtime/size），两侧判定一致，不会出现一侧放行一侧拒绝。

## 二、建议（按优先级）

1. **README 时序图里的 `task`+`go` 表述易误读**。serve() 协议只认
   `task`/`quit`，`go` 不是协议命令，而是预填给 traj_engine 内
   `input()` 的确认行。建议在 README 该行加一句注（"go=TE.run 确认行，
   serve 忽略"），避免后来者按两条协议命令实现第二套编排器。
2. **DRY 干跑并非零真机**：导航走 `pns_point.py --dry-run`（内部会真切
   `SWERVE_CHASSIS_POSE_CTRL`），worker 会真装载引擎。建议 README 明示
   "干跑 = 真装载彩排"，前置条件（WORKING 模式、pns 在跑、急停在手）
   与 --exec 相同。
3. **SIGTERM 兜底无善后**：worker 收 SIGTERM 是 Python 硬终止，不触发
   KeyboardInterrupt、不归位、无 @@TASK_DONE。作为最后兜底可接受，但
   建议 README 标注"SIGTERM/SIGKILL 后臂停在半程，需人工确认臂位"。

## 三、环境口径说明（现状可用，勿"统一"）

worker 用 `/data/galbot/lib`（inference 原版口径），导航子进程
`pns_point.py` 自补 `/userdata/update/manual_update/lib`。两个路径下都有
`galbot_sdk`，各自 proven、互不冲突。统一口径时注意导航侧 SDK 版本以
manual_update 为准（运行时 pns 读的就是这份配置树）。

## 四、两版编排器的分工建议

| 文件 | 定位 |
|---|---|
| `run_dock_loop_wb.py`（WorkBuddy 侧） | **零真机协议级冒烟**：假 worker + 假导航，不装引擎、不切控制器，验证握手/循环/rc/中断链。`--smoke --cycles 2` 已通过（2026-11-12 09:34）。 |
| `run_dock_mission.py`（vs code 侧） | **真装载彩排与真机执行**：DRY=引擎真装载+导航可达闸，--exec=真跑。 |

建议真机首跑顺序：`run_dock_loop_wb.py --smoke`（已完成）→
`run_dock_mission.py --cycles 1 --rounds 2`（DRY 彩排）→
`--cycles 1 --rounds 2 --exec`（小步真跑）→ 完整循环。
