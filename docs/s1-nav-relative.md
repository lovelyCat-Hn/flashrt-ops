# S1 底盘相对导航执行器使用说明

适用对象：`scripts/s1/nav/pns_move_relative.py`
适用范围：S1 底盘（swerve_chassis）在已完成初始定位的地图上执行相对位移或绝对目标导航
相关：固定点位往返请用 `scripts/s1/nav/pns_point.py`，说明见 `docs/s1-nav-points.md`

---

## 1. 功能概述

`pns_move_relative.py` 以 SDK 官方例程 `examples/s1/python/galbot_navigation/navigate_to_goal_v2.py`
的顺序调用 pns 导航服务，完成以下工作：

1. 初始化 `GalbotRobot` 与 `GalbotNavigation`；
2. 切换底盘控制器至 `SWERVE_CHASSIS_POSE_CTRL` 并校验返回值；
3. 等待定位状态就绪；
4. 向 pns 查询目标可达性；
5. 下发相对目标（base_link 坐标系）或绝对目标（map 坐标系）；
6. 以固定周期读取位姿，输出位移进度，按平台判据判定到位；
7. 停止导航、按优雅三连释放 SDK 资源。

相对目标以车体当前朝向为基准，正数朝车头方向，负数朝车尾方向，终点朝向保持为起点朝向。
调用方只需给出位移量与速度上限，路径形状、是否原地转向、是否中途重规划均由 pns 决定。

---

## 2. 运行环境

| 项目 | 取值 |
|---|---|
| 解释器 | `/home/galbot/miniforge3/envs/flash_pyrt311/bin/python`（Python 3.11） |
| SDK 路径 | `PYTHONPATH=/userdata/update/manual_update/lib` |
| SDK 版本 | 1.10.0 |
| 前置条件 | 机器人处于工作模式，pns 与 localization 两类服务已启动，`is_localized=True` |
| 建议工作目录 | `/home/galbot/holy/scripts/s1/nav` |

统一运行前缀：

```bash
cd /home/galbot/holy/scripts/s1/nav
export PYTHONPATH=/userdata/update/manual_update/lib
PY=/home/galbot/miniforge3/envs/flash_pyrt311/bin/python
```

---

## 3. 参数

### 3.1 位置参数

| 参数 | 含义 | 缺省 |
|---|---|---|
| `dist` | 相对位移，单位 m；正值朝车头，负值朝车尾 | `1.0` |

### 3.2 可选参数

| 参数 | 含义 | 缺省 |
|---|---|---|
| `--omni` | 提高全向规划优先级（`omni_plan=True` → `try_omni_plan_first: 1`）。关闭时规划器仍可能在全向直达探测通过后自行直线平移，转身与否以其裁决为准 | 关闭 |
| `--speed V` | 本任务速度上限，单位 m/s，同时作用于 x、y 轴；角速度上限取 `1.5V` | `0.2` |
| `--acc A` | 全局加速度上限，单位 m/s²，需与 `--jerk` 配套 | 不下发 |
| `--jerk J` | 全局加加速度上限，单位 m/s³ | 不下发 |
| `--pos X Y` | 改用 map 系绝对目标，`dist` 失效 | 不使用 |
| `--yaw DEG` | 绝对目标模式下的终点朝向，单位度 | 保持当前朝向 |
| `--dry-run` | 仅查询可达性，不下发运动 | 关闭 |
| `--timeout S` | 单次任务超时，单位 s | `40` |

`--acc` 与 `--jerk` 调用 `set_navigation_kinematics_limits`，属于运行态全局修改，对所有下发方生效，
pns 重启后恢复配置文件取值。只调整速度时无需使用这两个参数。

### 3.3 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 到位 |
| `2` | 初始化失败、未定位或 `nav.init` 失败 |
| `3` | pns 判定目标不可达，未下发 |
| `4` | 超时或未确认到位 |
| `5` | 控制器切换失败 |

---

## 4. 典型用法

### 4.1 前置检查

```bash
$PY -u pns_move_relative.py 1.0 --dry-run
```

输出包含起点位姿、目标换算后的 map 坐标、`check_path_reachability` 结果。`--dry-run` 不产生运动，
适合在执行前确认地图与定位状态。

### 4.2 正前方 1 m

```bash
$PY -u pns_move_relative.py 1.0 --omni --speed 0.2
```

### 4.3 低速短距（0.15 m/s，0.3 m）

```bash
$PY -u pns_move_relative.py 0.3 --omni --speed 0.15
```

### 4.4 后退 1 m

```bash
$PY -u pns_move_relative.py -1.0
```

**转身与否由 pns 规划器自动裁决，`--omni` 不是"转/不转"开关。** 日志证据（2026-11-11）：

- 13:56 `omni_plan=False`：`dir: Heading`，先掉头约 190°、正向行进、再转回，1 m 用时 14.3 s；
- 15:35 `omni_plan=False`：规划器探测 `probeOmniDirectGoal` 通过后
  `segment_plan: prefer omni-direct goal target`，`dir: Omni`，旋转段 0 点，直接平移，5.6 s；
- 14:03 `omni_plan=True`：`try_omni_plan_first: 1`，全向直达，6.2 s；同日另有一次
  probe 判 `omni-direct collides`（路点距障碍 0.613 m < 门槛 0.65 m）而回退的记录。

即 `omni_plan=False` 是"不优先全向"，规划器仍会在 probe 通过时自行选择直达；
`omni_plan=True` 提高全向方案优先级（附带 `try_omni_plan_min_obstacle_distance: 0.65`
的安全门槛），但极端环境仍会回退。判断实际行为的依据是 `traj info` 的旋转段点数
（0 = 未转身），而非开关本身。

### 4.5 绝对目标

```bash
$PY -u pns_move_relative.py --pos 1.20 -0.30 --yaw 90
```

### 4.6 柔和加减速

```bash
$PY -u pns_move_relative.py 1.0 --omni --speed 0.18 --acc 0.25 --jerk 0.8
```

---

## 5. 输出说明

单次执行的标准输出分三段。

**初始化段**

```
[ctrl] switch_controller(SWERVE_CHASSIS_POSE_CTRL) → ControlStatus.SUCCESS
[loc] is_localized → True
[pose] 起点 x=-0.002 y=+0.006 yaw=+0.05°
[goal] 相对目标 +1.000 m(base_link) → map 约 (+0.998, +0.007)
[reach] check_path_reachability → True
```

**执行段**：每秒一行，给出相对起点的平面位移与当前位姿。

```
  [ 2.4s] 位移 0.294 m  x=+0.291 y=+0.012 yaw=-0.01°
```

**收尾段**

```
[pose] 终点 x=+0.467 y=+0.012 yaw=-0.03°
[pose] 净位移 0.470 m  用时 5.7 s  平均 0.083 m/s
[nav] 目标残差 0.030 m（到位判据 ≤ 0.06）  平台判据 stopped=True
[nav] 到位
[exit] rc=0
```

---

## 6. 实现要点

### 6.1 只切控制器，不调用 start_controller

官方例程的初始化只有三步：`robot.init()`、`nav.init()`、`switch_controller(SWERVE_CHASSIS_POSE_CTRL)`。
额外调用 `start_controller` 或 `acquire_controller` 会使 pns 失去经由 WBC 执行轨迹的通路，
表现为轨迹正常下发、`dispatch_result=success`、FSM 进入 `EXEC_TRAJ`，但底盘位移接近 0。
2026-11-11 13:26 的失败即由此产生。

### 6.2 非阻塞下发配合主动监控

`navigate_to_goal_v2` 以 `is_blocking=False` 下发，脚本自行按 0.4 s 周期读取 `get_current_pose`。
该模式下 `check_goal_arrival()` 不随任务完成刷新，返回值恒为 `False`，因此不作为唯一判据。

到位判据为二选一：

- `check_goal_arrival()` 返回 `True`；
- 位移平台连续 1.6 s 变化小于 0.004 m，且目标残差不超过 0.06 m。

`get_navigation_status()` 在本机型会抛出异常并中断任务，脚本未使用该接口。

### 6.3 系统残差

pns 的到位阈值为 `arrive err config(x/y/yaw): 0.05, 0.05, 0.12`，底盘停在目标前约 2.5–3 cm 属于正常表现。
实测三组：目标 1.0 m 实到 0.973 m，目标 0.5 m 实到 0.470 m，目标 −0.5 m 实到 −0.469 m。

需要更高精度时调用 `set_navigation_arrival_threshold(np.array([0.03, 0.03, 0.03]))`，
有效范围 0.03–2.0。

---

## 7. 速度与平滑度

速度受三层限制，逐层收紧：

| 层 | 位置 | 取值 | 生效方式 |
|---|---|---|---|
| 控制层 | pns 配置 `max ctrl vel` | 0.5 / 0.5 / 0.7 | pns 启动时读取 |
| 规划层 | pns 配置 `vel_upper` | 0.3 / 0.3 / 0.5 | pns 启动时读取 |
| 任务层 | `max_vel` 入参（`--speed`） | 由调用方指定 | 单次任务 |

规划层的 0.3 m/s 是硬上限。任务层取值超过 0.3 不产生效果，需要在配置文件中修改 `vel_upper` 并重启 pns。
低于 0.3 时任务层直接生效，pns 日志可见对应记录：

```
I20261111 14:06:59.433075 6070 trajectory_planner.cpp:3226] velLimitUpper: 0.12 0.12 0.18
```

加速度与加加速度只提供全局接口，无逐任务参数：

```python
nav.set_navigation_kinematics_limits(vel_limit, acc_limit, jerk_limit)
# vel  有效范围 [0.05, 1.5]
# acc  有效范围 [0.05, 6.0]
# jerk 有效范围 [0.05, 12.0]
```

2026-11-11 的 A/B 对比（0.5 m 行程，acc 2.0 与 0.3 各一次）未观测到明显差异：
两次用时分别为 3.41 s 与 3.52 s，平均速度 0.140 m/s 与 0.134 m/s。
原因是行程过短，加速段仅占 0.3–0.4 s，且 20 Hz 位姿采样的定位抖动（约 ±0.05 m/s，
换算加速度约 ±1 m/s²）大于待测差值。加减速参数的有效性需在 1.5 m 以上行程验证。

起步突兀感来自「静止至巡航」的过渡。同时降低速度与加速度才产生明显区别，
建议组合为 `--speed 0.18 --acc 0.25 --jerk 0.8`。

---

## 8. pns 侧核对方法

脚本输出之外，可在 pns 日志中核对执行细节：

```bash
L=/userdata/log/service_navigation_plan/service_navigation_plan.INFO

# 轨迹结构：rotation / translation / rotation 三段及各自点数
grep -a "traj info" $L | tail -3

# 到位结果与残差
grep -a "robot arrive" $L | tail -3

# 规划限值
grep -aE "velLimitUpper|accLimitUpper" $L | tail -6

# 避障是否参与
grep -a "Motion plan with collision" $L | tail -3

# 逐秒位姿（x y yaw）
grep -a "current pose" $L | tail -10
```

`traj info` 中 `segments` 三个数字为三段的轨迹点数。第一个数字为 0 表示未发生原地转向，
多于 0 表示实际执行了原地转向。注意：未转身不等于 `--omni` 生效——规划器在
`omni_plan=False` 下也可能经 `probeOmniDirectGoal` 探测通过后自行选择全向直达，
开关只决定"是否优先"而非"是否转身"。

---

## 9. 安全约束

1. 执行前确认急停按钮在手边。模式切换与控制器接管瞬间，手臂可能失力矩坠落（2026-10-27 事件）。
2. 先执行 `--dry-run` 确认可达性，再下发运动。
3. 首次在某位置使用时，位移量从 0.3 m 起，确认方向与预期一致后再放大。
4. 脚本内置看门狗：任务超时（`--timeout`）或位移平台判定后自动停止，并调用 `stop_navigation`。
5. 退出统一走 `stop_navigation` → `request_shutdown` → `wait_for_shutdown` → `destroy`，
   最后以 `os._exit` 结束进程。S1 在解释器正常退出阶段存在 segfault，故不使用常规返回。
6. 脚本注册 `faulthandler.dump_traceback_later(120, exit=True)`，挂死时自动输出全部线程栈并退出。
7. **避障无独立开关，默认开启，分两层**。接口层 `enable_collision_check` 在
   `navigate_to_goal`、`navigate_to_goal_v2`、`navigate_with_velocity` 三个签名中默认值均为 `True`；
   服务层由 `local_plan_config.toml` 强制：`collision_check_plan_path = true`、
   `collision_check_period = 0.01`（每 10 ms 检查一次）、`check_collision_radius = 2.5`、
   `check_collision_height = 1.2`、`safe_distance = 0.55`、`start_unchecked_radius = 0.4`。
   本脚本固定传入 `True`，不提供关闭通道；服务层的路径碰撞检查也不受调用方参数影响。

---

## 10. 已知问题与注意事项

1. **`check_goal_arrival()` 在非阻塞模式下不刷新**，恒返回 `False`。判定到位依赖平台判据，
   日志中 `[nav] check_goal_arrival → False （非阻塞模式下可能不刷新）` 属预期输出。
2. **`dir(GalbotNavigation)` 返回空列表**，接口为动态绑定。查询可用接口应阅读类型存根
   `/userdata/update/manual_update/lib/galbot_sdk/s1/__init__.pyi`。
3. **`dump_navigation_configs()` 可能返回 `(False, 'WAIT_INITIALIZED')`**，
   需 pns 完成一次任务后才可读取。
4. **`omni_plan=True` 存在后方盲区风险**。出处见
   `/data/galbot/config/default/service_navigation_plan/galbot_s1/local_plan_config.toml` 中的注释：

   ```
   omni_plan = false   # 全向规划，建议在全局静态点云环境下使用，动态地图因雷达在机器人后方为盲区，
                       # 扫描不到后方障碍物，可能全向运动（尤其后退）会发生碰撞
   ```

   即 `pns_nav_session.py` 中「S1 铁律 omni_plan=False」的来历：动态障碍物进入车尾 180° 盲区时，
   全向（尤其倒车）规划不会减速也不会绕行。2026-11-11 实测 `omni_plan=True` 在空旷区域可用
   （1 m 平移 6.2 s、残差 0.025 m、无原地转向），但该结论仅在后方无动态障碍时成立。
   现场有人走动、车尾朝向通道口时，应保持默认 `omni_plan=False`（RTR 模板，先转向再平移，
   使障碍物落在雷达视场内）。取 `True` 的前提是确认后方盲区无障碍且环境为静态。
5. **夹爪状态不影响导航**。pns 配置 `end_effector` 中 `right_arm="None"`，
   且 `ignore_body_state` 为 `true` 时全身状态被忽略，双臂控制器处于 `FAULT` 不阻断底盘导航。
6. **`ignore_body_state` 修改需重启生效**。运行态配置文件为
   `/data/galbot/config/default/service_navigation_plan/galbot_s1/aphropm_pns_server_config.toml`，
   实际指向 `/userdata/update/manual_update/config/default/` 下的同名文件。修改后切换模式或重启整机，
   启动日志 `pns_core.cpp:997] ignore_body_state: 1` 可确认生效。

---

## 11. 版本记录

| 日期 | 内容 |
|---|---|
| 2026-11-11 | 首版。脚本纳入 `scripts/s1/nav/`，完成往返运动验证（前进 0.5 m / 后退 0.5 m，退出码 0，残差 0.030 m） |
