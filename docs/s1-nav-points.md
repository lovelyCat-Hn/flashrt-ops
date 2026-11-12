# S1 点位录制与标签导航

脚本：`scripts/s1/nav/pns_point.py`
适用：固定点位往返（工位、货架、充电位、观察位）的录制与复用。
配套：`docs/s1-nav-relative.md`（相对位移执行器 `pns_move_relative.py`）

---

## 1. 用途

把"当前站在哪里"存成带名字的标签，之后按标签直接导航，不再依赖"前进 1 m / 后退 1 m"这类相对量。

```
录制：机器人开到目标位置 → python pns_point.py record kitchen
复用：任意位置 → python pns_point.py go kitchen
```

点位为 map 系绝对坐标（`x, y, yaw`），与当前朝向无关。录制一次，长期可用；换图后自动失效保护。

---

## 2. 快速开始

```bash
cd ~/holy/scripts/s1/nav
PY=~/miniforge3/envs/flash_pyrt311/bin/python

# 1) 把机器人开到点位 A，录制
$PY pns_point.py record home --note "起点"

# 2) 开到点位 B，录制
$PY pns_point.py record shelf --note "货架前"

# 3) 查看
$PY pns_point.py list

# 4) 从任意位置回到 home，先预检
$PY pns_point.py go home --dry-run

# 5) 确认后执行
$PY pns_point.py go home --speed 0.15
```

`record` 只读位姿，不切控制器、不下发运动，可在任意已定位时刻执行；`go` 才涉及控制器切换与运动。

---

## 3. 子命令与参数

### 3.1 record

```bash
$PY pns_point.py record LABEL [--note TEXT] [--force]
```

| 参数 | 说明 |
|---|---|
| `LABEL` | 标签名，建议 ASCII、无空格（中文可用，`list` 显示对齐会略乱） |
| `--note` | 备注，写入点位文件 |
| `--force` | 同名标签已存在时覆盖（默认拒绝，返回码 6） |

录制内容：`x, y, z=0, yaw_deg`、创建时间、备注、地图标识。`z` 恒为 0，S1 为平面底盘。

### 3.2 go

```bash
$PY pns_point.py go LABEL [--speed V] [--omni] [--dry-run] [--timeout S] [--ignore-map]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--speed V` | 0.2 | 本任务速度上限 m/s，x/y 同值，角速度取 1.5V |
| `--omni` | 关 | 提高全向规划优先级；车尾为雷达盲区，倒车方向慎用 |
| `--dry-run` | 关 | 只做可达性查询，不下发运动 |
| `--timeout S` | 120 | 单次任务超时 |
| `--ignore-map` | 关 | 点位地图与当前地图不一致时仍强制执行 |

执行流程：`robot.init` → `nav.init` → 切 `SWERVE_CHASSIS_POSE_CTRL` → 校验 `is_localized` →
`check_path_reachability` → `navigate_to_goal_v2(goal, pose_frame='map')` → 轮询到位。

到位判据与 `pns_move_relative.py` 一致：`check_goal_arrival()` 为真，或位移平台连续 1.6 s
变化小于 0.004 m 且目标残差 ≤ 0.06 m（非阻塞模式下前者不刷新，故以后者为主）。

### 3.3 list

```bash
$PY pns_point.py list
```

离线可用，不导入 SDK、不连接 pns。输出标签、坐标、朝向、创建时间、备注，以及点位文件
当前的记录地图。

### 3.4 rm

```bash
$PY pns_point.py rm LABEL [LABEL ...]
```

删除前打印被删点的坐标。标签不存在返回码 6，其余标签照常处理。

---

## 4. 点位文件

默认路径：`~/holy/data/s1_nav_points.json`，可用全局参数 `--file PATH` 指定其他位置。

```json
{
  "schema": 1,
  "map": {
    "name": "galbot_pre_map-<hash>",
    "dir": "/var/maps/galbot_pre_map-<hash>",
    "pcd_mtime": 1794374791,
    "pcd_size": 22484986
  },
  "points": {
    "home": {
      "x": 0.217, "y": -0.001, "z": 0.0, "yaw_deg": -0.31,
      "created": "2026-11-11T16:03:54",
      "note": "起点",
      "map": { "name": "galbot_pre_map-<hash>", "pcd_mtime": 1794374791, "...": "..." }
    }
  },
  "updated": "2026-11-11T16:05:26"
}
```

写入方式为 `json.dump` 到 `.tmp` 后 `os.replace`，避免中断留下半个文件。文件损坏时
所有子命令返回码 7 并打印原因，不覆盖原文件。

---

## 5. 地图一致性

点位坐标只在录制时的地图上有效。`record` 会把当前地图标识写入点位：

- 地图名：`os.path.realpath('/var/maps/cur')` 的目录名（`map_manager_node` 切模式时会生成
  `galbot_pre_map-<hash>`）；
- 点云指纹：`cur/global_cloud_cleaned.pcd` 的 mtime 与文件大小。

`go` 前逐项比对，不一致时：

```
[map] 警告：点位录制时的地图与当前不一致（地图目录 A → B）
[map] 目标坐标可能已失效，确认无碍后加 --ignore-map 强制执行
```

返回码 8，不执行。确知新旧地图几何一致（例如仅重新打包 cur 目录、点云未变）时可加
`--ignore-map`。重新建图后建议重录全部点位，不要依赖该参数。

---

## 6. 实测记录（2026-11-11 16:03–16:05）

```
record __origin   → x=+0.217 y=-0.001 yaw=-0.31°
前进 1 m（pns_move_relative.py 1.0 --speed 0.15）→ 净位移 0.970 m，8.9 s
record __pointA   → x=+1.185 y=-0.004 yaw=-0.36°
go __origin --speed 0.15
  → 下发 (True,'SUCCESS')，净位移 0.940 m，用时 9.2 s
  → 目标残差 0.030 m，yaw 误差 0.04°，平台判据 stopped=True，rc=0
```

录制与回访闭环成立，残差水平与相对位移脚本一致（pns 的 `arrival_pos_err` 为 0.05 m）。
测试点位已用 `rm` 清理，文件保留空结构。

---

## 7. 与 pns_move_relative.py 的分工

| 场景 | 脚本 | 目标表达 |
|---|---|---|
| 相对位移（前进/倒车 N 米） | `pns_move_relative.py` | base_link 相对量 |
| 固定点位往返 | `pns_point.py go` | map 系绝对坐标 + 朝向 |
| 临时绝对目标（命令行给坐标） | `pns_move_relative.py --pos X Y --yaw D` | map 系绝对坐标 |

两者共用同一套初始化顺序、到位判据与关闭纪律，可互相替代排查问题。

---

## 8. 已知限制

1. **终点朝向固定。** `go` 使用录制时的 `yaw` 作为终点朝向，不提供"仅到位置不管朝向"的选项。
   若只需位置，录点时把车头摆到期望朝向即可。
2. **不做点位插值或路径拼点。** 一次 `go` 只走一个标签；多点巡游需外层脚本循环调用。
3. **无并发写保护。** 两个进程同时 `record` 可能互相覆盖，单人操作不受影响。
4. **点位是底盘中心位姿。** 机器人以 `base_link` 原点到达，不是车头或机械臂末端；贴边作业时
   需考虑车体外廓。
5. **依赖定位有效性。** `record` 与 `go` 都要求 `is_localized()` 为真；定位丢失时拒绝执行，
   返回码 2。位置是否正确仍需人工确认一次（评分高不等于坐标对）。
6. **`--omni` 的盲区风险** 与相对位移脚本相同：车尾 180° 无雷达覆盖，倒车方向全向平移
   可能撞到动态障碍，详见 `docs/s1-nav-relative.md` 第 10 节。

---

## 9. 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功（录制完成 / 到位） |
| 2 | 初始化或定位失败，或参数缺失 |
| 3 | pns 判不可达，未下发 |
| 4 | 超时或未确认到位 |
| 5 | 控制器切换失败 |
| 6 | 标签不存在，或录制时同名标签已存在 |
| 7 | 点位文件损坏或读取失败 |
| 8 | 点位与当前地图不一致，未下发 |

共同纪律：结束统一 `stop_navigation` → 优雅三连 → `os._exit`；`faulthandler` 300 s 兜底。
