# config/ 目录导读（base + 机型继承，一页纸）

**这是什么**：部署运行参数的唯一正式存放处。三个文件：`base.toml`（机型无关
红线/语义默认）+ `g1.toml` / `s1.toml`（机型配置，经 `include` 继承 base）。
TOML 格式、全中文注释、注释即定案依据，本 README 是导读。
**给谁读**：换机部署、调工作点、排查"参数到底从哪来的"的人。日常跑脚本不用读它。

## 1. 文件分工与继承

| 文件 | 放什么 |
|---|---|
| `base.toml` | 跨机型红线（`cache_frames=1`、`tier="bf16"`）+ 语义默认（`chunk_mode`/`traj_dt`/`gripper.enabled`/`ctrl_hz`），带判死依据全文 |
| `g1.toml` | G1 全部工作点（pace/near 家族、speeds、ckpt/prompt/pose_file、[lerobot] 档） |
| `s1.toml` | S1 继承骨架（执行脚本未建，建时按 g1 样式落键，复用 g1_config.load()） |

机型文件头写 `include = "base.toml"`（相对本文件目录解析）；加载器递归解析、
**按节深合并**：机型文件同名键覆盖 base，机型节里没写的键从 base 补齐。

## 2. 加载机制（一条链）

```
入口脚本（run_g1_loop.py 等 10 个；legacy/ 下存档脚本不在此列）
  └─ import g1_config（scripts/g1/inference/g1_config.py，共享配置模块）
       └─ load() → 默认读 DEFAULT_PATH = /home/galbot/holy/config/g1.toml（绝对路径）
            └─ _load_toml_resolved()：解析 include → base+机型深合并
                 机型文件缺失不崩 → 退回模块内 BUILTIN 内置默认（与 toml 同源兜底）
                 ⚠ base 缺失 = 硬报错（红线静默丢失不可接受）
```

- 脚本启动会打印 `[config] ... 覆盖: ...`——即本文件实际生效了哪些键，一眼可核对
- 个别脚本支持 `--config <路径>` 指向别的 toml（A/B 实验用），日常不传

## 3. 优先级（四态，永久规则）

**命令行显式传参 > 机型文件 > base.toml > 脚本内置默认（BUILTIN）**

- CLI 支持配置的参数一律 `default=None`，不传才落配置
- ⚠ **`--exec`（真实运动）永不进配置**，每次命令行显式给——安全红线
- 布尔项显式关用 `--no-xxx` 旗标（如 `--no-grip`）

## 4. 机器本地键（拉库/换机先核对这两处）

| 键 | 说明 |
|---|---|
| `[run].ckpt` | 部署目录，**各机各不同**（g1.toml 内有逐机清单与切回存档） |
| `[warmup].pose_file` | 预热目标位姿 json，随 ckpt 目录走（现外置于部署目录 episode_start_task0.json） |

git pull 拉到别人的覆盖值时，改回本机路径再跑（启动打印会暴露张冠李戴）。

## 5. 怎么改工作点

1. 找到对应小节（速查表见 §7），键旁注释写着**定案依据**（哪天、几连跑、什么判死）
2. 改值保存即生效（下次运行读取），**勿删注释**——它是回滚和"为什么是这个数"的唯一记录
3. 提速类参数单变量改（如 `--pace-div 1.55` 先 CLI 试，稳了再落文件）
4. 改完同步 `scripts/g1/inference/g1_config.py` 的 BUILTIN（见 §6）
5. 动 base.toml 的红线键（cache_frames/tier）前先读键旁判死注释，改值须同步 BUILTIN
   并在机型文件写同名键做本机覆盖更安全（base 是共用的）

## 6. BUILTIN 同源义务

`g1_config.py` 的 `BUILTIN` dict 与 g1.toml **逐键同值**：toml 是运行时真相，
BUILTIN 是 toml 缺失/损坏时的兜底。**两边一起改**，改完可用启动打印核对两边一致。
BUILTIN 覆盖面=完整机型档（含 base 的键），机型文件缺失时仍得到完整有效配置。
单测覆盖：BUILTIN 全键、优先级各路径、真实 toml 解析（含 include 链）。

## 7. 谁读哪些小节（速查）

| 脚本（scripts/g1/） | 读的小节 |
|---|---|
| `inference/run_g1_execute.py` | `[run]` `[execute]` `[gripper]` |
| `inference/run_g1_loop.py` | `[run]` `[loop]` `[gripper]` |
| `inference/run_g1_inference.py` | `[run]` `[inference]` |
| `inference/run_g1_replay.py` | `[run]` `[replay]`（另跨读 `[loop].max_excursion/.switch_dist`——[loop] 节名不可改） |
| `body/g1_pose_warmup.py` | `[run]` `[warmup]` `[gripper]` |
| `body/g1_grip.py` | `[run]` `[gripper]` |

（`[run].prompt` 三条原句=训练 tasks.parquet 原文，换 ckpt 必须同步换句。）

## 8. 排错

- **"参数没生效"**：看启动 `[config]` 打印——没列出 = toml 没覆盖它，可能是 CLI 显式传了、
  或键名拼错（tomllib 静默忽略未知键）
- **启动报 "[loop].xxx 已改名"**：`steps_per_round`/`steps_per_cmd` 已更名
  `n_action_steps`/`steps_per_command`（2026-10-08，对照表 docs/lerobot-alignment.md）
  ——load() 硬报错防静默回落，按提示改 toml 键名即可
- **启动报 "include 的基座文件不存在"**：机型文件头 `include` 指向的文件被删/改名
  ——把 base.toml 补回来，别把 include 行删了绕过（红线会静默丢）
- **toml 文件被删/路径错**：机型文件缺失不崩、静默走 BUILTIN——行为变了先确认文件在不在
- **py3.8 系统解析报 tomllib 找不到**：toml 必须用 flash_pyrt311 环境的脚本跑
  （`~/holy/run.sh` 已封正确解释器），别用系统 python3 直接调
- **各机参数互串**：几乎都是 `[run].ckpt` 没改回本机路径（§4）
