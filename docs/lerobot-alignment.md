# lerobot 术语对照与命名正规化（2026-10-08）

本文是闭环推理脚本与 [lerobot](/home/galbot/lerobot) 官方实现的**术语对照与命名档案**：
哪些参数改成了 lerobot 标准名、哪些自研参数保留原名及为什么、同名异义的概念如何消歧。
命名正规化是纯更名+遥测，**行为零变化**（工作点 v3=25/0.35/1.5 数值与语义均未动）。

配套：`docs/问题-解决.md`（RTC 执行侧移植的机制详注）、`config/g1.toml [loop]`（参数权威注释）、
`agent_memory/galbot-g1-loop-pace-tuning.md`（调参定案史）。

## 一、改名总表（2026-10-08 生效，旧名不留别名）

| 旧名 | 新名 | lerobot 对应术语与出处 |
|---|---|---|
| `--steps-per-round` / toml `steps_per_round` | `--n-action-steps` / `n_action_steps` | lerobot pi05/ACT 标准术语：每 chunk 执行 K 步再重推理（`policies/pi05/configuration_pi05.py:39` 默认 50）。语义注：消费 chunk 一半就重观测 ≈ async_inference 的 `chunk_size_threshold=0.5`（`async_inference/configs.py:137`），25/50 恰为其默认比例 |
| `--steps-per-cmd` / `steps_per_cmd` | `--steps-per-command` / `steps_per_command` | 无对应物（lerobot 逐 action 下发）；仅展开缩写 |
| `--horizon` / attr `horizon` | `--chunk-size` / `chunk_size` | lerobot `chunk_size`（pi05 默认 50，一致）。旧名与 lerobot RTC 的 `execution_horizon`（前缀引导长度，`policies/rtc/configuration_rtc.py:48` 默认 10）撞名，**两者不是同一概念**。env `FLASH_RT_PI05_ACTION_CHUNK_SIZE` 属 FlashRT 内部，不改 |
| `--no-rtc-hold`（store_true） | `--late-action {hold,stale}`（默认 hold） | 无对应物。"RTC" 字样消歧（见 §四）；`stale`=原 A 脚本行为（旧 chunk 深尾续航），A/B 对照改用 `--late-action stale` |
| `--rtc-hold-max` | `--hold-max`（默认 6） | 同上 |
| `--rtc-ramp` | `--hold-ramp`（默认 0） | 同上。机制≈lerobot 执行侧 blend/crossfade（RTC 本体无此项） |
| `--env-margin` | `--envelope-margin`（默认 0.15） | 无对应物（真机安全层）；消 env=环境 歧义 |
| `--env-abort-n` | `--envelope-abort-n`（默认 5） | 同上 |
| `--no-env-guard` | `--no-envelope-guard` | 同上 |
| 脚本 `run_g1_loop_rtc.py` | `run_g1_loop.py` | 唯一正统闭环入口；原 A/B 差异收敛为 `--late-action` 两态 |
| A 脚本 `run_g1_loop.py` | `scripts/inference/legacy/run_g1_loop_stale.py` | 存档（CLI 旗名保持旧名，mapping 值侧同步新 toml 键）；复现 A 行为优先用 `run_g1_loop.py --late-action stale` |

## 二、不改名清单及理由

`--pace`、`--pace-div`、`--near-div`、`--near-gap`、`--delta-max`、`--speed`、`--rounds`、
`--grip`/`--grip-speed`/`--grip-effort`/`--grip-chg`、`--grip-state-cmd`、`--nav-suspend`、
`--max-excursion`、`--settle`/`--settle-frac`/`--switch-dist`、`--chunk-mode`、`--traj-dt`、
`--cache-frames`、`--ckpt`、`--prompt`、`--exec`、`--config`、`--log-file`。

理由分三类：
1. **自研配速家族（pace/pace-div/near-div/near-gap）**：lerobot 无执行配速概念（刚性位置伺服
   逐条指令配速是我们独有问题；最接近的 lerobot 物是 `utils/action_interpolator.py` 的
   `multiplier`，但那是插值不是配速）。改名会摧毁 BENCHMARKS/记忆的工作点简写
   （"25/0.35/1.5"=n-action-steps/pace/pace-div）。
2. **真机安全/踩坑产物（grip-state-cmd、nav-suspend、envelope 家族、max-excursion）**：
   语义绑本机踩坑实录（爪回读滞后 6.3-8s、导航 GPU 争用、包络护栏范畴错误修复），
   名字已自解释。
3. **`--prompt`**：lerobot 对应词是 `task`（`async_inference/configs.py:122`
   RobotClientConfig.task=语言指令），但 config `[run]` 段被全目录脚本共用，改名扩散超范围；
   语义上 prompt 对 VLM 亦准确。仅在此记录对照。

## 三、pace vs inference_latency vs environment_dt（易混辨析）

| 概念 | 所在 | 本质 |
|---|---|---|
| `inference_latency` | lerobot 服务端（`async_inference/configs.py:59-61`） | **人为设定的最低响应延迟**：`GetActions` 里推理+序列化完成后 `sleep(max(0, inference_latency − elapsed))` 补足（`policy_server.py:256-258`）。是节流/模拟旋钮，不参与动作时间戳 |
| "推理水位"（~0.33-0.35s @v3+nav-suspend） | 本脚本（实测值） | 真实推理+取图+开销的墙钟下限，**卡 `--pace` 下限**（pace<水位→hold 窗回流）。与 inference_latency 概念同源（都在管推理延迟与节拍的关系），但一个是测量一个是设定 |
| `environment_dt`（=1/fps） | lerobot（`configs.py:86-89`） | 客户端动作消费节拍：每 tick 消费 1 条 action 后 `sleep(max(0, dt−elapsed))`（`robot_client.py:489`）。本脚本最近似物 = `pace/n_action_steps`（14ms/步 @v3），但 lerobot 一 tick 只走 1 步、按数据集原速；本脚本一窗消费 K 步且可超速执行——执行配速是 lerobot 没有的问题 |
| `--n-action-steps` | 本脚本 | 对应 lerobot 两条路径的重观测节奏：rollout 栈 `queue_threshold`（`rollout/inference/rtc.py:461`）与 async 栈 `chunk_size_threshold`（队列消费过半即触发新推理） |
| `--steps-per-command`（合步） | 本脚本 | lerobot 无直接对应；`ActionInterpolator`（`multiplier>1` 时按 fps×multiplier 下发插值动作）是最近似物 |

## 四、"RTC" 同名三义消歧

本机历史上 "RTC" 有三个含义，现已从脚本命名中移除该字样：

1. **本脚本（旧 B 脚本）的迟到兜底**：推理迟到时 hold 最后目标单调收敛 + 10ms 轮询早退
   （现 `--late-action hold`）。只搬了 PI 论文/lerobot RTC 的**执行侧调度约定**
   （extension/等待），机制详注见 `docs/问题-解决.md`。现名"late-action hold"。
2. **lerobot RTC**（`policies/rtc/` + `rollout/inference/rtc.py`）：**推理侧** denoise_step
   前缀引导（inference-time inpainting）+ `ActionQueue.merge` 按实测延迟丢新 chunk 前
   `delay=ceil(latency/time_per_step)` 步（`action_queue.py:196-223`、`rtc.py:144-157`）。
   引导层（`modeling_rtc.py:122`）**故意未搬**，五条理由见 `docs/问题-解决.md:141-162`
   （CUDA graph 冲突/autograd 穿整网/delay≈1 轮收益趋零/ckpt 非 RTC 训练/新超参回归风险）。
3. **FlashRT 内部 prefix 对齐**（`run_dataset_execute.py:327` 提及，仅 Thor 路线支持）：
   与前两者均无关。

另：`--hold-ramp`（坡升 0.5→1.0）≈ lerobot 执行侧 blend/crossfade 思路的自研实现，
lerobot RTC 本体无此参数。

## 五、lerobot 无对应物的自研参数（出处声明）

以下参数**不是** lerobot/RTC 论文的内容，勿在文档中归因给它们：

- `--pace`/`--pace-div`：2026-09-24 自研 v4 追踪式配速（`docs/问题-解决.md:58` 明确与 RTC 无关）。
  速度=导程÷(div×窗口)，到达时刻=div×pace，覆盖率=1/div，兼滞后滤波。
  统一经验律：臂速÷数据集原速 = n÷(30×pace×div)。
- `--near-div`/`--near-gap`：近距阻尼（单调几何收敛+滤放置犹豫摆幅，9/30 真机验证）。
- `--grip-state-cmd`：SDK 爪回读滞后 6.3-8s > pick 关键窗 ~7s 的本机踩坑对策（pick 闭环必带）。
- `--nav-suspend`：导航栈 SIGSTOP 搁置（推理 375.3→320.5ms，10-07 实测）。
- envelope 家族：关节包络护栏（比对对象=BASE_ARM+delta 绝对构型，10-07 修范畴错误）。

## 六、日志字样对照（外部笔记/grep 用）

| 旧字样 | 新字样 |
|---|---|
| `[loop-rtc] B 变体` | `[loop]` |
| `RTC hold（默认）` / `A 深尾续航（--no-rtc-hold）` | `迟到兜底: hold` / `迟到兜底: stale（--late-action stale）` |
| `--no-rtc-hold` | `--late-action stale` |
| `RTC hold N 窗` | `hold N 窗` |

## 七、遥测：LatencyTracker（2026-10-08 接入）

`run_g1_loop.py` 的每轮遥测（推理/取图/单指令/整轮/配速窗口）改用 `LatencyTracker`
类，结构参照 lerobot `policies/rtc/latency_tracker.py:24-72`（Apache-2.0），**两处有意偏离**：

1. `maxlen=None`（无界）：lerobot 默认 `maxlen=100` 会截断 200 轮跑的统计窗口，p50/p95 必变；
2. 保持 float64：去掉 lerobot `latency_tracker.py:67` 的 `np.float32` cast，保证与旧
   `np.percentile` 路径打印逐字节一致（`np.quantile(x,0.95)`≡`np.percentile(x,95)`，numpy 同一实现）。

输出格式不变（`p50 %.1f | p95 %.1f | max %.1f ms`），仅内部容器统一。

## 八、config 附注

- **`[loop]` 节名锁定**：`run_g1_replay.py` 跨节读 `[loop].max_excursion/.switch_dist`，
  节名不可改（改=静默回落 BUILTIN）。
- **旧键名硬报错**：`g1_config.load()` 现对 `[loop]` 内旧键（`steps_per_round`/`steps_per_cmd`）
  直接 SystemExit 并提示新键名；未认键打印告警——堵三态回填（CLI>config>BUILTIN）静默回落。
- BUILTIN 镜像已同步 v3（0.35/1.5）；`[loop].catch_timeout` 为死配置（无消费方，
  replay 读的是 `[replay].catch_timeout`），留作历史遗留注记。
