# G1 部署日常命令速查（echo 机）

> 只放"怎么跑"。原理、排障、实测数据见 `DEPLOY.md` / `USAGE.md` / `BENCHMARKS.md`。
> 参数已集中到 **`config/g1.toml`**，日常命令都很短；临时调整用 CLI 覆盖
> （优先级：**CLI 显式 > config 文件 > 内置默认**，脚本启动会打印 config 覆盖了哪些项）。

所有命令统一用启动器 `~/holy/run.sh`（自带 SDK 库路径 + flash_pyrt311 环境，
新开 shell 直接用，无需定义变量；参数原样透传）。

**安全**：所有会动的脚本执行前有回车确认；运动等待中随时按 `q` 退出
（SDK 阻塞中 Ctrl-C 无效，q 走独立线程）；紧急停止拍物理急停。

---

## 1. 日常运行（按顺序）

### ① 预热 → 数据集工作姿态（零位 → 臂+头 → 躯干/腿 → 夹爪闭合并行）

```bash
~/holy/run.sh ~/holy/scripts/inference/g1_pose_warmup.py
```

- 2026-09-23 真机全绿：腿 5 关节 set_joint_positions 直接收（SUCCESS 逐关节到位）；
  state.mean 距自然站高仅 13-16°，动作温和
- 夹爪闭合与段① 并行发出（反馈滞后 ~6.3s 被动作期重叠掉，末端统一核对）
- 临时项：`--skip-zero`（已在工作位附近） / `--skip-leg` / `--no-grip` / `--speed 0.1`

**达标信号**：`双臂维(0-7,8-15) 外 0 维 → ✅ 达标`；夹爪 0%±10；腿最大偏差 <0.05 rad。
（leg/head 窄维亮灯属传感器噪声，已知无害）

⚠ **每次挪动底盘后必须重跑预热**（2026-09-23 实证：挪机器人到位后臂下垂漂移
0.3~0.5 rad、4 个臂维出 [-1,1] 域，模型即使场景对齐也继续输出均值）。
挪动后臂偏得小，直接 `--skip-zero` 走直控回位即可。

### ② 只读推理（不动机器人，验语义/延迟）

```bash
~/holy/run.sh ~/holy/scripts/inference/run_g1_inference.py
```

**看什么**：chunk (10,16) 打印；推理 p50 应 ~240-260 ms（fixed 模式恒定）；
臂数值呈 -1.2~1.6 结构化=模型锚定到场景了；全员贴 0=还在输出均值（场景不对，见 §4）。

### ③ 3 步 smoke（真实驱动，含夹爪）

```bash
~/holy/run.sh ~/holy/scripts/inference/run_g1_execute.py --exec
```

- 每步限幅 ±0.05 rad + 0.15 rad/s 限速，说破天也只挪这些
- `--no-grip` 可关夹爪；`--steps 1` 更保守

**达标信号**：全部 `SUCCESS`、回读偏差 <20 mrad（实测 1.9）；
夹爪终态读数 ≈ 指令宽度（终读等 8s 消化反馈滞后）。

### ④ 闭环 receding-horizon（连续任务）

```bash
~/holy/run.sh ~/holy/scripts/inference/run_g1_loop.py --exec
```

- config 内置 2026-09-23 真机 5/5 全绿组合：
  `--speed 0.25 --settle-frac 0.5 --steps-per-cmd 3 --switch-dist 0.06 --chunk-mode track`
- 漂移护栏：任一关节偏离起始位 >3.0 rad 自动停（`--max-excursion` 调；
  按数据集 199 轨合法抓取包络 max 2.785 重标，旧 0.25 真任务 100% 误触）
- rounds 默认 60（单次抓取全程预算，一轮≈0.4s / 0.15 rad 行程）
- 推理在执行期间后台完成，p50 ~255 ms、重规划 2.5 Hz 为正常水位

**看什么**：每轮遥测分位数；回读跟踪误差 mrad；相邻步指令增量（抖动代理）。
已知残留：指令切换瞬间一次抖动（待调项，见 agent_memory）。

---

## 2. 夹爪标定（换机械爪/漂移才需要）

```bash
# 只读探针（判断反馈活着没：width/velocity/effort/is_moving）
~/holy/run.sh ~/holy/scripts/inference/gripper_calib.py --read-only

# 全行程标定（先左后右，结束停在张开位）→ 抄下打印的 --grip-wmin/--grip-wmax
~/holy/run.sh ~/holy/scripts/inference/gripper_calib.py
```

拿到新标定后重跑 §3 的 prep 步带上 `--grip-wmin <值> --grip-wmax <值>`，
然后 `norm_align_check` 过一遍。**当前实测：0.0005 / 0.1200 m（2026-09-23）。**
坑：命令立即执行但反馈滞后 ~6.3s（is_moving 恒 False），判"没动"前轮询 ≥8s——
脚本已内置该守卫。

---

## 3. 权重接入（换 ckpt 才需要，三步）

```bash
# ① stats 提取（微调输出目录 → norm_stats.json）
~/holy/run.sh ~/holy/scripts/inference/lerobot_stats_extract.py --ckpt <微调输出>/pretrained_model

# ② 组装部署目录（软链 14G 权重 + config + manifest；norm_mode 必须显式）
~/holy/run.sh ~/holy/scripts/inference/g1_ckpt_prep.py \
    --src <微调输出>/pretrained_model --out ~/holy/models/<新目录> \
    --mode q01_q99 --grip-wmin 0.0005 --grip-wmax 0.1200

# ③ 归一化对齐检查（全绿才用）
~/holy/run.sh ~/holy/scripts/eval/norm_align_check.py \
    --dataset ~/holy/datasets/pick_place_balence --ckpt ~/holy/models/<新目录>
```

然后改 `config/g1.toml` 的 `[run].ckpt` 指向新目录即可。
训练指令就两句（tasks.parquet 原句，prompt 必须用它们）：

```
Left arm pick up A. Right arm pick up A.
Left arm places A in the top-left corner. Right arm places A in the top-left corner.
```

---

## 4. 场景要求（语义验证前置，2026-09-23 实证）

训练场景 = **桌面 + 蓝色料盒（放在带脚轮小推车上）+ 盒内白色空气开关
（红色扳把，"A"），双臂从桌沿伸入、起点爪几乎压在盒沿正上方**。
场景不复现时模型输出"无条件均值"（臂 14 维 ≈0 rad、夹爪 ≈10-13%）——
这是 OOD 标准行为，不是 bug；warmup 臂位即采集起始位，摆好场景后从 §① 重跑。

**摆位对位工具**（十几秒/轮，不加载模型）：

```bash
~/holy/run.sh ~/holy/scripts/inference/g1_scene_check.py
# 输出 /tmp/scene_check/compare_*.png（左=当前 / 右=训练并排）
```

判据（**腕部两张是决定性的**，头部构图像不够——2026-09-23 教训：
头部看似接近、模型仍输出均值，腕部一看盒子缩在视野边缘）：
- ✅ 蓝盒占腕部视野下半较大比例、空气开关清晰可见、盒沿在爪正前下方
- ❌ 盒子缩在视野边缘/很远、看到的多是空桌板 → 机器人向前挪或盒子往桌沿挪

参考帧入库在 `docs/scene_reference/`（训练 episode0 起点 + 当前反例各一套）。

---

## 5. 验收与探针（装机/排障）

```bash
bash ~/holy/scripts/verify_deploy.sh            # 一键验收（只读+GPU 推理，9 项）

# state prompt 模式 A/B（exact vs fixed 的 800ms 差距复现实验）
~/holy/run.sh ~/holy/scripts/inference/g1_state_prompt_mode_test.py

# 显存/分配探针
~/holy/run.sh ~/holy/scripts/inference/g1_alloc_probe.py
```

---

## 6. config/g1.toml 速览

| 小节 | 管什么 | 当前工作点 |
|---|---|---|
| `[run]` | ckpt 目录 + 任务指令 | pi05_g1_ft / pick up A 原句 |
| `[warmup]` | 预热速度/跳段 | 0.15 rad/s，全段执行 |
| `[execute]` | smoke 步数/限幅/速度 | 3 步 / 0.05 rad / 0.15 rad/s |
| `[loop]` | 闭环全部参数 | speed 0.25、合步 3、switch 0.06、track |
| `[gripper]` | 开关+速度/力矩/阈值 | 开 / 0.05 m/s / 30 N / 2% |
| `[inference]` | 只读推理 | 10 轮 / hold 10 / int8_full / 30 Hz |

`--exec`（真实运动）**不进配置**，每次命令行显式给——防误触。
改 ckpt/prompt/参数：直接编辑 toml；临时换：CLI 传一次即可。
