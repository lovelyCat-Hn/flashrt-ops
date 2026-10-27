# flashrt-ops

G1/S1 人形机器人 pi0.5/FlashRT 推理部署与运维仓库（Jetson Orin，L4T R35.6.x）。S1 机型建档与维护见 [S1_MAINTENANCE.md](S1_MAINTENANCE.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [DEPLOY.md](DEPLOY.md) | 装机部署（环境打包/权重/tokenizer，配 verify_deploy.sh 一键验收） |
| [USAGE.md](USAGE.md) | 日常操作手册（闭环执行器/编排器/使用配方） |
| [BENCHMARKS.md](BENCHMARKS.md) | 基准数据报表（零判读，只收数据表+条件+复现命令；旧档 BENCHMARKS_ARCHIVE.md） |
| [COMMANDS.md](COMMANDS.md) | 常用命令速查 |
| [S1_MAINTENANCE.md](S1_MAINTENANCE.md) | S1 机维护手册（本机档案/部署状态/待办；关节速查见 docs/s1-joint-api-quickref.md） |

## 目录速览

```
scripts/             脚本，顶层只分机型：g1/ 与 s1/
  g1/inference/        真机闭环/执行入口（run_g1_loop 系列 / 编排器 / g1_config 共享配置）
  g1/deploy/           权重部署链（ckpt_prep 装配 / extract_normalizer_stats / load_pi05 验收）
  g1/body/             本体操作/标定/恢复（底盘横移·夹爪·姿态预热·右臂恢复；warm_core 共享核心）
  g1/g1_joint_test_src/ 关节测试工具集（数字孪生推流 / IK 交互 / 关节读取复位）
  g1/probes/           只读探针（视觉依赖度 / 双构造共存 等）
  g1/test/             性能诊断与回放台
  g1/eval/             精度评估（tf/seam/chunk 增量谱）
  g1/dataset/          数据集工具（含起始位姿提取）
  g1/ab_test/          A/B 对照脚本目录（每对比一子文件夹 + README）
  s1/                  S1 机型专用脚本（探针/标定/执行器，2026-10-27 起）
policies/            policy 层（每 policy 一目录：engine.py 加载器+说明；pi05 在役/fineart_vla 规划）
verify_deploy.sh     装机一键验收入口（见 DEPLOY.md）
config/              运行配置：base.toml 公共基座 + g1/s1 机型继承（导读见 config/README.md）
evidence/            探针/消融原始证据件（按日期归档）
docs/                专题文档（术语对齐等）
models/ FlashRT/     权重与运行环境（tar 打包分发，不入 git）
```

## 约定

- 提交信息：中文一行式，前缀标注类别（如 `evidence:` / `修 …` / `模块名：改动`）。
- C++ 源（`scripts/g1/g1_joint_test_src/`）遵循仓库根 `.clang-format` / `.clang-tidy`（tqyh standard 规范，Google 基础 + 4 空格 + 100 列）。
- 运行日志、数据集、外部克隆仓库一律不入库（见 `.gitignore`）。
