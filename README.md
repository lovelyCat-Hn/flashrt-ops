# flashrt-ops

G1 人形机器人 pi0.5/FlashRT 推理部署与运维仓库（Jetson Orin，L4T R35.6.4）。

## 文档索引

| 文档 | 内容 |
|---|---|
| [DEPLOY.md](DEPLOY.md) | 装机部署（环境打包/权重/tokenizer，配 verify_deploy.sh 一键验收） |
| [USAGE.md](USAGE.md) | 日常操作手册（闭环执行器/编排器/使用配方） |
| [BENCHMARKS.md](BENCHMARKS.md) | 基准数据报表（零判读，只收数据表+条件+复现命令；旧档 BENCHMARKS_ARCHIVE.md） |
| [COMMANDS.md](COMMANDS.md) | 常用命令速查 |

## 目录速览

```
scripts/inference/   闭环推理入口（run_g1_loop / run_g1_loop_traj / run_g1_loop_native）
scripts/g1/          整备标定 / 共享核心（g1_warm_core 等）
scripts/probes/      只读探针（视觉依赖度 / 双构造共存 等）
scripts/test/        性能诊断与回放台
scripts/ab_test/     A/B 对照脚本目录（每对比一子文件夹 + README）
scripts/dataset/     数据集工具
config/              g1.toml 等运行配置（导读见 config/README.md）
evidence/            探针/消融原始证据件（按日期归档）
docs/                专题文档（术语对齐等）
models/ FlashRT/     权重与运行环境（tar 打包分发，不入 git）
```

## 约定

- 提交信息：中文一行式，前缀标注类别（如 `evidence:` / `修 …` / `模块名：改动`）。
- C++ 源（`g1_joint_test_src/`）遵循仓库根 `.clang-format` / `.clang-tidy`（tqyh standard 规范，Google 基础 + 4 空格 + 100 列）。
- 运行日志、数据集、外部克隆仓库一律不入库（见 `.gitignore`）。
