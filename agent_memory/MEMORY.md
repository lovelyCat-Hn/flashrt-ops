# Memory Index

> **读法规约**：先读 [OVERVIEW.md](OVERVIEW.md)（全部记忆的分话题摘要 + 「何时读」指引），按需选读细节文件，**勿全量通读记忆目录**。修改任何细节文件后，同步刷新 OVERVIEW 对应摘要段。

- [记忆概要 OVERVIEW](OVERVIEW.md) — 18 条记忆的分话题摘要与阅读时机；需要记忆时从这里入手
- [Galbot SDK group 模式读取顺序错位](galbot-sdk-group-mode-ordering-pitfall.md) — 读关节必须按名字显式读取，group 模式与 names 配对会全错
- [G1 数字孪生架构](galbot-g1-digital-twin-setup.md) — Jetson 推流 + Windows MuJoCo 渲染；只 mj_forward 不 mj_step；新 clone XML 需修 inertia
- [GalbotMotion 帧命名空间坑](galbot-motion-frame-namespace-pitfall.md) — set_end_effector_pose 传链名+显式 Parameter()；IK/GET/SET 三套帧名互不通用；桩文件不可信
- [jetson_ik_move 工具与已验证事实](galbot-jetson-ik-move-tool.md) — IK 交互执行器；leg 只能前后/升降（y 假解坑）；四元数 [qx,qy,qz,qw]
- [G1 建图/定位栈要点](galbot-g1-slam-mapping-stack.md) — FAST-LIO 风格 IESKF；engine_tools 菜单 1 存图/4+5 增量更新/2 发初始位姿；Score≥0.8 看日志；updata_maps 工具集；bin 质心可判推车覆盖
- [G1 导航栈排障](galbot-g1-navigation-stack-troubleshooting.md) — 换图必补 global_cloud_cleaned.pcd 软链；关节空=RT/急停侧问题用 wbcs_test 查；launcher 会组杀兄弟进程
- [G1 底盘朝向与导航 UI](galbot-g1-chassis-frame-nav-ui.md) — 车头=yaw 直接所指(FRONT_OFF=0,曾误判 −x 已回退);2D 导航工具点障碍一律拒绝;SDK 残留线程需 os._exit
- [pi0.5 环境准备](galbot-pi05-env-setup.md) — Orin sm_87;INT8 W8A8 连同 QuaRot 旋转版均判死(09-30),部署定档 bf16 终审;W8A16-decoder 仅剩无损候选(~20-30ms)未开工;兼容红线 numpy==1.26.4
- [FlashRT 环境打包迁移](galbot-flashrt-env-migration-pack.md) — 同路径 tar 解压即用零编译；软链/editable 随包走；换用户名走干净配方；JP6 不可用；v2 包含权重/norm_stats/tokenizer/scripts，部署指南 ~/holy/DEPLOY.md + verify_deploy.sh 一键验收
- [pi0.5↔GalbotSDK 集成](galbot-pi05-sdk-integration.md) — 3.11 同进程已验证；相机压缩图 CPU 软解；arm 7-DoF 与 libero 动作空间需映射；运行时 1.8.1
- [G1 微调数据集判读](galbot-pi05-g1-finetune-data.md) — pick_place_balence 16/23 维【右臂在前】(info.json 权威，记反会把臂甩背后)；③层 800ms 慢推理已破案(state 文本进 prompt，fixed 模式根治)；闭环 5 轮全绿，残留指令切换抖动待调
- [G1 PVT 轨迹接口危险实录](galbot-g1-pvt-trajectory-hazard.md) — 零运动探针仍致剧烈抖动，traj 路线封存；提速加剧抖动(每点全停,冲击∝速度)，平滑走 track+合步
- 基准报告在 `~/holy/BENCHMARKS.md`（对外汇报用，echo 机 2026-09-21 全部实测数据）；操作手册在 `~/holy/USAGE.md`（装机+日常使用，随 DEPLOY.md 构成三件套文档）
- [安装类命令用户亲自执行](galbot-user-runs-install-commands.md) — conda/pip/apt 安装发命令清单给用户跑;我只做只读检查和判读
- [本机 GitHub 连接](galbot-github-remote-setup.md) — SSH 账号 lovelyCat-Hn 可用；ghproxy 镜像会卡死 fetch 已移除勿加回；push 仅限 SSH；上游 flashrt-project 无写权限
- [G1 CUDA graph 段错误修复](galbot-g1-cuda-graph-instantiate-fix.md) — L4T r35.6 iGPU 旧式 Instantiate 必崩走 WithFlags；PI05_NO_GRAPH 已默认关；bench 收益仅 ~3% 勿期待翻倍
- [本机（第三台机）部署进度](galbot-machine3-deployment-state.md) — R35.6.4；环境/热修/tokenizer 全就绪；G1 工作点=pi05_g1_place+only_place（组装三步+对齐校验全绿）；⚠ only_place 起始夹爪 ≈33% 非 0%；warmup 位姿已外置（--pose-file > config > 零配置三级，g1_pose_extract 生成）；horizon 默认 50（9/29 定，10 切片弃用）
- [G1 闭环执行器调参定案](galbot-g1-loop-pace-tuning.md) — 臂速律=n/(30×pace×div)；pace 焊死 0.455 只减不增；div=覆盖率/滞后滤波；工作点 20/1.55 或 25/1.8；梦游期签名+落盒即 q 停；合步不跨 chunk/抽帧在发令瞬间/spc>n 被钳位
- [本机相机 transport 不匹配根因](galbot-machine3-camera-transport-unmatch.md) — 时好时坏，重启采集栈即愈（9/29 16:19 重启后 SDK 6/6 恢复）；失败窗口期相机话题对所有外部进程隐身，embosa 绑定旁路也不通；工具=read_camera_bypass.py/sdk_camera_smoke.py/embosa_topic_tool
