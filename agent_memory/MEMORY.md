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
- [pi0.5 环境准备](galbot-pi05-env-setup.md) — Orin sm_87;INT8 W8A8 全路线判死(09-30 消融+旋转;10-06 复核闭环),部署 bf16;**W8A16-decoder 10-07 落地**(8f08ca85,weight-only INT8 手写 kernel,FVK_PI05_RTX_W8A16_DECODER=1 默认关;M=10 每层 −49%,闭环 chunk50 −5ms/chunk10 −10ms,质量近无损;graph-on 229.9ms;**SDK 在场实测 −18.8ms**,loop 388 归因闭合=栈常驻+138/并行取图+13;⚠ 重编 .so 须 env nvcc 11.8);兼容红线 numpy==1.26.4
- [FlashRT 环境打包迁移](galbot-flashrt-env-migration-pack.md) — 同路径 tar 解压即用零编译；软链/editable 随包走；换用户名走干净配方；JP6 不可用；v2 包含权重/norm_stats/tokenizer/scripts，部署指南 ~/holy/DEPLOY.md + verify_deploy.sh 一键验收
- [pi0.5↔GalbotSDK 集成](galbot-pi05-sdk-integration.md) — 3.11 同进程已验证；相机压缩图 CPU 软解；arm 7-DoF 与 libero 动作空间需映射；运行时 1.8.1
- [G1 微调数据集判读](galbot-pi05-g1-finetune-data.md) — pick_place_balence 16/23 维【右臂在前】(info.json 权威，记反会把臂甩背后)；③层 800ms 慢推理已破案(state 文本进 prompt，fixed 模式根治)；闭环 5 轮全绿，残留指令切换抖动待调
- [G1 PVT 轨迹接口危险实录](galbot-g1-pvt-trajectory-hazard.md) — 零运动探针仍致剧烈抖动，traj 路线封存；提速加剧抖动(每点全停,冲击∝速度)，平滑走 track+合步
- [G1 臂控制模式与安全护栏](galbot-arm-control-mode-safety.md) — 臂=刚性位置伺服(SDK 无阻抗切换入口)；臂端 fault 不回传恒 SUCCESS；压桌防护=包络护栏(d26f9ce)；Motion.init 脱机挂死 FK 死路；夹爪 0.5 m/s 疑超域致 fault
- 基准报告在 `~/holy/BENCHMARKS.md`（对外汇报用，echo 机 2026-09-21 全部实测数据）；操作手册在 `~/holy/USAGE.md`（装机+日常使用，随 DEPLOY.md 构成三件套文档）
- [安装类命令用户亲自执行](galbot-user-runs-install-commands.md) — conda/pip/apt 安装发命令清单给用户跑;我只做只读检查和判读
- [本机 GitHub 连接](galbot-github-remote-setup.md) — SSH 账号 lovelyCat-Hn 可用；ghproxy 镜像会卡死 fetch 已移除勿加回；push 仅限 SSH；上游 flashrt-project 无写权限
- [G1 CUDA graph 段错误修复](galbot-g1-cuda-graph-instantiate-fix.md) — L4T r35.6 iGPU 旧式 Instantiate 必崩走 WithFlags；本机 run_* 四脚本 9/30 解封默认开图（env 门控，回退=前缀 PI05_NO_GRAPH=1），闭环 60 轮已验（p50 380ms）；bench 收益仅 ~3% 勿期待翻倍
- [cache_frames=2 判死](galbot-pi05-cache-frames-dead.md) — state 进 prompt 每轮 set_prompt 重置前缀计数=永远全量空转；bench 3.8× 对本管线不可达（droid 基准 prompt 恒定才有）；knob 默认 1 勿再试
- [本机（第三台机）部署进度](galbot-machine3-deployment-state.md) — R35.6.4；环境/热修/tokenizer 全就绪；G1 权重=**10-05 切回 pick**：pi05_g1_pick_deploy+句 "Left arm pick up A..."（place 三键在 config 注释存档可切回）；⚠ only_place 起始夹爪 ≈33% 非 0%；warmup 位姿已外置；horizon 默认 50；⚠ 判别跑疑空爪，**持物物理 place 复验仍挂着**；10-04 RTC 实验完结：B 脚本 hold 兜底 14 跑全绿、**工作点 v2=25/0.38/1.65 已入 config（R 落盒 14.0-14.5s 纪录）**
- [G1 闭环执行器调参定案](galbot-g1-loop-pace-tuning.md) — 臂速律=n/(30×pace×div)；**工作点 v2（10-04 定案）=25/0.38/1.65（1.33×，RTC B 脚本 hold 兜底，R 落盒 14-14.5s）**；pace 低于水位=死 knob（0.35-0.43 墙钟相同）、div 饱和、spc=25 唯一结构杠杆（速度律命中）；9/30 的 0.40 破位/0.38 判死只属于 A 脚本深尾路径；div=覆盖率/滞后滤波；夹爪 0.15 m/s；双臂严格串行、q 停=R 落盒才停；梦游期签名+肥尾；合步不跨 chunk/抽帧在发令瞬间/spc>n 被钳位；**导航栈 SIGSTOP 搁置=−55ms 且 GR3D 50→7%（10-07 实测，水位击穿 pace 有望清零 hold，真机闭环未验）**
- [本机相机 transport 不匹配根因](galbot-machine3-camera-transport-unmatch.md) — 时好时坏，完整重启采集栈即愈（10-07 第三次应验；⚠ 开机首代也会踩，16:44 整机重启后即中）；失败窗口期相机话题对所有外部进程隐身，旁路 reader 也 unmatch，健康代旁路能配对但载荷 0 字节拿不到图（旁路只配判隐身）；工具=read_camera_bypass.py/sdk_camera_smoke.py/embosa_topic_tool
