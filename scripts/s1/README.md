# S1 脚本目录

**S1 机型专用脚本统一放本目录**（2026-10-27 起约定）：关节只读探针、夹爪标定、
S1 执行器/闭环等。G1 系脚本在 `scripts/{inference,g1,probes,test,ab_test,dataset}`，勿混放。

- `body_control/`：数字孪生（真机→MuJoCo 镜像，自 G1 迁移；只读无指令路径）
- `nav/`：底盘移动/定位（自 G1 g1_base_move 迁移；进程隔离+干净关闭纪律保留）

- 关节/API 速查：`docs/s1-joint-api-quickref.md`
- 维护手册与待办：`S1_MAINTENANCE.md`
- ⚠ 真机脚本安全底线：模式切换/接管瞬间手臂可能失力矩坠落（2026-10-27 掉臂事件）——
  脚本必须内置急停可退路径，跑前确认急停在手边。
