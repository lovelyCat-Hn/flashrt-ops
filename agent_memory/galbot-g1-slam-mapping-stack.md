---
name: galbot-g1-slam-mapping-stack
description: G1 建图/定位栈要点:mapping_server=FAST-LIO 风格 IESKF;engine_tools 交互菜单(1存图/4+5增量更新);地图切换/日志/Score 验证方法;updata_maps 工具集
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e26bfa3-5043-47cc-9fd8-74a4472db417
  modified: 2026-09-20T01:54:55.972Z
---

Galbot G1 建图与定位栈(2026-09-18 实测逆向确认;2026-09-20 补 engine_tools 逆向):

- **算法**:FAST-LIO 风格 LiDAR-IMU 紧耦合迭代 ESKF。证据:配置 `/data/galbot/config/mes/eskf.config` 参数名与 FAST-LIO2 逐字一致(gyr_cov/acc_cov/b_gyr_cov/extrinsic_est_en);定位日志模块名 `front_odometry_ieskf.cc`。回环模块存在但默认关(`use_loop_closure=0`)。
- **流程**:启动 `/data/galbot/bin/mapping_server` → 按急停推车 → `/data/galbot/bin/engine_tools` 交互菜单输 1 存图 → Ctrl+C;换图:`mv /var/maps/cur 备份 && mv 新图 cur`;定位由 `launcher.service` 自动拉起,重启才加载新图;发初始位姿:engine_tools 输 2。
- **engine_tools 菜单全表(2026-09-20 从二进制 strings/symbols 逆向)**:1 保存地图 / 2 发初始定位 / 3 pcd→osm / **4 开始更新地图 / 5 结束更新地图(增量更新入口!要求定位 Score≥0.8,新图固定落 /var/maps/updated_map/)** / 6 位姿轨迹录制 / 7 odom 轨迹 / 8 电子围栏 / 9 重定位 / 10 雷达IMU时间戳 / 11 雷达型号 / 12 外参 / 88 退出。菜单 1 = 发 JSON `{"time_stamp","mode","map_name"}` 到 DDS `/galbot/mes/savemap`(embosa),mapping_server 的 `save_map_handle` 收到后落盘,完成日志 `save_map result: success!!!`(/userdata/log/mapping_server/)。SDK **没有**任何存图 API(GalbotNavigation 仅 10 个方法)。
- **存图/更新脚本化工具集(2026-09-20 建,离线验证通过)**:`lesson_2_2_navigation/updata_maps/`——map_common.py(PCD 版式/定位分数解析/engine_tools 管道驱动/体检/2D 预览)、save_full_map.py(菜单1 全量)、update_map_incremental.py(菜单4+5 增量)、activate_map.py(默认 dry-run,--deploy 换 cur+补软链+删旧esdf,纯 mv/ln 无 sudo)、verify_map_setup.py(--map 查候选图/无参查整机+Score)。engine_tools 可用 stdbuf 管道盲发驱动(88 退出已实测)。
- **PCD 两种版式**:localization 版 26B/点(FIELDS x y z intensity ring time,SIZE 4 4 4 4 2 8)= cur 现用;mapping_server 原始输出可能 24B 甚至 **binary_compressed(LZF,列序)**(0731 备份图即此)。activate 前要转换(map_common.convert_to_localization_pcd 支持 binary+binary_compressed,纯 python LZF 兜底与 liblzf 逐字节一致)。SDK 版式要求未知,deploy_map.py(yuqiz/hinge)惯例是转成 26B。
- **验证**:`tail /userdata/log/localization_server/localization_server.INFO`,grep `score:`,≥0.8 为成功(实测 0.997/0.948/0.972)。启动瞬间一条 `imu disconnected` ERROR 属正常。localization_server 每秒往该日志写 `pub score once, score:`——脚本解析日志即可拿分数,零 SDK 依赖。
- **地图格式**:`/var/maps/cur/*.bin`,每点 4×float32(xyz intensity),一 bin 一个关键帧子图;定位加载后会整理子图(39 bin→35)。完整图目录=global_cloud.pcd+poses.txt+times.txt+end_pose.txt+*.bin+map_topo.osm+relocalization/(特征,冷启动由 relocalization_server 生成)+global_cloud_cleaned.pcd(软链!pns 启动硬依赖)+global_cloud_cleaned.esdf(pns 冷启动自建,换图必须删旧的)。
- **坑**:`systemctl stop launcher.service` 会卡 ~90s——hpu_launcher 服务树大(794 任务/21G),TimeoutStopUSec=90s 后强杀,state 变 failed 属正常,不影响 start。`galbot_svc_hpu_comm.service` 是独立服务,不受影响。**换图生效只走冷启动,绝不能脚本 restart launcher(会组杀兄弟进程且可能只拉起部分服务)**。backup 图目录里的 cleaned 软链指向 cur(跨目录悬空),体检要查 realpath。
- **可视化**:`/home/galbot/holy/g1_joint_test_src/lesson_2_1_mapping_slam/` 有现成脚本:test_view_map.py(俯视/侧视/PCD 导出)、test_slice_map.py(高度切片)、轨迹提取(bin 质心≈关键帧位置,可判定推车覆盖范围)。判断建图好坏看**障碍层切片(离地 0.2~2m)**,整云直看会糊。幽灵判据=轨迹走廊(0.6m)内障碍带点数(cur 图为 0=干净)。
- **场地事实**:大开阔厅,净高 4.8m 方格吊顶,中央有 ~4m 圆形天井开孔;环氧地坪反光强,雷达远距地面回波弱、multipath 条纹杂点多;旧图(0731)与新图都是"原地小圈"建图,有效覆盖≈轨迹周边 15m。
- 相关:[[galbot-g1-digital-twin-setup]] [[galbot-g1-navigation-stack-troubleshooting]]
