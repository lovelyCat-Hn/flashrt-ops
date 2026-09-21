---
name: galbot-g1-navigation-stack-troubleshooting
description: G1 导航栈排障:pns 要 global_cloud_cleaned.pcd 软链;换图后必须重建;关节读不到=RT/WBC 侧问题(急停锁存);launcher 会组杀兄弟进程
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e26bfa3-5043-47cc-9fd8-74a4472db417
  modified: 2026-09-18T07:36:06.039Z
---

Galbot G1 导航栈(service_navigation_plan/pns)排障要点(2026-09-18 实测):

- **换图后必补软链**:pns 启动硬性读 `/var/maps/cur/global_cloud_cleaned.pcd`,缺失则陷入 "Global ESDF map file does not exist" 初始化死循环(每 10s 重试),**所有导航/可达性请求一律拒绝**。官方工作流 = 建软链指向同目录 `global_cloud.pcd`(旧图里该文件就是 30 字节的符号链接)。ESDF(.esdf,旧图 176MB)缺失不要紧,pns 会从 pcd 自建(0.05m 分辨率,~1 分钟,期间进程 99% CPU 属正常,曾有一次自建中崩溃,手动重启后正常)。
- **关节读不到 = RT 侧问题,不是 Jetson 服务**:SDK 读关节名/值为空 ⟺ pns 报 `singorix/wbcs/sensor update timeout`(该 DDS 话题由机器人实时控制器直发,发布者不在 Jetson 上)。此时 pns 有状态门禁(二进制判词 "joint state invalid, navi failed."),请求在规划前被拒,连原地→原地都不可达。根因常是急停后 RT 未重新使能。诊断:官方工具 `/data/bin/wbcs_test` 输 `r` 看 No Errors;恢复:急停按下去再旋开(FAQ Q3);不行按胳膊下示教器绿色按钮;最重整机断电重启。
- **pns 常态噪音(可忽略)**:1Hz 广播 malfunction 134217730、wbcs timeout 警告——9 月旧日志同样存在,慢性;`joint state count` 平时就是 0(待机时 WBC 流静默)。galbot_one_golf 型号缺 `default_col_setting.toml`(只有 S1 有)→ "left/right arm attach tool failed" 也是出厂常态,不挡底盘导航。
- **launcher 组管理行为**:kill 掉 launcher 管理的服务(pns/localization/relocalization 等)会触发**大规模组杀**:相机栈、motion_plan、fusion、swallows(WBC 桥)全被连带杀掉并标记 active-kill **不再重启**,launcher 进程列表退化成 8 个基础服务,关节数据断流(SDK init 挂起)。手动 nohup 拉起部分可行但麻烦,**冷断电重启是可靠恢复手段**(2026-09-18 下午实测)。教训:要在重启前 kill 服务换图,不如直接换完图冷重启。
- **launcher 按整机模式拉服务**:`mode_service_config_default.toml`(/userdata/user_config/ 可覆盖 default;/data/galbot/config/{pw1,store_vla}/ 有变体)按 WORKING/TELEOPERATION/DEPLOYMENT/DATA_COLLECTION 模式决定服务清单。WORKING_MODE 才含 singorix_wbcs_main(即进程 swallows,wbcs 话题发布者)+pns+相机栈+motion_plan。**systemctl restart launcher 可能只起 8 个基础设施服务(模式引擎未生效)→ 关节数据全无 → switch_controller TIMEOUT**;冷断电重启才按 auto_enter_mode_type=WORKING_MODE 完整拉起。排障先看 hpu_launcher 日志 `process_list` 条数。
- **"原地小圈"建图的幽灵障碍坑(2026-09-18 确诊,同日已验证修复)**:推车绕机器人 r≈1m 小圈建图,推车人影+环氧地坪 multipath 在**轨迹经过区**留下大片 0.5~1.5m 高伪点 → pns ESDF 判 `start state collision` → check_path_reachability 全✗含自检,navigate_to_goal 同门禁被拒。判别法:poses.txt 轨迹是机器人走过的地方,走廊内不可能有实墙,密集即伪点(注意 poses.txt 每行是 3×4 行优先矩阵,平移取第 3/7/11 个数)。**修法已验证**:换 map_health_check 体检(轨迹走廊幽灵 0 点、起点 0.5m 净空)→ 换图目录+重建 cleaned.pcd 软链 → **冷断电重启**(pns 自建 ESDF,日志 ESDFManager map size 应等于新图尺寸)→ Score 0.995 → **reachability 首次返回 True,前进 1m 往返双双一次 SUCCESS,误差<3.5cm**。遥控驾驶工具:`lesson_2_1_mapping_slam/test_keyboard_teleop.py`(按住即走、斜坡加减速、平移转向可叠加)。
- **换图标准流程(验证版)**:①存图 engine_tools 菜单 1 → ②体检新图(幽灵/净空)→ ③`mv cur cur_bad_<日期>; mv 新图 cur; ln -s /var/maps/cur/global_cloud.pcd /var/maps/cur/global_cloud_cleaned.pcd`,确认无残留 .esdf → ④**直接冷断电重启**(比逐个重启服务可靠,完整 WORKING_MODE 启动自动加载新图)→ ⑤验证:无 "Global ESDF map file does not exist"、无 "Collision detected"、localization Score≥0.8、关节数 23。
- 相关:[[galbot-g1-slam-mapping-stack]]
