---
name: galbot-g1-chassis-frame-nav-ui
description: G1 底盘朝向最终结论:车头=yaw 直接所指(FRONT_OFF=0,曾误判 −x 已回退);实时 2D 导航工具的安全语义(点障碍拒绝)
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e26bfa3-5043-47cc-9fd8-74a4472db417
  modified: 2026-09-18T09:32:30.166Z
---

Galbot G1 (golf 底盘) 朝向约定与导航交互工具(2026-09-18 现场验证):

- **底盘系朝向(最终结论,2026-09-18 晚)**:localization yaw = 底盘系 +x 轴的方位角;**车头 = yaw 直接所指(+x),可视化 FRONT_OFF=0**。此前曾按现场对照纠正为 −x(yaw+180°),用户当晚确认那个纠正本身是错的并回退——历经 0°/+90°/−90°/+180°/0° 五轮。教训:这类朝向标定问题反复翻转时,一律以用户最新现场观察为准,不要拿旧结论反驳。URDF 摆放(torso_head_mount 在 +y)与实车几何不符,仅作参考。可视化代码:`lesson_2_2_navigation/test_2d_map_view.py`、`test_2_2_realtime_nav.py` 的 FRONT_OFF。
- **实时 2D 导航工具** `lesson_2_2_navigation/test_2_2_realtime_nav.py`:点击→自研 A*(0.1m 膨胀栅格,膨胀 0.44m=底盘外接圆)预览红虚线→网页/按钮确认才下发 navigate_to_goal;**安全语义:点击落在障碍∪膨胀区/图外一律拒绝(目标点绝不吸附),只有起点允许吸附**(定位噪声);STOP 常驻按钮 = stop_navigation+清预览。SDK 无规划路径查询,预览是自研 A* 估计,实际执行以 pns 为准。
- 网页模式:无 DISPLAY 自动启用(MJPEG + HTML 按钮,0.0.0.0:8765,无鉴权仅限调试);主线程 `os._exit(0)` 收尾——galbot_sdk/DDS 残留非 daemon 线程会卡住正常退出(Ctrl+C 后进程不死的根因,旧进程只能 kill -9)。
- 相关:[[galbot-g1-navigation-stack-troubleshooting]]
