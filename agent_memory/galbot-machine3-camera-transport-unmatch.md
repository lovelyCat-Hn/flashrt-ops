---
name: galbot-machine3-camera-transport-unmatch
description: 本机（第三台）SDK 相机时好时坏：失败窗口期相机话题对所有外部进程（SDK/embosa 绑定/topic_tool）整体隐身，重启采集栈即愈（9/29 16:19 重启后 6/6 恢复）；旁路工具已沉淀
metadata:
  node_type: memory
  type: project
  originSessionId: 55e42dec-9ad0-4468-b717-690667d2e886
  modified: 2026-09-29T08:26:06.742Z
---

本机（第三台，GBS_1.16.0.2.rc88）SDK 1.8.1 相机数据问题（2026-09-29 持续追查，关联 [[galbot-machine3-deployment-state]]）：

- **能用的铁证**：9/21 11:25 `project_yuqiz/base_infer/outputs/predictions.jsonl` 真图 [480,640,3]×3（robot_io.py = GalbotRobot.get_rgb_data 同款路径）；9/24 14:05 py3.8 SDK 会话 28 分钟零 fetch failed（且是开机后第一个 SDK 会话——boot 14:01:15）。**"rc88 天生拒绝 SDK"定案已被推翻**。
- **用户经验规律**：重启后第一次能读，之后都不能读（未在本机复现成功过：9/29 两次开机后首探针均失败；其中一次机械臂未上电）。
- **9/29 实测失败机理**：SDK participant 创建时 libembosa 打 "transport no support"（反汇编 `fastdds_participant` 入口 `ldr w0,[x20,#4]` + `cbz w0,1b22f4` 只收 type 0）→ 回退 DEFAULT_UDP_SHM；**capture 守护进程自己的统计表里完全没有 SDK reader 的行**（连 unmatch 都无）= 两个发现域不通。发布侧完全健康：writer 开机即建、swallows/fusion（transport 6 iceoryx2 reader）秒配、30FPS MJPEG 持续推 `server_ip=192.168.1.50`（外部机器，现不通）。
- **已排除**：HPU 持久文件在破坏窗口（9/24 15:00→9/29 09:00）零变更；/dev/shm 13% 无泄漏无僵尸；embosa_ip_config 改动无效已还原（enable_modify_embosa_cfg 开机会重写）；SSH 占总线（谬）；SDK 库被改（md5 一致）；SDK 1.9.1 沙盒测试同样失败（"transport no support" ×7，且其 InterCoreSHMDataClient 导入是内联空壳 ret/0，红鲱鱼）。
- **未钉死的变量**：XCU 侧状态/版本（9/26 前后可能动过）；boot 竞态顺序；9/29 早上 09:41-13:28 之间用户是否用 app/预览占过状态。
- **下一步选项**：① 只重启单个 capture 守护进程验证状态污染位置（比整机重启轻）；② 用本机 mihomo 代理（127.0.0.1:7890）fetch SDK V1.10.0（9/22 发布，aarch64 库已重编）沙盒再测；③ 反汇编 capture 二进制拿 MJPEG 推流协议做本地收图旁路（cameradata 项目只有配置无接收端）。
- 沙盒测试法（可复用）：V1.9.1 库在 /tmp/sdk191/{lib,py}，`LD_LIBRARY_PATH=/tmp/sdk191/lib:/data/galbot/lib PYTHONPATH=/tmp/sdk191/py` 跑探针，机器人文件零改动。注意 /tmp 重启即清。

**2026-09-29 16:19 恢复（重启即愈再应验）+ 旁路判决**：
- 采集栈整体重启（head/right/left capture 换新 PID，surround 这轮没起）后 SDK 1.8.1 **立即恢复 6/6 路**：head L/R + arm L/R color（640×480 JPEG，臂 ~23KB/头 ~106KB）+ arm depth（16UC1 scale=10000=0.1mm），图像内容正常，10s 连续拉 1906 帧全部新鲜时间戳。自检脚本 `~/holy/scripts/test/sdk_camera_smoke.py`。
- **旁路实验判决（重要）**：失败窗口期内，用机器自带 embosa Python 绑定（V9.1.2 与 rc88 同版，`/userdata/update/manual_update/lib/python3.8.10`，系统 py3.8 可用；必须 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，系统 protobuf 4.25.9 拒旧式 _pb2）建与 fusion 同款 LARGE_DATA_TRANSPORT=6 reader 也 **0 帧**；`embosa_topic_tool list` 看不到任何相机话题、echo 无输出（对照 /odom/base_link echo 正常）。→ 失败时 capture 侧发布/发现对**所有外部进程**隐身，不是 SDK 单方面问题；**旁路救不了失败窗口，重启采集栈才是解**。
- 用法沉淀：`QosConf.CreateIntraCoreQos(...)` 返回的就是完整 Qos（core_comm 默认 0=INTRA_CORE）→ `node.CreateSerializationReader(image_pb2.Image, topic, cb, qos)`；相机消息=galbot.sensor_proto.Image(header/height/width/encoding/data/depth_scale)；proto 包在 /data/galbot/lib/python3/site-packages/galbot。relocalization_server 内嵌 py3.8 同款绑定读 lidar 正常，旁路通路本身有效。厂家调试工具 `/data/galbot/bin/embosa_topic_tool`（list/type/info/qos/hz/echo）。旁路脚本 `~/holy/scripts/test/read_camera_bypass.py`（SDK 再坏时先跑它 + topic_tool 判断是"整体隐身"还是"仅 SDK 隐身"，仅后者旁路才有戏）。
- 旧"transport_type=0 反汇编定案"降级为背景：rc88 确实拒绝非零 participant transport（`transport no support` 日志），但 fusion 同样回退 DEFAULT_UDP_SHM 仍配对相机 → 该日志行与配对成败无关，勿再当主因。
