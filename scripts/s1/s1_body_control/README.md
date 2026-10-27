# s1_body_control — S1 数字孪生（真机 → MuJoCo 单向镜像）

自 G1 数字孪生（`scripts/g1/g1_joint_test_src/` 的 jetson_sender/windows_viewer 系）迁移。
架构不变：**发送端读真机 → JSON 行 TCP :9999 @30Hz → Windows 端 MuJoCo 渲染**，
协议按关节名键控， viewer 端靠名字查模型 qpos 下标。

| 文件 | 跑在哪 | 作用 |
|---|---|---|
| `s1_sender.py` | 本机（Jetson） | SDK 读 17 关节推流（纯只读，无指令路径） |
| `s1_viewer.py` | Windows | mirror 实时镜像 / free 冻结设计 + 滑条面板 + IK + 自碰报警 |
| `s1_constant_sender.py` | 任意 | 全零假数据端，viewer 离线开发/链路自检 |
| `s1_make_mjcf.py` | 本机 / Windows | 厂商 URDF → MuJoCo 可用模型（重写 package:// + 编译） |

## S1 与 G1 版差异（迁移改动点）

1. **17 关节**：双臂 14 + torso 1 + 头 2（G1 21 含 5 leg；S1 无 leg 组）。夹爪不推流
   （S1 夹爪=独立 API 物理宽度 0.007-0.11 m，基座 URDF 无夹爪几何；并入模型后再扩展）
2. **名字别名**：SDK `torso_joint1` ↔ URDF `torso_lift_joint1`（滑柱关节，单位**米**，
   行程 [0,1]）——viewer 内 `MJ_NAME_ALIAS` 处理
3. **渲染模型**：G1 有现成 MJCF 仓库；S1 只有厂商 URDF，用 `s1_make_mjcf.py` 产出
4. **SDK**：`galbot_sdk.s1`（1.10.0），`init(set())` 写法照抄 ~/yuqiz 真机跑通代码；
   读取按显式名字（group 模式乱序坑，G1 教训沿用）

## 三步跑法

```bash
# ① 模型（本机先产出 ready URDF；Windows 也要一份 galbot_s1_description 目录）
~/miniforge3/envs/flash_pyrt311/bin/python s1_make_mjcf.py
#   → s1_twin_ready.urdf（package:// 已重写为绝对路径 + compiler 块）
#   Windows: 整目录拷 /data/galbot/config/galbot_description/galbot_s1_description/
#   （570M meshes 不入 holy 仓），--model 指向本地产物

# ② 发送端（本机；先核对 hostname -I 第一个值是不是 eth0 的 .100 网段）
PYTHONPATH=/data/galbot/lib ~/miniforge3/envs/flash_pyrt311/bin/python s1_sender.py
#   无真机自检: python3 s1_constant_sender.py

# ③ 渲染端（Windows，需 mujoco；G1 时代已装）
python s1_viewer.py --model s1_twin_ready.urdf --host <本机IP>
python s1_viewer.py --model s1_twin_ready.urdf --mode free --panel   # 滑条设计
```

## 红线（继承 G1 + S1 新账）

- **viewer 只准 `mj_forward`，绝不 `mj_step`**——物理步进=失力瘫软，viewer 无任何
  指令路径是刻意设计，改代码别加"顺手发指令"
- 本目录脚本全是只读/仿真侧；真机运动请走独立执行器并遵守
  `scripts/s1/README.md` 的掉臂纪律（急停在手边）
- 头部限位两处口径不一致（SDK 注释 vs teleop TOML），滑条以**模型限位**为准，
  真机命令值以 TOML 为准——两套数字别混用

## 验证状态（诚实账）

- ✅ 本机已验：SDK import（flash_pyrt311）、.pyi 签名核对、make_mjcf 重写步
  （package:// 清零 + compiler 块注入；21 可动关节=17 body+4 轮转向，另有 4 轮
  驱动 continuous 关节）、四个脚本 py_compile
- ⏳ 未验（缺环境）：mujoco 编译步（本机无 mujoco，装好后跑 `--compile` 或直接
  在 Windows 编）、Windows 端渲染、真机 sender→viewer 链路
