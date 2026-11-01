"""S1 就位/归位目标位姿推导（common/home 分包，Phase 4 2026-11-01）。

home_pose_targets 原文=run_s1_loop.py（适配器版）逐行搬运；S1_HOME_POSE 改经
common/bodies.s1 模块属性读取——该处是 config/s1.toml [warm_pose] 覆盖的唯一
汇聚点（bb1567b 起 traj 入口在 B.S1_HOME_POSE 回填合并），适配器线经本模块
读同一处，两线同吃覆盖。home_to（链钳渐进+时间轴批就位机）仍住适配器入口：
其依赖 plan_row/read_s1_state/send_grips/sleep_to 全是入口内单消费者执行链，
回放台经 spec_from_file_location 从入口取用同批代码对象（勿挪）。

G1 对应机制=scripts/g1/body/g1_warm_core.load_target（[warmup].pose_file 三级
来源）——S1/G1 warm 形态各自真机验收过（s1_warm @0.15 rad/s vs g1 手跑
warmup 脚本），勿合并。
"""
import numpy as np

import common.bodies.s1 as _B
from policies.base import S1Targets, gripper_pct_to_width_m


def s1_home_pose() -> dict:
    """当前生效的 S1 就位姿态 dict（[warm_pose] 覆盖后=合并结果；未覆盖=
    policies/base.py 烘档）。"""
    return _B.S1_HOME_POSE


def home_pose_targets() -> S1Targets:
    """S1_HOME_POSE（47 集首帧中位；[warm_pose] 可覆盖）→ S1Targets（爪=全开宽度 m）。"""
    hp = s1_home_pose()
    return S1Targets(
        left_arm=np.array(hp["left_arm"], dtype=np.float32),
        right_arm=np.array(hp["right_arm"], dtype=np.float32),
        left_grip_m=gripper_pct_to_width_m(hp["grip_pct"]),
        right_grip_m=gripper_pct_to_width_m(hp["grip_pct"]),
        torso=float(hp["torso"]),
        head=np.array(hp["head"], dtype=np.float32))
