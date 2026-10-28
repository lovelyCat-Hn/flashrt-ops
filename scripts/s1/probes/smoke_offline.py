"""S1 回放台离线冒烟（stub 零运动，可在无机器人环境跑）——回归闸：
①exec 路径：set_joint_commands 每 tick 一次、tfs=0、行数对账；
②per-tick 重读链钳：stub 回读=最后已发值 → 链式 0.05/tick 渐进；
③块尾回读残差=0（perfect tracking stub）；④遥测结构完整。
跑法： PYTHONPATH=/data/galbot/lib:<scripts/s1/inference> python smoke_offline.py
（run_s1_loop/replay 模块层 argparse → argv 由本脚本内部换位，勿外部传参）
"""
import sys
sys.argv = ["replay_stream_bench.py", "--synthetic", "2", "--ctrl-hz", "500"]
import importlib.util
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rsb", _here / "replay_stream_bench.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

sent = []           # (names, tfs)——每次直发一条
cur = {}            # 关节名→最后已发值（stub 的"物理"状态）

class Stub:
    def get_joint_positions(self, joint_names=None, **kw):
        return [cur.get(n, 0.0) for n in joint_names]

    def get_gripper_state(self, name):
        class G:
            width = 0.05
        return G()

    def set_joint_commands(self, cmds, joint_names=None, time_from_start_s=0.0):
        assert time_from_start_s == 0.0, f"tfs 应=0，got {time_from_start_s}"
        for n, c in zip(joint_names, cmds):
            cur[n] = float(c.position)
        sent.append((list(joint_names), time_from_start_s))
        return None

    def set_gripper_command(self, *a, **kw):
        return None


names, chunks, grip = m.load_traj(None, 2)
stub = Stub()

from policies.base import HEAD_JOINTS, TORSO_JOINTS

body17 = m.ARM_JOINTS + TORSO_JOINTS + HEAD_JOINTS
tel = m._new_tel(executing=True)
m.run_replay(stub, body17, names, chunks, grip, True, tel)

# dry-run 口径：虚拟跟随下钳位应远小于冻结态（回归 advance_arm_state）
tel_dry = m._new_tel(executing=False)
m.run_replay(None, None, names, chunks, grip, False, tel_dry)
assert tel_dry["chunks"][0]["clamped"] <= tel["chunks"][0]["clamped"] * 2

total_rows = sum(c["rows"] for c in tel["chunks"])
assert len(sent) == total_rows == 100, (len(sent), total_rows)
assert all(t == 0.0 for _, t in sent)
for c in tel["chunks"]:
    assert c["frames"] == 50 and c["clamped"] > 0, c      # 首段链钳必咬（幅值 0.15>0.05/拍）
    assert c["residual_max"] < 1e-6, c                    # stub perfect tracking
    assert c["late_ms_max"] >= 0.0
# 链钳渐进断言：末块末行，任意臂关节 |cur−chunk 末行目标| ≤ 0.05+ε（per-tick 重读下应≈0）
last = chunks[-1][-1]
worst = max(abs(cur[n] - v) for n, v in zip(names, last) if n in m.ARM_JOINTS)
assert worst <= 0.05 + 1e-9, worst
print("exec+dry-run smoke OK")
print("  sent calls:", len(sent), "| all tfs=0:", all(t == 0.0 for _, t in sent))
print("  per-chunk (exec):", [(c["clamped"], round(c["residual_max"], 9))
                              for c in tel["chunks"]])
print("  worst arm residual after chain clamp:", round(worst, 4), "rad")
