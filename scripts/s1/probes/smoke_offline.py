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
    def __init__(self, cur, sent):
        self.cur, self.sent = cur, sent

    def get_joint_positions(self, joint_names=None, **kw):
        return [self.cur.get(n, 0.0) for n in joint_names]

    def get_gripper_state(self, name):
        class G:
            width = 0.05
        return G()

    def set_joint_commands(self, cmds, joint_names=None, time_from_start_s=0.0):
        assert time_from_start_s == 0.0, f"tfs 应=0，got {time_from_start_s}"
        for n, c in zip(joint_names, cmds):
            self.cur[n] = float(c.position)
        self.sent.append((list(joint_names), time_from_start_s))
        return None

    def set_joint_commands_batch(self, traj):
        names = list(traj.joint_names)
        pts = traj.points
        assert pts, "batch 空"
        tfs = [p.time_from_start_second for p in pts]
        assert all(b < a for b, a in zip(tfs, tfs[1:])), f"tfs 须严格递增: {tfs}"
        for n, c in zip(names, pts[-1].joint_command_vec):
            self.cur[n] = float(c.position)     # 末点=批末行目标
        self.sent.append((names, ("batch", len(pts))))
        return None

    def set_gripper_command(self, *a, **kw):
        return None

    def get_joint_states(self, joint_names=None, **kw):
        class S:
            pass
        out = []
        for n in joint_names:
            s = S()
            s.joint_name = n
            s.position = self.cur.get(n, 0.0)
            s.velocity = s.acceleration = s.current = 0.0
            out.append(s)
        return out


names, chunks, grip = m.load_traj(None, 2)
stub = Stub(cur, sent)

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

# ── 跨行 batch 路径（rows_per_cmd=2 × frames_per_row=4 → 100 行=50 次调用，
#    每次 8 点、tfs 覆盖 2 行窗口 66.7ms 口径）──
cur2, sent2 = {}, []
tel_b = m._new_tel(executing=True)
m.rsl.args.rows_per_cmd, m.rsl.args.frames_per_row = 2, 4
m.rsl._args[0] = m.rsl.args
stub2 = Stub(cur2, sent2)
m.run_replay(stub2, body17, names, chunks, grip, True, tel_b)
m.rsl.args.rows_per_cmd, m.rsl.args.frames_per_row = 1, 1
m.rsl._args[0] = m.rsl.args
assert len(sent2) == 50, len(sent2)
assert all(kind == "batch" and npts == 8 for _, (kind, npts) in sent2), sent2[:3]
assert sum(c["frames"] for c in tel_b["chunks"]) == 400   # 100 行 × 4 帧
assert all(c["residual_max"] < 1e-6 for c in tel_b["chunks"])
worst2 = max(abs(cur2[n] - v) for n, v in zip(names, chunks[-1][-1]) if n in m.ARM_JOINTS)
assert worst2 <= 0.05 + 1e-9, worst2
print("batch-path smoke OK: 50 calls × 8 pts (R=2×fr=4), tfs 严格递增")

# ── 高频日志路径（--log-hz 口径）：tick 对齐 + 节流密度（网格钳 dl 后 ~3-4 读/tick；
#    曾有双 bug：热路径懒 import 顶爆 tick0 + 网格越界尾部空转 → 114 样本/tick）──
m.rsl.args.ctrl_hz, m.args.ctrl_hz = 50, 50
m.rsl.args.log_hz, m.args.log_hz = 200, 200
m.rsl._args[0] = m.rsl.args
cur3, sent3 = {}, []
tel3 = m._new_tel(executing=True)
log3 = {}
stub3 = Stub(cur3, sent3)
m.run_replay(stub3, body17, names, chunks, grip, True, tel3, log3)
m.rsl.args.ctrl_hz, m.args.ctrl_hz = 500, 500
m.rsl.args.log_hz, m.args.log_hz = 0.0, 0.0
m.rsl._args[0] = m.rsl.args
assert len(log3["t"]) > 0 and len(log3["pos"]) == len(log3["t"])
assert len(log3["ticks_tgt"]) == len(log3["ticks_t"]) == 100
assert len(log3["pos"][0]) == len(names)
per_tick = len(log3["t"]) / len(log3["ticks_t"])
assert 0.5 < per_tick <= 6, f"节流失守: {per_tick:.1f} 读/tick"
print(f"log-path smoke OK: {len(log3['t'])} 突发样本 / 100 tick"
      f"（{per_tick:.1f}/tick，节流正常）")
