#!/usr/bin/env python
"""零运动可达性探针：问 pns 本人 check_path_reachability（不切控制器、不下发）。

用法: PYTHONPATH=/userdata/update/manual_update/lib python pns_reach_probe.py [dx [dy]]
默认测当前位姿正前方 1m。退出码 0=可达。
"""
import math
import os
import sys

import numpy as np

from galbot_sdk.s1 import GalbotNavigation

sweep = len(sys.argv) > 1 and sys.argv[1] == "sweep"
dx = float(sys.argv[1]) if len(sys.argv) > 1 and not sweep else 1.0
dy = float(sys.argv[2]) if len(sys.argv) > 2 and not sweep else 0.0

nav = GalbotNavigation()
if not nav.init():
    print("[probe] GalbotNavigation.init 失败")
    os._exit(2)

ok_loc = False
for _ in range(20):
    try:
        if nav.is_localized():
            ok_loc = True
            break
    except Exception as e:
        print(f"[probe] is_localized 异常: {e}")
        break
    import time
    time.sleep(0.5)
print(f"[probe] is_localized={ok_loc}")

p = nav.get_current_pose()
print(f"[probe] current_pose = {p}")
if isinstance(p, (list, tuple)) and len(p) >= 7:
    x, y, qz, qw = p[0], p[1], p[5], p[6]
else:
    x, y = p.position.x, p.position.y
    qz, qw = p.orientation.z, p.orientation.w
yaw = math.degrees(math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * qz * qz))
print(f"[probe] 当前 ({x:.4f}, {y:.4f}) yaw≈{yaw:.2f}°")

start = p
if sweep:
    print("[probe] 八方向扫描（0.5m 与 1m）+ 参考点：")
    targets = []
    for dist in (0.5, 1.0):
        for ang in range(0, 360, 45):
            a = math.radians(ang)
            targets.append((f"{ang:>3}° {dist}m", x + dist * math.cos(a), y + dist * math.sin(a)))
    targets.append(("远点 3m 前方", x + 3.0, y))
    verdicts = []
    for name, gx, gy in targets:
        goal = np.array([gx, gy, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        try:
            ok = bool(nav.check_path_reachability(goal, start))
        except Exception as e:
            ok = f"异常:{e}"
        verdicts.append((name, ok))
        print(f"  {name}: {ok}")
    n_ok = sum(1 for _, v in verdicts if v is True)
    print(f"[probe] 汇总: {n_ok}/{len(verdicts)} 可达")
    ok = n_ok > 0
else:
    goal = np.array([x + dx, y + dy, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    print(f"[probe] 目标 ({goal[0]:.4f}, {goal[1]:.4f}) yaw=0°")
    try:
        ok = nav.check_path_reachability(goal, start)
        print(f"[probe] check_path_reachability → {ok}")
    except Exception as e:
        print(f"[probe] check_path_reachability 异常: {e}")
        ok = None

try:
    nav.stop_navigation()
except Exception:
    pass
print("[probe] done", flush=True)
os._exit(0 if ok else 1)
