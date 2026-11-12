#!/usr/bin/env python
"""定点清除地图里机器人自身的噪点壳（S1 建图 CloudCompare 清杂步骤的定点脚本版）。

背景（2026-11-10 room1102 图实测）：建图后机器人身体周边 0.3~1.5m 留下一圈稀疏
自噪点（多路径反射+本体回波，"全身噪点"）；0.1m 栅格+0.44m 膨胀后每个噪点
变成膨胀盘，连成笼子把机器人起始位姿封死——pns 的 ESDF 同源同病，表现=
"没有路径可规划"。官方流程本应 CloudCompare 手工擦除（本图当时只做了
cp global_cloud.pcd global_cloud_cleaned.pcd 的偷懒路径）。

做法：删除【障碍切片带内、以指定位姿为心 R 半径圆柱内】的点——只动障碍带
（0.2~2m），地面回波与天花板不动；墙等结构在 R 外不受影响（用前先 --dry-run
看径向剖面确认 R 内只有噪点没有真障碍）。

安全性：写前自动备份 global_cloud_cleaned.pcd → *.bak-<日期>；dry-run 默认，
--apply 才写；定位/pns 在跑也不受影响（它们用启动时载入的内存图，重启后生效）。

用法（flash_pyrt311）：
  python clean_self_noise.py --at 0.16 0.15 --radius 1.5            # dry-run 剖面
  python clean_self_noise.py --at 0.16 0.15 --radius 1.5 --apply   # 实际写入
"""
import argparse
import os
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_nav_ui import load_pcd_xyz  # 复用 PCD 解析（xyz 部分）


def load_pcd_raw(path):
    """binary PCD → (header_text, raw_bytes)；保留全部 26B 记录用于原样重写。"""
    with open(path, "rb") as f:
        head = f.read(4096)
        di = head.find(b"DATA binary\n")
        if di < 0:
            di = head.find(b"DATA binary\r\n")
        if di < 0:
            raise RuntimeError("只支持 DATA binary 版式")
        n_pts = int([l.split()[1] for l in head.decode(errors="replace").splitlines()
                     if l.startswith("POINTS")][0])
        data_off = head.find(b"\n", di) + 1
        f.seek(data_off)
        raw = f.read(n_pts * 26)
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", default="/var/maps/cur", help="地图目录（默认 cur）")
    ap.add_argument("--at", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="噪点中心（map 系 m；机器人建图时的位置，可用定位位姿）")
    ap.add_argument("--radius", type=float, default=1.5,
                    help="清除半径 m（默认 1.5；先 dry-run 看剖面确认 R 内无真障碍）")
    ap.add_argument("--band", type=float, nargs=2, default=(0.2, 2.0),
                    metavar=("LO", "HI"), help="障碍切片带（离地 m，只清这带内的点）")
    ap.add_argument("--apply", action="store_true", help="实际写入（默认只 dry-run）")
    args = ap.parse_args()
    if args.at is None:
        ap.error("给 --at X Y（机器人建图时位置）")

    pcd = os.path.join(args.map, "global_cloud_cleaned.pcd")
    if not os.path.isfile(pcd):
        pcd = os.path.join(args.map, "global_cloud.pcd")
    xyz = load_pcd_xyz(pcd)
    raw = load_pcd_raw(pcd)
    assert len(raw) // 26 == len(xyz), "raw 与 xyz 点数不一致"

    z = xyz[:, 2]
    floor_z = -0.01  # 与 s1_nav_ui 同一估地面口径
    band = (z > floor_z + args.band[0]) & (z < floor_z + args.band[1])
    d = np.hypot(xyz[:, 0] - args.at[0], xyz[:, 1] - args.at[1])
    kill = band & (d < args.radius)

    print(f"[clean] {pcd}  总点 {len(xyz)}")
    print(f"[clean] 中心({args.at[0]},{args.at[1]}) R={args.radius}m "
          f"带[{args.band[0]},{args.band[1]}]m  将删 {int(kill.sum())} 点")
    edges = np.arange(0, args.radius + 0.3, 0.3)
    hist, _ = np.histogram(d[kill & band], bins=edges)
    for i in range(len(hist)):
        print(f"  r {edges[i]:.1f}~{edges[i+1]:.1f}m: {hist[i]} 点")
    keep = ~kill
    print(f"[clean] 保留 {int(keep.sum())} 点")
    if not args.apply:
        print("[clean] dry-run，未写盘。确认后加 --apply")
        return

    bak = pcd + f".bak-{__import__('datetime').date.today().isoformat()}"
    if not os.path.exists(bak):
        shutil.copy2(pcd, bak)
        print(f"[clean] 已备份 → {bak}")
    head = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
            "FIELDS x y z intensity ring time\nSIZE 4 4 4 4 2 8\n"
            "TYPE F F F F U F\nCOUNT 1 1 1 1 1 1\n"
            f"WIDTH {int(keep.sum())}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {int(keep.sum())}\nDATA binary\n").encode()
    rec = np.frombuffer(raw, np.uint8).reshape(-1, 26)[keep]
    with open(pcd, "wb") as f:
        f.write(head)
        f.write(rec.tobytes())
    print(f"[clean] 已写入 {pcd}（{int(keep.sum())} 点，原 {len(xyz)} 点）。"
          f"定位/pns 重启后生效；score 异常就用备份回滚")


if __name__ == "__main__":
    main()
