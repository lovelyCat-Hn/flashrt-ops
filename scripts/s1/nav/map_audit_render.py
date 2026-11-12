#!/usr/bin/env python
"""新图体检渲染：全景 2D 占用栅格 + 机器人驻留位周边放大密度图（一次性诊断用）。"""
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_nav_ui import load_pcd_xyz

pcd = sys.argv[1] if len(sys.argv) > 1 else "/var/maps/room1102/global_cloud.pcd"
at = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.081, 0.023)
outdir = "/tmp/s1_map_audit"
os.makedirs(outdir, exist_ok=True)

xyz = load_pcd_xyz(pcd)
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
print(f"[audit] {pcd}  点数 {len(xyz)}")

# 地面 = <0.5m 内最密的 10cm z-bin 中心
zb = np.arange(-2.0, 0.5, 0.1)
hist, _ = np.histogram(z[z < 0.5], bins=zb)
floor_z = zb[int(hist.argmax())] + 0.05
print(f"[audit] floor_z ≈ {floor_z:.2f}")

band = (z > floor_z + 0.2) & (z < floor_z + 2.0)
print(f"[audit] 障碍带点数 {int(band.sum())}")

# ---- 全景 10cm 栅格 ----
res = 0.1
xe = np.quantile(x[~(z > floor_z + 3.0)], [0.001, 0.999])
ye = np.quantile(y[~(z > floor_z + 3.0)], [0.001, 0.999])
nx = int(np.ceil((xe[1] - xe[0]) / res)) + 1
ny = int(np.ceil((ye[1] - ye[0]) / res) + 1)
ix = ((x - xe[0]) / res).astype(int)
iy = ((y - ye[0]) / res).astype(int)
ok = band & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
grid = np.zeros((ny, nx), np.int32)
np.add.at(grid, (iy[ok], ix[ok]), 1)
occ = (grid > 0).astype(np.uint8) * 255
img = cv2.dilate(occ, np.ones((5, 5), np.uint8))  # 0.44m≈5格 膨胀示意
img = cv2.flip(img, 0)  # row0=maxy
cv2.imwrite(f"{outdir}/full_10cm.png", img)
print(f"[audit] 全景 extent x[{xe[0]:.1f},{xe[1]:.1f}] y[{ye[0]:.1f},{ye[1]:.1f}] "
      f"→ {outdir}/full_10cm.png ({nx}x{ny})")

# ---- 驻留位周边 8m 窗 2cm 密度图 ----
r = 4.0
zres = 0.02
win = (np.abs(x - at[0]) < r) & (np.abs(y - at[1]) < r) & band
wx, wy = x[win], y[win]
zn = int(2 * r / zres) + 1
gx = ((wx - (at[0] - r)) / zres).astype(int)
gy = ((wy - (at[1] - r)) / zres).astype(int)
dens = np.zeros((zn, zn), np.float32)
np.add.at(dens, (gy, gx), 1.0)
vis = np.clip(dens / max(dens.max(), 1) * 255 * 4, 0, 255).astype(np.uint8)
vis = cv2.applyColorMap(255 - vis, cv2.COLORMAP_BONE)  # 点多=黑
cx, cy = int(r / zres), int(r / zres)
cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
for rr in (5, 10, 15):  # 1m/2m/3m 参考圈
    cv2.circle(vis, (cx, cy), rr * 5 // 1, (128, 128, 128), 1)
vis = cv2.flip(vis, 0)
cv2.imwrite(f"{outdir}/zoom_2cm.png", vis)
print(f"[audit] 放大窗 8x8m @({at[0]},{at[1]}) 带内 {int(win.sum())} 点 → zoom_2cm.png")
print("[audit] 放大图：白=空 黑=点密（越黑点越多），红叉=机器人位姿，灰圈=1/2/3m")
