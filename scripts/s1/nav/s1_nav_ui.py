#!/usr/bin/env python
"""S1 点击导航可视化台（G1 lesson_2_2_navigation/test_2_2_realtime_nav.py 的 S1 重建版）。

浏览器打开 → 看地图点位置 → A* 预览 → 确认下发 navigate_to_goal。
安全语义照搬 G1 定案：点击落在障碍/膨胀区/图外一律拒绝（目标点绝不吸附），
只有起点（定位位姿）允许吸附；STOP 常驻按钮 = stop_navigation + 清预览。

架构（对照 G1 版的三点改动）：
  1. 背景图从 MJPEG 改为静态 PNG + 前端 canvas 叠加（地图不变就不重推流，
     位姿/路径走 2Hz JSON 轮询，headless 零依赖：stdlib + numpy + cv2）
  2. SDK 全部延迟导入且只在 --exec 在线模式加载；默认离线模式零 SDK 零运动
     （--start 给起点即可离线试 A* 预览，先验地图背景对不对再碰 pns）
  3. S1 差异照 s1_base_move.py：galbot_sdk.s1 命名空间、omni_plan 必须 False
     （pyi 明示 S1 不支持全向规划，True=INVALID_INPUT）、navigate_to_goal
     timeout 默认 8s 太短 → 本工具默认 120s（到点即停，超时 SDK 自动 stop）

依赖：flash_pyrt311（numpy 1.26.4 + cv2 4.10.0 已验）；在线模式另需
PYTHONPATH/LD_LIBRARY_PATH=/userdata/update/manual_update/lib（同 s1_base_move.py）。
前置：在线模式需 pns(service_navigation_plan) 在跑——本机开机 DATA_COLLECTION
模式不拉 pns，须切 WORKING 模式（mode_service_config_default.toml）。

用法：
  # 离线建图预览（零运动，先跑这个看地图对不对）
  python s1_nav_ui.py                          # 默认图 /var/maps/cur，:8765
  python s1_nav_ui.py --start 0.2 0.2          # 离线也试 A* 预览
  # 在线模式（WORKING 模式 + 定位 score>=0.8 之后）
  python s1_nav_ui.py --exec

浏览器 http://<机子IP>:8765 → 点图 → 预览红虚线 → 「下发」→ 机器人走 → STOP 停。
退出 Ctrl+C（SDK 优雅三连关闭后 os._exit，S1 退出段 segfault 教训）。
"""
import argparse
import heapq
import json
import math
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2

CACHE_DIR = "/tmp/s1_nav_ui_cache"


# ─────────────────────────── 地图 → 2D 栅格 ───────────────────────────

def load_pcd_xyz(path):
    """binary PCD（x y z intensity ring time, 4/4/4/4/2/8 B）→ (N,3) float32 xyz。"""
    with open(path, "rb") as f:
        head = f.read(4096)
        di = head.find(b"DATA binary\n")
        if di < 0:
            di = head.find(b"DATA binary\r\n")
        if di < 0:
            raise RuntimeError("只支持 DATA binary 版式的 PCD")
        n_pts = int([l.split()[1] for l in head.decode(errors="replace").splitlines()
                     if l.startswith("POINTS")][0])
        data_off = head.find(b"\n", di) + 1
        f.seek(data_off)
        pt = 26  # 4+4+4+4+2+8
        raw = f.read(n_pts * pt)
    arr = np.frombuffer(raw, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                             ("i", "<f4"), ("ring", "<u2"), ("t", "<f8")]))
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 200).all(axis=1)
    return xyz[ok]


DECLUTTER_MIN_CELLS = 8  # 连通域面积小于此格数(0.08m²)=建图碎点，剔除


def build_grid(map_dir, res, band_lo, band_hi, inflate_m, declutter=True):
    """PCD → (bg_bgr, meta, obstacles, valid)。

    经典 2D 占用栅格做法（G1 test_2d_map_view 口径的落地版）：
      墙=障碍切片(band) 点云；室内=墙环泛洪封闭区（激光地面回波远距弱，
      环氧地坪尤其如此——可通行域不能只靠地面回波，墙环补全）；gnd=地面
      回波观测（墙环外的开放区补充标记）；膨胀绕墙。
    band_lo/hi 相对地面（自动估地面=低位最密 z 直方仓）。
    数组行序=cy 从 miny 起（A* 内部口径）；渲染 PNG 翻转为 row0=maxy（北向上），
    前端 toPx 用同一约定。valid=可设目标（室内∪gnd 观测 且 非膨胀）。
    """
    pcd = os.path.join(map_dir, "global_cloud_cleaned.pcd")
    if not os.path.isfile(pcd):
        pcd = os.path.join(map_dir, "global_cloud.pcd")
    xyz = load_pcd_xyz(pcd)

    # 地面 = z 低位最密 0.1m 仓（大厅地面回波必然主导低位直方）
    z = xyz[:, 2]
    zmin, zmax = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    lo_half = zmin + (zmax - zmin) * 0.25
    low = z[z < lo_half]
    bins = np.arange(zmin, lo_half + 0.1, 0.1)
    hist, edges = np.histogram(low, bins=bins)
    floor_z = float((edges[int(hist.argmax())] + edges[int(hist.argmax()) + 1]) / 2)

    gnd = xyz[(z > floor_z - 0.3) & (z < floor_z + 0.3)]  # 地面回波（仅供 floor 校验参考）
    band = xyz[(z > floor_z + band_lo) & (z < floor_z + band_hi)]

    # 范围 = 全部非天花板结构点（地面+障碍层+低矮物）的 0.1/99.9 分位
    # （防离群点撑爆画布；天花板不参与——导航面只关心 2m 以下）
    struct_pts = xyz[z <= floor_z + band_hi]
    px = np.percentile(struct_pts[:, 0], [0.1, 99.9])
    py = np.percentile(struct_pts[:, 1], [0.1, 99.9])
    minx = float(np.floor(px[0])) - 0.5
    maxx = float(np.ceil(px[1])) + 0.5
    miny = float(np.floor(py[0])) - 0.5
    maxy = float(np.ceil(py[1])) + 0.5
    W = int(round((maxx - minx) / res))
    H = int(round((maxy - miny) / res))

    def to_cell(pts):
        cx = ((pts[:, 0] - minx) / res).astype(np.int32)
        cy = ((pts[:, 1] - miny) / res).astype(np.int32)
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        return cy[ok], cx[ok]

    walls = np.zeros((H, W), bool)
    r, c = to_cell(band)
    walls[r, c] = True
    if declutter:
        # 孤立碎簇剔除（2026-11-10 room1102 新图：满图散点噪声每个都膨胀成盘，
        # 观感糊+把自由区切 fragmented）。成片结构（墙/桌面/设备）面积大不受影响；
        # 真实细杆（<0.08m² 独立柱）会被一起牺牲——预览口径，权威=pns 现场避障。
        nlab, labs, stats, _ = cv2.connectedComponentsWithStats(
            walls.astype(np.uint8), connectivity=8)
        kill = np.zeros(nlab, bool)
        kill[1:] = stats[1:, cv2.CC_STAT_AREA] < DECLUTTER_MIN_CELLS
        walls = kill[labs]

    # ⚠ 口径决策（2026-11-10 本图实测）：S1 这张图的墙环缺口大+走廊内杂物多，
    # 泛洪封闭内域会漏（1.1m 闭运算也只圈出 99m²）；且 poses.txt 是废值做不了
    # 射线刻槽。故未知区域按"可通行预览"处理——A* 只是预览辅助，真权威=pns
    # 自己的 ESDF/规划（它拿同一张 pcd 建的 ESDF，会拒不可达目标）。
    inflated = cv2.dilate(walls.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (2 * int(round(inflate_m / res)) + 1,) * 2)
                           ).astype(bool)
    valid = ~walls & ~inflated

    # 渲染 BGR：未知灰 / 可通行白 / 膨胀暖橙 / 墙黑；行翻转为 row0=maxy（前端约定）
    bg = np.full((H, W, 3), 180, np.uint8)
    bg[valid] = (255, 255, 255)
    bg[inflated] = (140, 190, 255)
    bg[walls] = (30, 30, 30)
    bg = np.ascontiguousarray(bg[::-1])
    # 角标：坐标方向 + 比例尺（画布即 map 系，方便对朝向争议）
    cv2.putText(bg, f"map +x ->", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    cv2.putText(bg, "map +y ^", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    bar = int(round(1.0 / res))
    cv2.line(bg, (8, bg.shape[0] - 10), (8 + bar, bg.shape[0] - 10), (0, 200, 0), 2)
    cv2.putText(bg, "1m", (10 + bar, bg.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 200, 0), 1)
    meta = {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
            "res": res, "w": W, "h": H, "floor_z": round(floor_z, 2),
            "map_dir": map_dir, "pcd": os.path.basename(pcd)}
    return bg, meta, walls, valid


def cached_grid(map_dir, res, band_lo, band_hi, inflate_m, force=False,
                declutter=True):
    """栅格落盘缓存（/tmp，键=图目录名+global_cloud mtime+参数），二次启动秒开。"""
    pcd = os.path.join(map_dir, "global_cloud_cleaned.pcd")
    if not os.path.isfile(pcd):
        pcd = os.path.join(map_dir, "global_cloud.pcd")
    key = "%s_%d_r%g_b%g_%g_%g_d%d.npz" % (os.path.basename(map_dir.rstrip("/")),
                                           int(os.path.getmtime(pcd)), res, band_lo,
                                           band_hi, inflate_m, int(declutter))
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = os.path.join(CACHE_DIR, key)
    if os.path.isfile(cpath) and not force:
        d = np.load(cpath, allow_pickle=False)
        return d["bg"], json.loads(str(d["meta"])), d["obstacles"], d["valid"]
    bg, meta, obs, valid = build_grid(map_dir, res, band_lo, band_hi, inflate_m,
                                      declutter)
    # meta 存 0-d 字符串数组（uint8 frombuffer 会在 str() 时被 repr 截断→JSON 炸）
    np.savez_compressed(cpath, bg=bg, meta=np.array(json.dumps(meta)),
                        obstacles=obs, valid=valid)
    return bg, meta, obs, valid


# ─────────────────────────── A*（可通行=observed & ~inflated）───────────

def astar(valid, meta, sx, sy, gx, gy):
    """map 米 → 栅格 A* → 简化折线(map 米) 或 None。8 邻接+视线简化。

    ⚠ 性能铁律：核心循环全部用 python 原生 list/float（首版用 numpy 数组
    索引，纯 python 循环里每次标量运算都装箱，20k 格对角线跑到分钟级）。
    """
    res, W, H, minx, miny = meta["res"], meta["w"], meta["h"], meta["minx"], meta["miny"]
    valid_l = valid.tolist()  # H 行 × W 列 python bool（热路径零 numpy）
    snap = max(1, int(round(1.0 / res)))  # 起点允许吸附 1m（G1：仅起点可吸附）

    def cell(x, y, allow_snap):
        cx, cy = int((x - minx) / res), int((y - miny) / res)
        if 0 <= cx < W and 0 <= cy < H and valid_l[cy][cx]:
            return (cx, cy)
        if not allow_snap:
            return None
        best, bd = None, (snap * res) ** 2
        for dy in range(-snap, snap + 1):
            yy = cy + dy
            if not (0 <= yy < H):
                continue
            row = valid_l[yy]
            for dx in range(-snap, snap + 1):
                xx = cx + dx
                if 0 <= xx < W and row[xx]:
                    d = (dx * res) ** 2 + (dy * res) ** 2
                    if d < bd:
                        bd, best = d, (xx, yy)
        return best

    gc = cell(gx, gy, False)  # ⚠ 目标点绝不吸附（G1 定案）；只有起点允许吸附
    sc = cell(sx, sy, True)
    if gc is None:
        return None, "点击处不可点：障碍/膨胀区/图外（目标点不吸附）"
    if sc is None:
        return None, "起点 1m 内无可通行格（定位点在障碍/膨胀/图外？）"

    INF = float("inf")
    g = [[INF] * W for _ in range(H)]
    g[sc[1]][sc[0]] = 0.0
    prev = {}
    pq = [(math.hypot(sc[0] - gc[0], sc[1] - gc[1]), 0.0, sc[0], sc[1])]
    DIR = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
           (1, 1, 1.41421356), (1, -1, 1.41421356), (-1, 1, 1.41421356),
           (-1, -1, 1.41421356))
    gx_c, gy_c = gc
    found = False
    while pq:
        f, cg, cx, cy = heapq.heappop(pq)
        if cx == gx_c and cy == gy_c:
            found = True
            break
        # 陈旧堆条目：f 含启发项 h，须剥出 g 再与当前 g 比（首版误用 f>cg，
        # h>0 时所有节点恒判陈旧，A* 退化成只能走到起点紧邻格）
        if f - math.hypot(cx - gx_c, cy - gy_c) > cg + 1e-9:
            continue
        for dx, dy, w in DIR:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and valid_l[ny][nx]:
                ng = cg + w
                if ng < g[ny][nx]:
                    g[ny][nx] = ng
                    prev[(nx, ny)] = (cx, cy)
                    heapq.heappush(pq, (ng + math.hypot(nx - gx_c, ny - gy_c), ng, nx, ny))
    if not found:
        return None, "A* 无通路（被障碍/膨胀带隔断）"
    cells, cur = [], gc
    while True:
        cells.append(cur)
        p = prev.get(cur)
        if p is None:
            break
        cur = p
    cells.reverse()

    def los(a, b):  # 线通行性：稠密采样步进
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        for i in range(n + 1):
            t = i / n
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t))
            if not valid_l[y][x]:
                return False
        return True

    simp, i = [cells[0]], 0
    while i < len(cells) - 1:
        # 贪心前推（首版"从队尾往回扫"是 O(n³)，长路径纯 python 要跑分钟级）
        j = i + 1
        while j + 1 < len(cells) and los(cells[i], cells[j + 1]):
            j += 1
        simp.append(cells[j])
        i = j
    dist = sum(math.hypot((simp[k + 1][0] - simp[k][0]) * res,
                          (simp[k + 1][1] - simp[k][1]) * res) for k in range(len(simp) - 1))
    path = [[round(simp[k][0] * res + minx + res / 2, 3),
             round(simp[k][1] * res + miny + res / 2, 3)] for k in range(len(simp))]
    return {"path": path, "dist_m": round(dist, 2)}, None


# ─────────────────────────── SDK 在线段（--exec 才加载）────────────────

class RobotLink:
    """SDK 会话：init/start_controller/nav 子系统；位姿 2Hz 后台轮询。SDK 全延迟导入。"""

    def __init__(self, nav_timeout, collision_check):
        self.lock = threading.Lock()
        self.nav_timeout = nav_timeout
        self.collision_check = collision_check
        self.pose = None          # (x, y, yaw_deg) | None
        self.localized = False
        self.nav_status = "未连接"
        self.task = "空闲"          # 导航任务状态（_monitor 刷新）
        self.sdk_ready = False
        self._stop = False
        self.robot = self.nav = None

    def start(self):
        from galbot_sdk.s1 import GalbotRobot, GalbotNavigation, S1ControllerName, ControlStatus
        self._S1ControllerName = S1ControllerName
        # 官方三步（docs/s1-nav-relative.md §6.1）：只 init + 切 POSE 控制器。
        # ⚠ 绝不 start_controller/acquire_controller——会夺走 pns→WBC 执行通路，
        #   症状=下发成功、FSM EXEC_TRAJ、底盘位移≈0（2026-11-11 实录）。
        robot = GalbotRobot()
        if not robot.init():
            raise RuntimeError("robot.init 失败")
        nav = GalbotNavigation()
        if not nav.init():
            raise RuntimeError("GalbotNavigation.init 失败")
        st = robot.switch_controller(S1ControllerName.SWERVE_CHASSIS_POSE_CTRL)
        if st != ControlStatus.SUCCESS:
            raise RuntimeError(f"switch_controller(SWERVE_CHASSIS_POSE_CTRL) 失败 {st}")
        self.robot, self.nav = robot, nav
        self.sdk_ready = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        self._poller = t

    def _poll_loop(self):
        # ⚠ 勿调 get_navigation_status()：本机必抛异常并中断任务（docs §6.2），任务状态走 _monitor
        while not self._stop:
            try:
                with self.lock:
                    self.localized = bool(self.nav.is_localized())
                    p = self.nav.get_current_pose() if self.localized else None
                    self.pose = self._parse_pose(p)
                    self.nav_status = "定位正常" if self.localized else "定位无效"
            except Exception as e:
                self.pose = None
                self.nav_status = f"轮询异常: {e}"
            time.sleep(0.5)

    @staticmethod
    def _parse_pose(p):
        if p is None:
            return None
        try:
            if isinstance(p, (list, tuple)) and len(p) >= 7:
                px, py, qx, qy, qz, qw = p[0], p[1], p[3], p[4], p[5], p[6]
            else:
                px, py = p.position.x, p.position.y
                o = p.orientation
                qx, qy, qz, qw = o.x, o.y, o.z, o.w
            yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy),
                                          1.0 - 2.0 * (qy * qy + qz * qz)))
            return (float(px), float(py), float(yaw))
        except Exception:
            return None

    def go(self, x, y, yaw_deg=None):
        """下发导航：先问可达性（不可达拒发），非阻塞下发+后台监控线程报进度。

        yaw_deg=None（默认）=终点保持当前朝向（pns_move_relative 同款语义）。
        """
        with self.lock:
            try:
                cur = self.nav.get_current_pose()
            except Exception as e:
                return f"读取当前位姿失败: {e}"
            if yaw_deg is None:
                qz, qw = float(cur[5]), float(cur[6])
                yaw_deg = math.degrees(math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * qz * qz))
            half = math.radians(yaw_deg) / 2.0
            goal = np.array([x, y, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)],
                            dtype=np.float64)
            try:
                reach = bool(self.nav.check_path_reachability(goal, cur))
            except Exception as e:
                reach = None
                print(f"[go] 可达性查询异常: {e}", flush=True)
            if reach is False:
                self.task = "pns 判不可达，未下发"
                return "pns 判不可达，未下发"
            st = self.nav.navigate_to_goal(goal, enable_collision_check=self.collision_check,
                                           is_blocking=False, timeout=self.nav_timeout,
                                           omni_plan=False)  # S1 铁律：车尾雷达盲区
            s = str(st)
            if "SUCCESS" not in s:
                self.task = f"下发失败 {s}"
                return s
        threading.Thread(target=self._monitor, args=(float(x), float(y)),
                         daemon=True).start()
        return s + "（监控中）"

    def _monitor(self, gx, gy):
        """非阻塞任务监控：check_goal_arrival 在非阻塞模式恒 False（docs §6.2），
        以「位移平台 1.6s + 残差 ≤0.06m」判到位；超时或平台未达则 stop 兜底。"""
        try:
            p0 = self.nav.get_current_pose()
        except Exception:
            self.task = "监控失败（位姿读取异常）"
            return
        best, still, t0 = 0.0, 0.0, time.time()
        while time.time() - t0 < self.nav_timeout:
            time.sleep(0.4)
            try:
                p = self.nav.get_current_pose()
            except Exception:
                continue
            el = time.time() - t0
            d = math.hypot(float(p[0]) - float(p0[0]), float(p[1]) - float(p0[1]))
            resid = math.hypot(float(p[0]) - gx, float(p[1]) - gy)
            if d > best + 0.004:
                best, still = d, 0.0
            else:
                still += 0.4
            self.task = f"执行中 位移{best:.2f}m 残差{resid:.2f}m"
            if still >= 1.6 and el > 1.0:
                break
        if resid <= 0.06:
            self.task = f"已到位（残差 {resid:.2f} m）"
        else:
            self.task = f"未确认到位（残差 {resid:.2f} m），已停"
            try:
                self.nav.stop_navigation()
            except Exception:
                pass
        print(f"[go] {self.task}", flush=True)

    def stop(self):
        with self.lock:
            try:
                return str(self.nav.stop_navigation())
            except Exception as e:
                return f"stop 异常: {e}"

    def shutdown(self):
        self._stop = True
        try:
            if self.robot is not None:
                self.robot.request_shutdown()
                self.robot.wait_for_shutdown()
                self.robot.destroy()
        except Exception as e:
            print(f"[sdk] 关闭异常（忽略）：{e}", flush=True)


# ─────────────────────────── HTTP 壳 ───────────────────────────

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>S1 导航台</title>
<style>
body{margin:0;background:#141a20;color:#dce3ea;font-family:system-ui,sans-serif}
#bar{padding:8px 14px;display:flex;gap:14px;align-items:center;border-bottom:1px solid #2a3340}
#bar b{font-size:15px}
.tag{padding:2px 10px;border-radius:10px;font-size:12px}
.off{background:#5a3a12}.on{background:#1d4d2b}
#info{font-size:13px;color:#9fb0c0;white-space:pre}
#wrap{position:relative;overflow:hidden;width:100vw;height:calc(100vh - 46px);cursor:crosshair}
canvas{position:absolute;left:0;top:0}
button{font-size:14px;padding:6px 18px;border:0;border-radius:6px;cursor:pointer}
#go{background:#2d7d46;color:#fff;disabled:true}
#stop{background:#b03030;color:#fff}
#go:disabled{background:#3a4652;color:#889}
</style></head><body>
<div id="bar"><b>S1 点击导航</b><span id="mode" class="tag off">离线预览</span>
<span id="pose">位姿 --</span><span id="nav">nav --</span>
<span id="score" style="flex:1"></span>
<span style="font-size:12px;color:#9fb0c0">■黑=障碍 ■橙=膨胀(不可点) □白=自由 □灰=未知(按可通行预览,pns为权威)</span>
<button id="go" disabled>下发</button><button id="stop">STOP</button></div>
<div id="wrap"><canvas id="bg"></canvas><canvas id="fg"></canvas></div>
<script>
let M=null,VB=null,view={s:1},preview=null,goalMark=null,startMark=null,down=false;
const bg=document.getElementById('bg'),fg=document.getElementById('fg');
const wrap=document.getElementById('wrap');
fetch('/meta').then(r=>r.json()).then(m=>{M=m;
  VB={minx:M.view_minx??M.minx,miny:M.view_miny??M.miny,
      maxx:M.view_maxx??M.maxx,maxy:M.view_maxy??M.maxy};
  VB.w=M.view_w??M.w;VB.h=M.view_h??M.h;init()});
function init(){
  const img=new Image();
  img.onload=()=>{bg.width=VB.w;bg.height=VB.h;fg.width=VB.w;fg.height=VB.h;
    bg.getContext('2d').drawImage(img,0,0);
    fit();loop();};
  img.src='/map.png?'+Date.now();
}
function fit(){const s=Math.min(wrap.clientWidth/VB.w,wrap.clientHeight/VB.h);
  view.s=s;[bg,fg].forEach(c=>{c.style.width=VB.w*s+'px';c.style.height=VB.h*s+'px';});}
addEventListener('resize',fit);
function toMap(e){const r=fg.getBoundingClientRect();
  const px=(e.clientX-r.left)/view.s, py=(e.clientY-r.top)/view.s;
  return [VB.minx+px*M.res, VB.miny+(VB.h-py)*M.res];}
function toPx(x,y){return [(x-VB.minx)/M.res*view.s,(VB.h-(y-VB.miny)/M.res)*view.s];}
wrap.addEventListener('click',async e=>{
  const [x,y]=toMap(e);
  const r=await fetch('/click',{method:'POST',body:JSON.stringify({x,y})})
    .then(r=>r.json());
  document.getElementById('score').textContent=r.ok
    ? `路径 ${r.dist_m} m，${r.path.length} 段` : ('✗ '+r.reason);
  if(r.ok){preview=r.path;goalMark=[x,y];document.getElementById('go').disabled=false;}
  else{preview=null;goalMark=null;document.getElementById('go').disabled=true;}
  draw();
});
document.getElementById('go').onclick=async()=>{
  if(!goalMark)return;
  const r=await fetch('/go',{method:'POST',body:JSON.stringify({x:goalMark[0],y:goalMark[1]})})
    .then(r=>r.json());
  document.getElementById('nav').textContent='nav '+r.status;
};
document.getElementById('stop').onclick=async()=>{
  await fetch('/stop',{method:'POST'});preview=null;goalMark=null;
  document.getElementById('go').disabled=true;draw();
};
async function loop(){
  const st=await fetch('/state').then(r=>r.json());
  if(st.mode==='exec')document.getElementById('mode').textContent='在线执行';
  document.getElementById('mode').className='tag '+(st.mode==='exec'?'on':'off');
  if(st.pose){document.getElementById('pose').textContent=
    `位姿 x=${st.pose[0].toFixed(2)} y=${st.pose[1].toFixed(2)} yaw=${st.pose[2].toFixed(0)}°`;
    startMark=st.pose;}
  else{document.getElementById('pose').textContent='位姿 --（离线'+(st.start?'(--start)':'未给')+'）';
    startMark=st.start?[st.start[0],st.start[1],0]:null;}
  document.getElementById('nav').textContent='nav '+st.nav_status;
  if(st.task){document.getElementById('score').textContent=st.task;
    if(st.task.indexOf('已到位')===0||st.task.indexOf('未确认到位')===0){
      preview=null;goalMark=null;document.getElementById('go').disabled=true;}}
  draw();setTimeout(loop,500);
}
function draw(){
  const c=fg.getContext('2d');c.clearRect(0,0,fg.width,fg.height);
  if(startMark){const[px,py]=toPx(startMark[0],startMark[1]);
    c.save();c.translate(px,py);c.rotate(-startMark[2]*Math.PI/180);
    c.fillStyle='#2d9dff';c.beginPath();c.moveTo(0,-9);c.lineTo(6,7);c.lineTo(-6,7);
    c.closePath();c.fill();c.restore();}
  if(preview&&preview.length>1){c.strokeStyle='#ff4040';c.lineWidth=2;
    c.setLineDash([6,5]);c.beginPath();
    preview.forEach((p,i)=>{const[px,py]=toPx(p[0],p[1]);i?c.lineTo(px,py):c.moveTo(px,py);});
    c.stroke();c.setLineDash([]);}
  if(goalMark){const[px,py]=toPx(goalMark[0],goalMark[1]);
    c.strokeStyle='#ffcc00';c.lineWidth=2;
    c.beginPath();c.arc(px,py,8,0,7);c.stroke();}
}
</script></body></html>"""


class Ui:
    def __init__(self, args, bg_png, meta, valid, link):
        self.args = args
        self.bg_png = bg_png
        self.meta = meta
        self.valid = valid
        self.link = link          # None=离线
        self.preview = None
        self.lock = threading.Lock()

    def state(self):
        if self.link is not None:
            return {"mode": "exec", "pose": self.link.pose,
                    "localized": self.link.localized,
                    "nav_status": self.link.nav_status, "task": self.link.task,
                    "start": None}
        return {"mode": "offline", "pose": None, "localized": False,
                "nav_status": "离线预览（无 SDK）", "task": "", "start": self.args.start}

    def click(self, x, y):
        start = None
        if self.link is not None and self.link.pose is not None:
            start = self.link.pose[:2]
        elif self.args.start:
            start = tuple(self.args.start)
        if start is None:
            return {"ok": False, "reason": "无起点（在线模式需定位有效；离线模式给 --start x y）"}
        res, err = astar(self.valid, self.meta, start[0], start[1], x, y)
        if err:
            return {"ok": False, "reason": err}
        self.preview = res["path"]
        return {"ok": True, "path": res["path"], "dist_m": res["dist_m"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", default="/var/maps/cur", help="地图目录（默认 cur）")
    ap.add_argument("--res", type=float, default=0.1, help="栅格分辨率 m（默认 0.1）")
    ap.add_argument("--band-lo", type=float, default=0.2, help="障碍切片下沿（离地 m）")
    ap.add_argument("--band-hi", type=float, default=2.0, help="障碍切片上沿（离地 m）")
    ap.add_argument("--inflate", type=float, default=0.44,
                    help="膨胀半径 m（G1 底盘外接圆口径；S1 更大可调）")
    ap.add_argument("--view-radius", type=float, default=12.0,
                    help="网页只渲染起点周边此半径窗口 m（0=全图；默认 12）")
    ap.add_argument("--no-declutter", action="store_true",
                    help="关掉孤立碎簇剔除（默认开：<0.08m² 碎点=噪声删掉）")
    ap.add_argument("--start", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="离线模式 A* 起点（map 系 m；在线模式起点=定位位姿）")
    ap.add_argument("--exec", dest="do_exec", action="store_true",
                    help="在线模式：连 SDK，点击确认后真下发 navigate_to_goal（需 pns+定位）")
    ap.add_argument("--nav-timeout", type=float, default=120.0,
                    help="navigate_to_goal 超时 s（SDK 默认 8s 太短；超时自动 stop）")
    ap.add_argument("--no-collision-check", action="store_true",
                    help="下发时关动态避障（G1 工位直达惯例；默认开）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rebuild", action="store_true", help="强制重建栅格缓存")
    args = ap.parse_args()

    t0 = time.time()
    bg, meta, obstacles, valid = cached_grid(args.map, args.res, args.band_lo,
                                             args.band_hi, args.inflate, args.rebuild,
                                             not args.no_declutter)
    ok, free = int(valid.sum()), int((valid & ~obstacles).sum())
    print(f"[map] {meta['map_dir']}  点云 {meta['pcd']}  地面 z≈{meta['floor_z']}m")
    print(f"[map] 全图 x {meta['minx']}~{meta['maxx']}  y {meta['miny']}~{meta['maxy']}"
          f"  格 {meta['w']}x{meta['h']}  可通行格 {free}/{ok}"
          f"  （{time.time()-t0:.1f}s）", flush=True)

    # 视图裁剪：网页只显示起点周边窗口（A* 与 valid 栅格仍用全图，不受影响）
    if args.view_radius and args.view_radius > 0:
        rc = args.start if args.start else (
            (meta["minx"] + meta["maxx"]) / 2, (meta["miny"] + meta["maxy"]) / 2)
        rad = args.view_radius
        wx0 = max(meta["minx"], rc[0] - rad)
        wx1 = min(meta["maxx"], rc[0] + rad)
        wy0 = max(meta["miny"], rc[1] - rad)
        wy1 = min(meta["maxy"], rc[1] + rad)
        c0 = int((wx0 - meta["minx"]) / meta["res"])
        c1 = int(round((wx1 - meta["minx"]) / meta["res"]))
        r0 = int((wy0 - meta["miny"]) / meta["res"])
        r1 = int(round((wy1 - meta["miny"]) / meta["res"]))
        Hf = meta["h"]
        bg = bg[Hf - r1:Hf - r0, c0:c1].copy()  # bg 已翻转 row0=maxy
        meta = dict(meta)
        meta.update(view_minx=wx0, view_miny=wy0, view_maxx=wx1, view_maxy=wy1,
                    view_w=c1 - c0, view_h=r1 - r0)
        print(f"[view] 只显示起点周边 {rad}m：x {wx0:.1f}~{wx1:.1f}  y {wy0:.1f}~{wy1:.1f}"
              f"  窗 {meta['view_w']}x{meta['view_h']} 格", flush=True)
    cv2.imwrite(os.path.join(CACHE_DIR, "bg.png"), bg)

    link = None
    if args.do_exec:
        link = RobotLink(args.nav_timeout, not args.no_collision_check)
        print("[sdk] 在线模式：连接 SDK...", flush=True)
        link.start()
        print("[sdk] 就绪（位姿轮询 2Hz）", flush=True)

    ui = Ui(args, None, meta, valid, link)
    png_bytes = open(os.path.join(CACHE_DIR, "bg.png"), "rb").read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 静默访问日志
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/map.png"):
                self._send(200, png_bytes, "image/png")
            elif self.path == "/meta":
                self._send(200, json.dumps(meta).encode())
            elif self.path == "/state":
                self._send(200, json.dumps(ui.state()).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if self.path == "/click":
                self._send(200, json.dumps(ui.click(body.get("x", 0), body.get("y", 0))).encode())
            elif self.path == "/go":
                if link is None:
                    self._send(200, json.dumps(
                        {"status": "离线模式不下发；加 --exec 进在线模式"}).encode())
                elif link.pose is None:
                    self._send(200, json.dumps(
                        {"status": "定位无效，拒绝下发"}).encode())
                else:
                    st = link.go(body.get("x", 0), body.get("y", 0))  # 终点保持当前朝向
                    print(f"[go] navigate_to_goal({body.get('x'):.2f}, {body.get('y'):.2f}) → {st}",
                          flush=True)
                    self._send(200, json.dumps({"status": st}).encode())
            elif self.path == "/stop":
                st = link.stop() if link is not None else "离线无操作"
                ui.preview = None
                print(f"[stop] {st}", flush=True)
                self._send(200, json.dumps({"status": st}).encode())
            else:
                self._send(404, b"{}")

    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"[ui] http://{args.host}:{args.port}  （{('在线执行' if link else '离线预览')}模式）",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] 退出", flush=True)
    finally:
        srv.server_close()
        if link is not None:
            link.shutdown()
        os._exit(0)  # DDS 残留线程防挂起（优雅三连已先行）


if __name__ == "__main__":
    main()
