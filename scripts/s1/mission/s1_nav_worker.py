#!/usr/bin/env python
"""常驻导航 worker（S1 dock 编排线，2026-11-12）：pns_point 同源 go 逻辑。

动机：编排线每腿 pns_point 子进程冷启动 ~8-10s（SDK init+1.5s 等+优雅三连
+进程起停），机器人在此期间站桩——用户实测 22.1s/腿里行驶只占 12.1s。
本进程 SDK 会话跨腿保持，省的是冷启动；安全闸一项不省：切控制器/
is_localized/地图一致性/可达性每腿照跑（do_go 原逻辑逐位复用，pns_point
仅两处机械改动：尾部 __main__ 护栏 + bring_up 会话复用，CLI 行为不变）。

协议（run_dock_mission.py --nav-resident ↔ 本进程；与推理 worker 同构）：
  启动 → bring_up(need_ctrl=False)（不切控制器：spawn 时 worker 任务可能
         正持底盘通路）→ 打 @@NAV_READY rc=N（N!=0=bring_up 失败码）
  go LABEL [--speed V] [--timeout S] [--dry-run] [--omni] [--ignore-map]
       → 每腿：切控制器→定位确认→地图闸→可达闸→(彩排|下发+到位判据)
       → 打 @@NAV_DONE rc=N（语义=pns_point 退出码；进程不死，腿失败可重试）
  quit → stop_navigation + 优雅三连 → @@BYE → os._exit(0)

看门狗：CLI 版顶部 faulthandler 300s 一次性炸弹对常驻进程不成立，改为每
腿重挂（timeout+60s dump+硬退）；腿挂死=进程亡 → 编排器重试时重生。
SIGINT 在腿中=KeyboardInterrupt → rc=9 上报不退进程；SIGTERM 起=硬终止
（无 @@NAV_DONE，编排器按失联处置——与推理 worker 同语义）。

用法（由编排器拉起，勿手动；手动导航仍走 nav/pns_point.py）：
  PYTHONPATH=/userdata/update/manual_update/lib python -u s1_nav_worker.py
"""
import faulthandler
import os
import sys

NAV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "nav")
sys.path.insert(0, NAV_DIR)

import pns_point   # noqa: E402  尾部已 __main__ 护栏，import 无副作用

faulthandler.enable()


def leg_watchdog(seconds):
    """重挂一次性看门狗（dump 全线程栈+硬退）。挂死腿=死进程，可诊断可重生。"""
    faulthandler.dump_traceback_later(seconds, exit=True)


def build_opts(tokens):
    """go 行参数 → pns_point.do_go 的 opts（语义对齐 pns_point.parse 默认值）。"""
    opts = {"file": pns_point.DEFAULT_FILE, "note": "", "force": False,
            "speed": 0.2, "omni": False, "dry": False, "timeout": 120.0,
            "ignore_map": False}
    label = None
    i = 0
    while i < len(tokens):
        a = tokens[i]
        if a == "--speed" and i + 1 < len(tokens):
            opts["speed"] = float(tokens[i + 1]); i += 2
        elif a == "--timeout" and i + 1 < len(tokens):
            opts["timeout"] = float(tokens[i + 1]); i += 2
        elif a == "--dry-run":
            opts["dry"] = True; i += 1
        elif a == "--omni":
            opts["omni"] = True; i += 1
        elif a == "--ignore-map":
            opts["ignore_map"] = True; i += 1
        elif label is None:
            label = a; i += 1
        else:
            print(f"[nav-worker] 忽略多余参数：{a}", flush=True); i += 1
    return label, opts


def main():
    m, rc = pns_point.bring_up(need_ctrl=False)
    if m is None:
        print(f"@@NAV_READY rc={rc}", flush=True)
        return rc
    print("[nav-worker] 常驻会话就绪（SDK 会话跨腿保持；每腿仍做切换+定位+"
          "地图+可达闸）", flush=True)
    print("@@NAV_READY rc=0", flush=True)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "quit":
            break
        tokens = line.split()
        if tokens[0] != "go":
            print(f"[nav-worker] 未知命令忽略：{line}", flush=True)
            print("@@NAV_DONE rc=2", flush=True)
            continue
        label, opts = build_opts(tokens[1:])
        if label is None:
            print("[nav-worker] go 缺标签", flush=True)
            print("@@NAV_DONE rc=2", flush=True)
            continue
        leg_watchdog(opts["timeout"] + 60.0)
        try:
            rc = pns_point.do_go(opts, label)
        except KeyboardInterrupt:
            print("[nav-worker] 腿中收到中断 → 止腿不死进程", flush=True)
            rc = 9
        except Exception:
            import traceback
            traceback.print_exc()
            rc = 1
        finally:
            try:
                if pns_point.nav is not None:
                    pns_point.nav.stop_navigation()
            except Exception as e:
                print(f"[nav-worker] stop_navigation 异常（忽略）：{e}", flush=True)
            faulthandler.cancel_dump_traceback_later()
        print(f"@@NAV_DONE rc={rc}", flush=True)

    print("[exit] 收尾 stop_navigation + 优雅三连", flush=True)
    try:
        if pns_point.nav is not None:
            pns_point.nav.stop_navigation()
    except Exception:
        pass
    try:
        if pns_point.robot is not None:
            pns_point.robot.request_shutdown()
            pns_point.robot.wait_for_shutdown()
            pns_point.robot.destroy()
            print("[exit] 优雅三连完成", flush=True)
    except Exception as e:
        print(f"[exit] 关闭异常（忽略）：{e}", flush=True)
    print("@@BYE", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    print(f"[exit] rc={_rc}", flush=True)
    os._exit(_rc)   # S1 退出段 segfault 教训：优雅三连后硬退兜底（CLI 同款）
