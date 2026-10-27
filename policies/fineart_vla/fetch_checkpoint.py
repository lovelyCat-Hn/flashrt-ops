#!/usr/bin/env python
"""从 HF 镜像拉取 fineart_vla checkpoint（纯 stdlib，支持断点续传，零依赖）。

背景：本机直连 huggingface.co 不通，hf-mirror.com 可达（2026-10-27 实测）。
不引入 huggingface_hub（引擎环境红线：装包可能动 numpy 触发 torch/flash_rt ABI 崩，DEPLOY §4-7）。

用法（后台跑）：
    nohup ~/miniforge3/envs/flash_pyrt311/bin/python \
        ~/holy/policies/fineart_vla/fetch_checkpoint.py > ~/holy/logs/fetch_fineart.log 2>&1 &

    # 可选旗标：--dest DIR  --repo REPO_ID  --mirror URL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

REPO = "lerobot/FineART-VLA-subtask-midtrained"
MIRROR = "https://hf-mirror.com"
UA = {"User-Agent": "Mozilla/5.0 (flashrt-ops fetch)"}


def http_read(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_files(mirror: str, repo: str) -> list[dict]:
    url = f"{mirror}/api/models/{repo}/tree/main?recursive=true"
    tree = json.loads(http_read(url))
    return [(f["path"], f.get("size", 0) or 0, f.get("lfs") or {}) for f in tree if f.get("type") == "file"]


def download_one(mirror: str, repo: str, path: str, dest: str, total: int) -> None:
    out = os.path.join(dest, path)
    os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
    url = f"{mirror}/{repo}/resolve/main/{path}"
    have = os.path.getsize(out) if os.path.exists(out) else 0
    if total and have == total:
        print(f"  ✓ 已完整 {path} ({total/1e6:.1f} MB)，跳过", flush=True)
        return
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(out, "ab" if have else "wb") as f:
        got = have
        last = got
        while True:
            chunk = r.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - last >= 200 << 20:  # 每 200MB 报一次进度
                rate = (got - have) / max(time.time() - t0, 1e-6) / 1e6
                print(f"  … {path} {got/1e9:.2f}/{total/1e9:.2f} GB ({rate:.1f} MB/s)", flush=True)
                last = got
    final = os.path.getsize(out)
    if total and final != total:
        raise RuntimeError(f"大小不符 {path}: {final} != {total}（可重跑续传）")
    print(f"  ✓ {path} 完成（{final/1e6:.1f} MB，续传 {have} B，耗时 {time.time()-t0:.0f}s）", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/holy/models/fineart_vla_midtrained"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--mirror", default=MIRROR)
    args = ap.parse_args()

    print(f"repo={args.repo}\nmirror={args.mirror}\ndest={args.dest}", flush=True)
    files = list_files(args.mirror, args.repo)
    files.sort(key=lambda x: x[1])  # 小文件先落，大文件最后（便于快速出首包）
    total_all = sum(t for _, t, _ in files)
    print(f"共 {len(files)} 个文件，合计 {total_all/1e9:.2f} GB", flush=True)
    for path, size, _lfs in files:
        print(f"→ {path}", flush=True)
        for attempt in (1, 2, 3):
            try:
                download_one(args.mirror, args.repo, path, args.dest, size)
                break
            except Exception as exc:
                print(f"  ✗ 第{attempt}次失败: {exc}", flush=True)
                if attempt == 3:
                    return 1
                time.sleep(5 * attempt)
    print("全部下载完成 ✅", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
