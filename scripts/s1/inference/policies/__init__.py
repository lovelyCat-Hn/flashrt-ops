"""policy 适配器注册表——run_s1_loop.py 经此按名加载，新增 policy = 新增一个文件。"""
from __future__ import annotations

from .base import PolicyAdapter
from .fineart_vla import FineartVlaAdapter
from .pi05 import Pi05Adapter

ADAPTERS: dict[str, type[PolicyAdapter]] = {
    FineartVlaAdapter.name: FineartVlaAdapter,
    Pi05Adapter.name: Pi05Adapter,
}


def create(name: str) -> PolicyAdapter:
    if name not in ADAPTERS:
        raise SystemExit(f"未知 policy {name!r}；可用：{sorted(ADAPTERS)}")
    return ADAPTERS[name]()
