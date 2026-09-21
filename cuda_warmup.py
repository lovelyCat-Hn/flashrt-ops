"""CUDA 暖场重试 —— pi0.5 / FlashRT 启动入口必加。

用法（在 import torch 之后、加载模型之前）:
    from cuda_warmup import cuda_warmup
    cuda_warmup()

背景: 本机(Jetson AGX Orin, JP5.1.4)上 cuBLAS 句柄的首次创建偶尔返回
CUBLAS_STATUS_ALLOC_FAILED，与系统锁页内存波动(相机栈)相关、呈阵发性。
重试即可通过，本函数把启动崩溃变成最多 ~20s 的延迟。
"""

import time


def cuda_warmup(retries: int = 20, delay: float = 1.0) -> None:
    import torch

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            torch.zeros(8, 8, device="cuda")            # 建 CUDA 上下文
            a = torch.ones(64, 64, device="cuda")
            (a @ a).sum()                               # 建 cuBLAS 句柄 + 首个 GEMM
            torch.zeros(4, device="cpu").pin_memory()   # 探一下锁页分配
            torch.cuda.synchronize()
            if attempt > 1:
                print(f"[cuda_warmup] 第 {attempt} 次尝试成功")
            return
        except RuntimeError as e:
            last_err = e
            print(f"[cuda_warmup] 第 {attempt}/{retries} 次失败: {str(e)[:60]}，{delay}s 后重试")
            time.sleep(delay)
    raise RuntimeError(f"cuda_warmup: {retries} 次重试后仍失败: {last_err}")


if __name__ == "__main__":
    cuda_warmup()
    import torch

    print("CUDA 就绪:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
