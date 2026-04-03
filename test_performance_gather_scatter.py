import torch

import benchmark
import triton
import triton.language as tl


def _select_cpu_backend_compat():
    try:
        import triton.backends as tb
        if "triton_shared" in tb.backends and "cpu" in tb.backends:
            tb.backends.pop("triton_shared", None)
    except Exception:
        pass
    benchmark.select_cpu_backend()


@triton.jit
def gather_scatter_kernel(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    gather = offs // 4
    x = tl.load(in_ptr + gather, mask=gather < N, other=0)
    tl.store(out_ptr + offs, x)


def gather_scatter(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    out = torch.empty_like(x)
    gather_scatter_kernel[(1,)](x, out, N=n)
    return out


@benchmark.measure()
def bench_gather_scatter(size):
    x = torch.arange(size, device="cpu", dtype=torch.int32)
    gather_scatter(x)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [64, 128, 256]:
        bench_gather_scatter(x)
