import torch

import triton
import triton.language as tl
import benchmark
benchmark.select_cpu_backend()


@triton.jit
def nested_use_same_level_loop_results(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m + offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[None, :]

    for i1 in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j1 in range(0, 2):
            a_ptrs += 2 * stride_n

        for i6 in range(0, 2):
            a1 = tl.load(a_ptrs)
            a_ptrs += 2 * stride_n
            a3 = tl.load(a_ptrs)
            tl.store(c_ptrs, a1)
            c_ptrs += 2 * stride_n

            c_ptrs += 2 * stride_n
            tl.store(c_ptrs, a3)
            c_ptrs += 2 * stride_n
            a_ptrs += 2 * stride_n

        a_ptrs += 2 * stride_n


@benchmark.measure()
def bench_nested_use_same_level_loop_results(n_rows, n_cols):
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape(
        [n_rows, n_cols]
    )
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)

    def grid(meta):
        return (1,)

    nested_use_same_level_loop_results[grid](x, output, x.stride(0), x.stride(1))


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for n_rows, n_cols in [(2**10, 2**10), (2**12, 2**12), (2**14, 2**14)]:
        bench_nested_use_same_level_loop_results(n_rows, n_cols)
