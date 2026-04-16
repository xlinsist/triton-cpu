import os

import torch

import triton
import triton.language as tl
import benchmark
benchmark.select_cpu_backend()


@triton.jit
def mask_kernel(input_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    offs = 100 + tl.arange(0, BLOCK_SIZE)
    out_offs = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offs, mask=offs < size, other=-1)
    tl.store(output_ptr + out_offs, values)


def run_mask(size):
    input_tensor = torch.arange(0, size + 128, device="cpu", dtype=torch.int32)
    output = torch.full((size,), -2, device="cpu", dtype=torch.int32)
    block_size = min(size, 4)
    mask_kernel[(1,)](
        input_tensor,
        output,
        size,
        BLOCK_SIZE=block_size,
    )
    return output


@benchmark.measure()
def bench_mask(size):
    run_mask(size)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [8*256, 16*256, 32*256]:
        bench_mask(size)
