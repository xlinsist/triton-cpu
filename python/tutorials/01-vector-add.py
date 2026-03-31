"""
Minimal Triton-CPU vector add smoke test.

Run:
  python python/tutorials/01-vector-add.py
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(x)
    n_elements = c.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, c, n_elements, BLOCK_SIZE=16)
    return c


def main() -> None:
    # Keep CPU execution single-threaded for a minimal/stable smoke test.
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Compatible with Triton variants that may not expose set_active_to_cpu.
    set_active_to_cpu = getattr(triton.runtime.driver, "set_active_to_cpu", None)
    if callable(set_active_to_cpu):
        set_active_to_cpu()

    x = torch.arange(16, dtype=torch.float32, device="cpu")
    y = torch.arange(16, dtype=torch.float32, device="cpu") * 10

    c = add(x, y)
    print(c)


if __name__ == "__main__":
    main()
