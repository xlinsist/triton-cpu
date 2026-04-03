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
def rope_kernel_fw(
    input_ptr,
    in_seq_stride,
    in_batch_stride,
    output_ptr,
    cos_ptr,
    sin_ptr,
    cos_stride,
    sin_stride,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_head = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_seq = tl.program_id(axis=2)

    half = head_dim // 2
    for off in range(0, half, BLOCK_SIZE):
        d = off + tl.arange(0, BLOCK_SIZE)
        mask = d < half

        cos = tl.load(cos_ptr + pid_seq * cos_stride + d, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + pid_seq * sin_stride + d, mask=mask, other=0.0)

        x1_off = pid_seq * in_seq_stride + pid_batch * in_batch_stride + pid_head * head_dim + d
        x2_off = x1_off + half

        x1 = tl.load(input_ptr + x1_off, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_off, mask=mask, other=0.0)

        y1 = tl.fma(x1, cos, -(x2 * sin))
        y2 = tl.fma(x1, sin, x2 * cos)

        tl.store(output_ptr + x1_off, y1, mask=mask)
        tl.store(output_ptr + x2_off, y2, mask=mask)


def rope_forward(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    seq, batch, heads, dim = t.shape
    out = torch.empty_like(t)
    cos = torch.cos(freqs[:seq]).to(t.dtype)
    sin = torch.sin(freqs[:seq]).to(t.dtype)
    rope_kernel_fw[(heads, batch, seq)](
        t,
        t.stride(0),
        t.stride(1),
        out,
        cos,
        sin,
        cos.stride(0),
        sin.stride(0),
        dim,
        BLOCK_SIZE=16,
    )
    return out


def rotary_pos_emb(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=freqs.dtype)
    return torch.outer(t, freqs).float()


@benchmark.measure()
def bench_rope(seq, batch, heads, dim):
    x = torch.randn((seq, batch, heads, dim), device="cpu", dtype=torch.float32)
    freqs = rotary_pos_emb(dim, seq)
    rope_forward(x, freqs)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for s in [64, 128]:
        bench_rope(s, 2, 4, 64)
