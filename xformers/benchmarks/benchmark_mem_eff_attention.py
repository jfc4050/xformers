# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import math
from functools import partial
from typing import Any, cast

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper

import xformers.ops
import xformers.ops.fmha as fmha

CHECK_CORRECTNESS = True
torch.backends.cuda.matmul.allow_tf32 = False


def create_attn_bias(
    bias_type,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    device,
    dtype,
    bias_requires_grad: bool = False,
):
    NoneType = type(None)
    if bias_type is NoneType:
        return None
    if bias_type is torch.Tensor:
        attn_bias = (
            torch.randn((batch_size * num_heads, 1, kv_len), device=device, dtype=dtype)
            * 3
        )
        return attn_bias.expand(batch_size * num_heads, q_len, kv_len)
    if bias_type is xformers.ops.LowerTriangularMask:
        return bias_type([1, q_len, kv_len], dtype=dtype, device=device)
    assert False, f"Unsupported bias type: {bias_type}"


def ref_attention_bmk(q, k, v, attn_bias=None, p=0.0):
    if isinstance(attn_bias, xformers.ops.AttentionMask):
        attn_bias = attn_bias.to_tensor()
    if isinstance(attn_bias, torch.Tensor):
        attn_bias = attn_bias.to(q.dtype)
    q = q * (1.0 / q.shape[-1] ** 0.5)
    if attn_bias is None:
        attn = q @ k.transpose(-2, -1)
    else:
        # print([x.dtype for x in [attn_bias, q, k]])
        # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
        # but faster, and is what is used in PyTorch now
        attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
    attn = attn.softmax(-1)
    if p > 0:
        attn = torch.nn.functional.dropout(attn, p=p)
    return attn @ v


def ref_attention(q, k, v, attn_bias, p=0.0):
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    out = ref_attention_bmk(T(q), T(k), T(v), attn_bias, p)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


min_run_time = 0.5
device = torch.device("cuda")

NUM_THREADS = [1] if device.type == "cuda" else [1, 40]
SHAPES = [
    # AlexaTM
    (8, 1024, 64, 128),

    # bedrock
    (1, 4096, 12, 128),
    (1, 8192, 12, 128),
]


seed = 0
FORCE_OP = None
# FORCE_OP = xformers.ops.MemoryEfficientAttentionOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionCutlassOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp
FORCE_OP = xformers.ops.TritonFlashAttentionOp
# FORCE_OP = xformers.ops.MemoryEfficientAttentionTritonFwdFlashBwOp


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


CASES = list(
    product_dict(
        shape=SHAPES,
        num_threads=NUM_THREADS,
        dropout_p=[0.0],  # TODO - add this
        attn_bias_cfg=[
            (torch.Tensor, False),
            (torch.Tensor, True),
        ],
        dtype=[torch.bfloat16],
    )
)


def create_tensors(shape, dtype, requires_grad=False):
    B, M, H, K = shape
    qkv = torch.rand(
        [B, M, 3, H, K], device=device, dtype=dtype, requires_grad=requires_grad
    )
    q, k, v = xformers.ops.unbind(qkv, 2)
    return qkv, q, k, v


def benchmark_forward(shape, num_threads: int, attn_bias_cfg, dropout_p, dtype):
    B, M, H, K = shape
    _, q, k, v = create_tensors(shape, dtype)
    attn_bias_type, attn_bias_requires_grad = attn_bias_cfg
    bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        num_heads=H,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
        bias_requires_grad=attn_bias_requires_grad,
    )
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)

    try:
        op = (fmha._dispatch_fw(inp), None) if FORCE_OP is None else FORCE_OP
    except NotImplementedError:
        return

    if not op[0].supports(inp):
        return

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = (
        f"{dtype_str} B={B}, M={M}, H={H}, K={K}, p={dropout_p}, "
        f" BiasT={attn_bias_type.__name__}, BiasGrad={attn_bias_requires_grad}"
    )

    try:
        r = xformers.ops.memory_efficient_attention(
            q, k, v, inp.attn_bias, op=op
        ).float()
        rr = ref_attention(
            q.float(),
            k.float(),
            v.float(),
            inp.attn_bias,
            inp.p,
        )

        assert not (
            inp.p > 0 and CHECK_CORRECTNESS
        ), "correctness checking not yet implemented for dropout"
        assert not CHECK_CORRECTNESS or (r - rr).abs().max() < 4e-3, (
            (r - rr).abs().max()
        )
        del r, rr
    except RuntimeError as e:  # OOM
        raise e

    for op_impl in [xformers.ops.TritonFlashAttentionOp, xformers.ops.MemoryEfficientAttentionCutlassOp]:
        yield benchmark.Timer(
            stmt="fn(q, k, v, attn_bias, p)",
            globals={
                "q": q,
                "k": k,
                "v": v,
                "attn_bias": inp.attn_bias,
                "p": dropout_p,
                "fn": partial(xformers.ops.memory_efficient_attention, op=op_impl),
            },
            label=f"attention (attn_bias={attn_bias_type})",
            description=op_impl[0].NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )
    yield benchmark.Timer(
        stmt="fn(q, k, v, attn_bias, p)",
        globals={
            "q": q,
            "k": k,
            "v": v,
            "attn_bias": inp.attn_bias,
            "p": dropout_p,
            "fn": ref_attention,
        },
        label=f"attention (attn_bias={attn_bias_type})",
        description="eager",
        sub_label=sub_label,
        num_threads=num_threads,
    )


def benchmark_backward(shape, num_threads: int, attn_bias_cfg, dropout_p, dtype):
    B, M, H, K = shape
    qkv, q, k, v = create_tensors(shape, dtype, requires_grad=True)

    attn_bias_type, attn_bias_requires_grad = attn_bias_cfg
    bias = create_attn_bias(
        attn_bias_type,
        batch_size=B,
        num_heads=H,
        q_len=M,
        kv_len=M,
        device=device,
        dtype=dtype,
        bias_requires_grad=attn_bias_requires_grad,
    )
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias, p=dropout_p)
    try:
        if FORCE_OP:
            op = FORCE_OP
        else:
            op_fw: Any = None
            op_bw = fmha._dispatch_bw(inp)
            if op_bw == fmha.flash.BwOp:
                op_fw = fmha.flash.FwOp
            elif op_bw == fmha.cutlass.BwOp:
                op_fw = fmha.cutlass.FwOp
            else:
                op_fw = fmha._dispatch_fw(inp)
            op = (op_fw, op_bw)
    except NotImplementedError:
        return
    if not (op[0].supports(inp) and op[1].supports(inp)):
        return

    dtype_str = {
        torch.bfloat16: "b16",
        torch.half: "f16",
        torch.float: "f32",
    }[dtype]
    sub_label = (
        f"{dtype_str} B={B}, M={M}, H={H}, K={K}, p={dropout_p},"
        f" BiasT={attn_bias_type.__name__}, BiasGrad={attn_bias_requires_grad}"
    )

    for op_impl in [xformers.ops.TritonFlashAttentionOp, xformers.ops.MemoryEfficientAttentionCutlassOp]:
        out = xformers.ops.memory_efficient_attention(
            inp.query, inp.key, inp.value, inp.attn_bias, inp.p, op=op_impl
        )
        grad_benchmark = torch.ones_like(q)
        yield benchmark.Timer(
            stmt="out.backward(grad, retain_graph=True)",
            globals={
                "out": out,
                "grad": grad_benchmark,
            },
            label=f"attention backward (attn_bias={attn_bias_type})",
            description=op_impl[1].NAME,
            sub_label=sub_label,
            num_threads=num_threads,
        )
        del out

    try:
        qkv.grad = None
        r = xformers.ops.memory_efficient_attention(
            q, k, v, inp.attn_bias, dropout_p, op=op
        )
        r.backward(torch.ones_like(q))

        grad = cast(torch.Tensor, qkv.grad)
        qkv.grad = None

        rr = ref_attention(q, k, v, inp.attn_bias, dropout_p)
        rr.backward(torch.ones_like(q))
        atol = 2e-4 + 2e-6 * K * M * math.sqrt(B) * math.sqrt(M)
        assert not (
            dropout_p > 0 and CHECK_CORRECTNESS
        ), "correctness checking not yet implemented for dropout"
        # type: ignore
        assert (
            not CHECK_CORRECTNESS or (grad - qkv.grad).abs().max() < atol
        ), f"{(grad - qkv.grad).abs().max()}"
        qkv.grad = None
        del r, grad

        yield benchmark.Timer(
            stmt="out.backward(grad, retain_graph=True)",
            globals={
                "out": ref_attention(q, k, v, inp.attn_bias, dropout_p),
                "grad": grad_benchmark,
            },
            label=f"attention backward (attn_bias={attn_bias_type})",
            description="vanilla",
            sub_label=sub_label,
            num_threads=num_threads,
        )
    except RuntimeError:  # OOM
        pass


benchmark_main_helper(benchmark_forward, CASES, min_run_time=min_run_time)
benchmark_main_helper(benchmark_backward, CASES, min_run_time=min_run_time)
