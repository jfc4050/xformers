# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import replace
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

import torch

from ... import _is_triton_available
from ..common import register_operator

if TYPE_CHECKING or _is_triton_available():
    from .flash_attn_triton import (
        _flash_attn_backward,
        _flash_attn_forward,
    )

    triton_flash_backward = _flash_attn_backward
    triton_flash_forward = _flash_attn_forward
else:
    triton_flash_backward = None
    triton_flash_forward = None

from .attn_bias import LowerTriangularMask, LowerTriangularMaskWithTensorBias
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
    check_lastdim_alignment_stride1,
)


def _prepare_inputs(inp: Inputs) -> Inputs:
    attn_bias = inp.attn_bias
    if isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        attn_bias = attn_bias._bias

    if isinstance(attn_bias, torch.Tensor) and attn_bias.ndim == 3:
        B = inp.query.shape[0]
        h = attn_bias.shape[0] // B
        attn_bias = attn_bias.reshape(B, h, attn_bias.shape[1], attn_bias.shape[2])

    if not isinstance(attn_bias, torch.Tensor):
        attn_bias = None

    # Make sure that the last dimension is contiguous
    query, key, value = [
        x if x.stride(-1) == 1 else x.contiguous()
        for x in [inp.query, inp.key, inp.value]
    ]
    return replace(inp, attn_bias=attn_bias, query=query, key=key, value=value)


@register_operator
class FwOp(AttentionFwOpBase):
    OPERATOR = triton_flash_forward
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        LowerTriangularMask,
        # TODO: backwards accuracy is failing for a few cases, perhaps we want to disable this for now.
        torch.Tensor,
        LowerTriangularMaskWithTensorBias,
    }
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    NAME = "tritonflashattF"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
        check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if d.device.type == "cuda":
            # Has only been tested on 8.0.
            # Fails on 7.5 with illegal memory access
            if torch.cuda.get_device_capability(d.device) != (8, 0):
                reasons.append("requires A100 GPU")
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:

        causal = isinstance(inp.attn_bias, (LowerTriangularMask, LowerTriangularMaskWithTensorBias))
        inp = _prepare_inputs(inp)

        out, lse, softmax_scale, seed, rng_offset = triton_flash_forward(
            q=inp.query,
            k=inp.key,
            v=inp.value,
            bias=inp.attn_bias,
            causal=causal,
            dropout_p=inp.p,
            softmax_scale=inp.scale_float,
        )
        ctx = Context(
            lse=lse,
            out=out,
            rng_state=torch.tensor([seed, rng_offset], dtype=torch.int64, device="cpu") if inp.p != 0 else None,
            op_bw=BwOp if inp.p != 0 else None,
        )
        return out, ctx


@register_operator
class BwOp(AttentionBwOpBase):
    OPERATOR = triton_flash_backward
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    CUDA_MINIMUM_COMPUTE_CAPABILITY = FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    # SUPPORTED_ATTN_BIAS_TYPES = FwOp.SUPPORTED_ATTN_BIAS_TYPES
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        LowerTriangularMask,
        # TODO: backwards accuracy is failing for a few cases, perhaps we want to disable this for now.
        torch.Tensor,
        LowerTriangularMaskWithTensorBias,
    }
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    NAME = "tritonflashattB"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
        check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if d.device.type == "cuda":
            if torch.cuda.get_device_capability(d.device) != (8, 0):
                reasons.append("requires A100 GPU")
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        causal = isinstance(inp.attn_bias, (LowerTriangularMask, LowerTriangularMaskWithTensorBias))
        inp = _prepare_inputs(inp)

        rng_seed = rng_offset = 0
        if inp.p != 0.0:
            if (
                ctx.rng_state is None
                or ctx.rng_state.dtype != torch.int64
                or ctx.rng_state.device.type != "cpu"
                or ctx.rng_state.shape != (2,)
            ):
                raise NotImplementedError(f"Invalid rng_state: {ctx.rng_state}")
            rng_seed, rng_offset = ctx.rng_state.tolist()

        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            grads = Gradients(
                dq=torch.empty_like(inp.query),
                dk=torch.empty_like(inp.key),
                dv=torch.empty_like(inp.value),
            )
            cls.OPERATOR(
                grad,
                inp.query,
                inp.key,
                inp.value,
                ctx.out,
                ctx.get_padded_lse(128),
                grads.dq,
                grads.dk,
                grads.dv,
                bias=inp.attn_bias,
                softmax_scale=inp.scale_float,
                causal=causal,
                dropout_p=inp.p,
                dropout_seed=rng_seed,
                dropout_seq_offset=rng_offset,
            )
        return grads
