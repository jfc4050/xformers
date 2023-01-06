from argparse import ArgumentParser

import pytest
import numpy as np
import torch
import torch.utils.benchmark as benchmark

import xformers
from xformers.ops import MemoryEfficientAttentionCutlassOp

# torch.manual_seed(0)
torch.set_printoptions(profile="full", linewidth=250, sci_mode=False)

import termcolor


def bmk2bmhk(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    # b * n_heads, seq, seq
    # b, n_heads, seq, seq
    # b, seq, n_heads, seq
    return tensor.reshape([-1, num_heads, tensor.shape[1], tensor.shape[2]]).permute(
        (0, 2, 1, 3)
    )


def ref_attention_bmhk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    rand_uniform: torch.Tensor,
    p: float,
    debug: bool = False,
) -> torch.Tensor:
    assert q.ndim == 4

    def T(t: torch.Tensor) -> torch.Tensor:
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    x = ref_attention(
        T(q),
        T(k),
        T(v),
        attn_bias=attn_bias,
        rand_uniform=rand_uniform,
        p=p,
        debug=debug,
    )
    x = x.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return x.permute((0, 2, 1, 3))


def ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor = None,
    rand_uniform: torch.Tensor = None,
    p: float = 0.0,
    debug: bool = False,
) -> torch.Tensor:
    orig_dtype = q.dtype
    if q.ndim == 4:
        return ref_attention_bmhk(
            q, k, v, attn_bias=attn_bias, rand_uniform=rand_uniform, p=p, debug=debug
        )

    scale_factor = 1 / q.shape[-1] ** 0.5
    q = q * scale_factor
    attn = q @ k.transpose(-2, -1)
    if debug:
        print(f"attention_before: {attn.shape}")
        print(attn)
    if attn_bias is not None:
        if len(attn_bias.shape) == 2:
            attn_bias = attn_bias.reshape(1, 1, attn_bias.shape[0], attn_bias.shape[1])
        elif attn_bias.shape[0] != attn.shape[0]:
            attn_bias = bmk2bmhk(attn_bias, k.shape[2])
        attn = attn + attn_bias

        if debug:
            print(f"attn after bias: {attn.shape}")
            print(attn)

    attn = attn.softmax(-1)
    if p > 0:
        if rand_uniform is not None:
            if debug:
                print(f"attn: {attn.shape} rand_uniform: {rand_uniform.shape}")
            attn = attn * (rand_uniform > p).to(attn.dtype)
            attn = attn / (1.0 - p)
            if debug:
                print("after dropout")
                print(attn)
        else:
            attn = torch.nn.functional.dropout(attn, p=p)

    attn = attn @ v

    return attn


def assert_allclose(
    out: torch.Tensor,
    ref: torch.Tensor,
    msg: str = "failed",
    atol: float = 9.5e-2,
    rtol: float = 2e-2,
) -> bool:
    assert out.shape == ref.shape
    flatten_diff = ((out - ref).abs() - atol - ref.abs() * rtol).flatten()
    max_pos = flatten_diff.argmax()
    num_different = torch.count_nonzero(flatten_diff > 0)
    percentage = num_different / flatten_diff.numel()
    passed, errmsg = torch.allclose(out, ref, rtol=rtol, atol=atol), (
        f"{msg}: "
        f"out={out.flatten()[max_pos]} and ref={ref.flatten()[max_pos]} (diff={flatten_diff[max_pos]} > 0)"
        f"/ atol={atol}, rtol={rtol}"
        f"/ total failing elements: {num_different}, percentage={percentage}"
    )
    if not passed:
        # print("out")
        # print(out.squeeze(), "red")

        # print("ref")
        # print(ref.squeeze())

        print("diff")
        print((out - ref).abs().squeeze())

        print(errmsg)

    return passed


@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attn(dtype, dropout_p):
    batch_sz = 1
    n_queries = 1024
    # n_queries = 32
    n_keys = 1024
    # n_keys = 1024
    n_heads = 1
    # head_dim = 24
    head_dim = 128
    seed = 10
    use_bias = False

    query = torch.randn(
        batch_sz,
        n_queries,
        n_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    key = torch.randn(
        batch_sz,
        n_keys,
        n_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    # key = torch.arange(0, n_keys * head_dim, dtype=dtype, device="cuda").reshape(n_keys, head_dim) / 100
    # key = key[None, :, None, :].expand(batch_sz, n_keys, n_heads, head_dim).contiguous()
    # key *= 10
    # key.requires_grad_(True)

    value = torch.randn(
        batch_sz,
        n_keys,
        n_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    bias = 5 * torch.randn(
        batch_sz * n_heads,
        n_queries,
        n_keys,
        dtype=dtype,
        device="cuda",
        requires_grad=False,
    )
    bias.requires_grad_(True)
    if not use_bias:
        bias = None
    rand_uniform = torch.empty(
        batch_sz,
        n_heads,
        n_queries,
        n_keys,
        dtype=torch.float32,
        device="cuda",
        requires_grad=False,
    )
    torch.manual_seed(seed)
    if dropout_p > 0:
        rand_uniform = torch.ops.xformers._cutlass_rand_uniform(dropout_p, rand_uniform)
        rand_uniform.requires_grad_(False)
        rand_uniform = rand_uniform.squeeze(1)
        rand_uniform = rand_uniform.reshape(batch_sz * n_heads, n_queries, n_keys)
    else:
        rand_uniform = None
    # print("bias")
    # print(bias)
    # print("rand_uniform", rand_uniform.shape)
    # print(rand_uniform)

    torch.manual_seed(seed)
    res = xformers.ops.memory_efficient_attention(
        query,
        key,
        value,
        bias,
        dropout_p,
        op=(xformers.ops.fmha.cutlass.FwOp, xformers.ops.fmha.cutlass.BwOp),
    )
    print("fwd")
    res_ref = ref_attention(
        query.float(),
        key.float(),
        value.float(),
        bias.float() if use_bias else None,
        rand_uniform,
        dropout_p,
        debug=False,
    ).to(dtype)

    assert assert_allclose(res, res_ref, msg="fwd")

    # grad_out = torch.arange(
    #     0,
    #     batch_sz * n_queries * n_heads * head_dim,
    #     dtype=dtype,
    #     device="cuda").reshape(batch_sz, n_queries, n_heads, head_dim).contiguous() / 100
    grad_out = torch.randn(batch_sz, n_queries, n_heads, head_dim, dtype=dtype, device="cuda")

    res.backward(grad_out, retain_graph=True)
    q_grad = query.grad
    k_grad = key.grad
    v_grad = value.grad
    if bias is not None and bias.requires_grad:
        bias_grad = bias.grad
    # print("bias - grad")
    # print(bias_grad)

    query.grad = None
    key.grad = None
    value.grad = None
    if bias is not None:
        bias.grad = None

    res_ref.backward(grad_out, retain_graph=True)
    q_grad_ref = query.grad
    k_grad_ref = key.grad
    v_grad_ref = value.grad
    if bias is not None:
        bias_grad_ref = bias.grad
    # print("bias - grad")
    # print(bias_grad_ref)

    failed = []

    name = "bwd-value"
    print(name)
    passed = assert_allclose(v_grad, v_grad_ref, msg=name)
    if not passed:
        failed.append(name)

    name = "bwd-key"
    print(name)
    passed = assert_allclose(k_grad, k_grad_ref, msg=name)
    if not passed:
        failed.append(name)

    name = "bwd-query"
    print(name)
    passed = assert_allclose(q_grad, q_grad_ref, msg=name)
    if not passed:
        failed.append(name)

    if bias is not None and bias.requires_grad:
        name = "bwd-bias"
        print(name)
        passed = assert_allclose(bias_grad, bias_grad_ref, name)
        if not passed:
            failed.append(name)

    if len(failed) > 0:
        # print("dO @ V")
        # print(grad_out.squeeze() @ value.squeeze().transpose(0, 1))
        assert False, f"failed: {failed}"


def run_once():
    batch_sz = 8
    n_queries = 1024
    n_keys = 1024
    n_heads = 64
    head_dim = 64
    dtype = torch.float16
    dropout_p = 0.5

    for _ in range(1):
        query = torch.randn(
            batch_sz,
            n_queries,
            n_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        key = torch.randn(
            batch_sz,
            n_keys,
            n_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        value = torch.randn(
            batch_sz,
            n_keys,
            n_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        bias = 5 * torch.randn(
            batch_sz,
            n_heads,
            n_queries,
            n_keys,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        bias.requires_grad_(True)
        res = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            bias,
            dropout_p,
            op=(xformers.ops.fmha.cutlass.FwOp, xformers.ops.fmha.cutlass.BwOp),
        )
        grad_out = torch.ones(
            batch_sz, n_queries, n_heads, head_dim, dtype=dtype, device="cuda"
        )

        res.backward(grad_out)


def run_benchmark(cfgs):
    results = []

    for cfg in cfgs:
        batch_sz, seq_len, n_heads, head_dim, dtype = cfg

        for bias_shape in [
            (batch_sz * n_heads, seq_len, seq_len),
            # (seq_len, seq_len),
            None,
        ]:
            for bias_requires_grad in (
                [False, True] if bias_shape is not None else [False]
            ):

                for dropout_p in [0.0, 0.5]:

                    sub_label = str((*cfg, bias_shape, bias_requires_grad, dropout_p))

                    query = torch.randn(
                        batch_sz,
                        seq_len,
                        n_heads,
                        head_dim,
                        dtype=dtype,
                        device="cuda",
                        requires_grad=True,
                    )
                    key = torch.randn(
                        batch_sz,
                        seq_len,
                        n_heads,
                        head_dim,
                        dtype=dtype,
                        device="cuda",
                        requires_grad=True,
                    )
                    value = torch.randn(
                        batch_sz,
                        seq_len,
                        n_heads,
                        head_dim,
                        dtype=dtype,
                        device="cuda",
                        requires_grad=True,
                    )
                    bias = (
                        torch.randn(
                            *bias_shape,
                            dtype=dtype,
                            device="cuda",
                            requires_grad=bias_requires_grad,
                        )
                        if bias_shape is not None
                        else None
                    )

                    globals = {
                        "query": query,
                        "key": key,
                        "value": value,
                        "bias": bias,
                        "dropout_p": dropout_p,
                    }

                    out_ref = ref_attention(query, key, value, bias, dropout_p)
                    out = xformers.ops.memory_efficient_attention(
                        query=query,
                        key=key,
                        value=value,
                        attn_bias=bias,
                        p=dropout_p,
                        op=(
                            xformers.ops.fmha.cutlass.FwOp,
                            xformers.ops.fmha.cutlass.BwOp,
                        ),
                    )

                    test_grad = torch.ones_like(out)

                    results.append(
                        benchmark.Timer(
                            stmt="ref_attention(query, key, value, bias, rand_uniform=None, p=dropout_p)",
                            setup="",
                            globals={"ref_attention": ref_attention, **globals},
                            num_threads=1,
                            label="attn",
                            sub_label=sub_label,
                            description="reference",
                        ).blocked_autorange(min_run_time=1)
                    )
                    results.append(
                        benchmark.Timer(
                            stmt="out_ref.backward(test_grad, retain_graph=True)",
                            setup="",
                            globals={"out_ref": out_ref, "test_grad": test_grad},
                            num_threads=1,
                            label="attn-bwd",
                            sub_label=sub_label,
                            description="reference",
                        ).blocked_autorange(min_run_time=1)
                    )
                    results.append(
                        benchmark.Timer(
                            stmt="xformers.ops.memory_efficient_attention(query, key, value, bias, dropout_p, op=(xformers.ops.fmha.cutlass.FwOp, xformers.ops.fmha.cutlass.BwOp))",
                            setup="import xformers",
                            globals=globals,
                            num_threads=1,
                            label="attn",
                            sub_label=sub_label,
                            description="cutlass",
                        ).blocked_autorange(min_run_time=1)
                    )
                    results.append(
                        benchmark.Timer(
                            stmt="out.backward(test_grad, retain_graph=True)",
                            setup="",
                            globals={"out": out, "test_grad": test_grad},
                            num_threads=1,
                            label="attn-bwd",
                            sub_label=sub_label,
                            description="cutlass",
                        ).blocked_autorange(min_run_time=1)
                    )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    parser = ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["benchmark", "profile", "test"])
    args = parser.parse_args()

    if args.mode == "benchmark":
        cfgs = [
            (8, 512, 64, 128, torch.float16),
            (8, 1024, 64, 128, torch.float16),
        ]
        run_benchmark(cfgs)
    elif args.mode == "profile":
        run_once()
    elif args.mode == "test":
        test_attn(torch.float16, 0.0)
