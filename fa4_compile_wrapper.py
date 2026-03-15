from __future__ import annotations

import sys
import types
from typing import Callable, Optional

sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")
sys.modules["flash_attn_3_cuda"] = types.ModuleType("flash_attn_3_cuda")

import torch
from flash_attn.cute.interface import _flash_attn_bwd as _fa4_flash_attn_bwd
from flash_attn.cute.interface import flash_attn_varlen_func as _fa4_flash_attn_varlen_func

__all__ = ["flash_attn_varlen_func"]

_FWD_OP = "fa4_compile_wrapper::_flash_attn_varlen_with_lse"
_BWD_OP = "fa4_compile_wrapper::_flash_attn_varlen_backward"


def _fake_lse(q: torch.Tensor, cu_seqlens_q: Optional[torch.Tensor]) -> torch.Tensor:
    num_heads = q.shape[-2]
    if cu_seqlens_q is None:
        if q.dim() != 4:
            raise ValueError(
                "Expected q with shape (batch, seqlen, num_heads, head_dim) when cu_seqlens_q is None"
            )
        batch_size, seqlen_q = q.shape[:2]
        return torch.empty(
            (batch_size, num_heads, seqlen_q),
            dtype=torch.float32,
            device=q.device,
        )
    return torch.empty((num_heads, q.shape[0]), dtype=torch.float32, device=q.device)


def _validate_supported_args(
    page_table: Optional[torch.Tensor],
    learnable_sink: Optional[torch.Tensor],
    softcap: float,
    score_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
) -> None:
    if page_table is not None:
        raise NotImplementedError("page_table is not supported by this wrapper")
    if learnable_sink is not None:
        raise NotImplementedError("learnable_sink is not supported by this wrapper")
    if softcap != 0.0:
        raise NotImplementedError("softcap != 0.0 is not supported by this wrapper")
    if score_mod is not None:
        raise NotImplementedError("score_mod is not supported by this wrapper")
    if aux_tensors is not None:
        raise NotImplementedError("aux_tensors is not supported by this wrapper")


def _normalize_window_size(
    window_size: tuple[Optional[int], Optional[int]],
) -> tuple[Optional[int], Optional[int]]:
    if len(window_size) != 2:
        raise ValueError("window_size must be a 2-tuple of (left, right)")
    return window_size


@torch.library.custom_op(_FWD_OP, mutates_args=(), device_types="cuda")
def _flash_attn_varlen_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    num_splits: int,
    pack_gqa: Optional[bool],
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fa4_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(window_size_left, window_size_right),
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        return_lse=True,
    )


@_flash_attn_varlen_with_lse.register_fake
def _flash_attn_varlen_with_lse_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    num_splits: int,
    pack_gqa: Optional[bool],
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        k,
        v,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
    )
    return torch.empty_like(q), _fake_lse(q, cu_seqlens_q)


@torch.library.custom_op(_BWD_OP, mutates_args=(), device_types="cuda")
def _flash_attn_varlen_backward(
    dout: Optional[torch.Tensor],
    dlse: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if softcap != 0.0:
        raise NotImplementedError("softcap != 0.0 is not supported by this wrapper")
    if dout is None:
        dout = torch.zeros_like(out)
    return _fa4_flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        deterministic=deterministic,
        dlse=dlse,
    )


@_flash_attn_varlen_backward.register_fake
def _flash_attn_varlen_backward_fake(
    dout: Optional[torch.Tensor],
    dlse: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        dout,
        dlse,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
    )
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _flash_attn_varlen_setup_context(ctx, inputs, output) -> None:
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
    ) = inputs
    out, lse = output
    del num_splits, pack_gqa
    ctx.save_for_backward(
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
    )
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right
    ctx.softcap = softcap
    ctx.deterministic = deterministic


def _flash_attn_varlen_autograd(
    ctx,
    grad_out: Optional[torch.Tensor],
    grad_lse: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], ...]:
    (
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
    ) = ctx.saved_tensors
    dq, dk, dv = _flash_attn_varlen_backward(
        grad_out,
        grad_lse,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
        ctx.softmax_scale,
        ctx.causal,
        ctx.window_size_left,
        ctx.window_size_right,
        ctx.softcap,
        ctx.deterministic,
    )
    return (dq, dk, dv) + (None,) * 14


_flash_attn_varlen_with_lse.register_autograd(
    _flash_attn_varlen_autograd,
    setup_context=_flash_attn_varlen_setup_context,
)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    return_lse: bool = False,
):
    _validate_supported_args(page_table, learnable_sink, softcap, score_mod, aux_tensors)
    window_size_left, window_size_right = _normalize_window_size(window_size)
    out, lse = _flash_attn_varlen_with_lse(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
    )
    return (out, lse) if return_lse else out
