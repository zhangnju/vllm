# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense W4A16 (GPTQ symmetric int4, group size 128) linear on the Radeon int8-WMMA kernel (RDNA4, gfx1200/gfx1201).

Small token counts (decode, M <= MAX_M) run the same fragment-ordered int4 x int8-WMMA kernel as the MoE backend
(csrc/rocm/moe_w4a16_rdna4.cu) with a single "expert"; larger M falls back to the Triton W4A16 GEMM, whose [K, N//8]
weight layout is kept alongside (the packed copy costs one more int4 copy of the layer weight).

Enabled with VLLM_ROCM_RADEON_LINEAR=1 for bf16 activations, K % 128 == 0, N % 16 == 0, symmetric weights.
"""

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.experts import radeon_rq
from vllm.model_executor.parameter import permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearLayerConfig
from .triton_w4a16 import TritonW4A16LinearKernel, _triton_w4a16_gemm_impl

_GROUP = 128
MAX_M = 64
_MAX_MB = MAX_M // 16


def _radeon_available() -> bool:
    if not current_platform.is_rocm() or not envs.VLLM_ROCM_RADEON_LINEAR:
        return False
    from vllm.platforms.rocm import on_gfx1201

    if radeon_rq.enabled():
        try:
            radeon_rq.rq_ops()
        except Exception:
            return False
        return on_gfx1201()
    return (
        on_gfx1201()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_w4a16_gemm_rdna4")
    )


def _pick_nw(n: int, m_blocks: int) -> int:
    # enough blocks to cover the CUs: each block streams nw * 16 weight rows
    for nw in (8, 4, 2):
        if (n // (nw * 16)) * m_blocks >= 128:
            return nw
    return 1


def _radeon_w4a16_gemm_impl(
    a: torch.Tensor,
    w: torch.Tensor,
    ws: torch.Tensor,
    ids: torch.Tensor,
    eids: torch.Tensor,
    ntpp: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    M, K = a.shape
    N = w.size(1)
    if M > MAX_M and radeon_rq.big_mode() and radeon_rq.enabled() and radeon_rq.act_id() == 0 and K % 128 == 0 and N % 16 == 0:
        rq = radeon_rq.rq_ops()
        if M >= radeon_rq.BIG_MIN_M:      # radeon_big: activations staged in LDS, 128-slot blocks of the single expert
            bm = 128
            nbl = (M + bm - 1) // bm
            sorted_ids = torch.arange(nbl * bm, dtype=torch.int32, device=a.device)   # ids >= M are the padding (invalid slots)
            eids = torch.zeros(nbl, dtype=torch.int32, device=a.device)
            ntpp = torch.full((1,), nbl * bm, dtype=torch.int32, device=a.device)
            xq, xs, xsum = rq.quant_x(a, 0)
            out = torch.empty(M, N, dtype=a.dtype, device=a.device)
            rq.gemm_big(xq, xs, xsum, w, ws, out, None, sorted_ids, eids, ntpp, 1, M, 0)
            return out
        if radeon_rq.big_mode() >= 2:      # mid sizes: the streaming rq kernel with 16-slot blocks
            mb = (M + 15) // 16
            sorted_ids = torch.arange(mb * 16, dtype=torch.int32, device=a.device)
            eids = torch.zeros(mb, dtype=torch.int32, device=a.device)
            ntpp = torch.full((1,), mb * 16, dtype=torch.int32, device=a.device)
            xq, xs, xsum = rq.quant_x(a, 0)
            out = torch.empty(M, N, dtype=a.dtype, device=a.device)
            rq.gemm(xq, xs, xsum, w, ws, out, None, sorted_ids, eids, ntpp, 1, M, radeon_rq.EPI_PLAIN, None, None, None, _pick_nw(N, mb), 0)
            return out
    if M > MAX_M:
        return _triton_w4a16_gemm_impl(a, b_q, scales, None, group_size, 8)
    mb = (M + 15) // 16
    if radeon_rq.enabled():
        rq = radeon_rq.rq_ops()
        act = radeon_rq.act_id()
        pq = radeon_rq.prequant_take(a) if act == 0 else None
        xq, xs, xsum = pq if pq is not None else rq.quant_x(a, act)
        out = torch.empty(M, N, dtype=a.dtype, device=a.device)
        rq.gemm(xq, xs, xsum, w, ws, out, None, ids, eids[:mb], ntpp[mb - 1 : mb], 1, M, radeon_rq.EPI_PLAIN, None, None, None, _pick_nw(N, mb), act)
        return out
    pq = radeon_rq.prequant_take(a)
    if pq is not None:
        xq, xs, xsum = pq[0].view(torch.int8), pq[1], pq[2]
    else:
        xq = torch.empty(M, K, dtype=torch.int8, device=a.device)
        xs = torch.empty(M, K // _GROUP, dtype=torch.float32, device=a.device)
        xsum = torch.empty(M, K // _GROUP, dtype=torch.int32, device=a.device)
        ops.moe_quant_x_g128_rdna4(a, xq, xs, xsum)
    out = torch.empty(M, N, dtype=a.dtype, device=a.device)
    ops.moe_w4a16_gemm_rdna4(
        xq, xs, xsum, w, ws, out, None, ids, eids[:mb], ntpp[mb - 1 : mb], 1, M, False, None, None, None, _pick_nw(N, mb)
    )
    return out


def _radeon_w4a16_gemm_fake(
    a, w, ws, ids, eids, ntpp, b_q, scales, group_size
) -> torch.Tensor:
    return torch.empty((a.size(0), w.size(1)), dtype=a.dtype, device=a.device)


direct_register_custom_op(
    op_name="radeon_w4a16_gemm",
    op_func=_radeon_w4a16_gemm_impl,
    mutates_args=[],
    fake_impl=_radeon_w4a16_gemm_fake,
)


class RadeonW4A16LinearKernel(TritonW4A16LinearKernel):
    """Radeon int8-WMMA W4A16 GEMM for decode-sized M, Triton W4A16 above (see module docstring)."""

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not _radeon_available():
            return False, "needs gfx1201 and VLLM_ROCM_RADEON_LINEAR=1"
        ok, reason = super().can_implement(c)
        if not ok:
            return ok, reason
        K, N = c.partition_weight_shape
        if c.weight_type != scalar_types.uint4b8 or c.zero_points:
            return False, "symmetric uint4b8 only"
        if c.group_size != _GROUP:
            return False, "group size 128 only"
        if c.act_type != torch.bfloat16:
            return False, "bf16 activations only"
        if K % _GROUP != 0 or N % 16 != 0:
            return False, "needs K % 128 == 0 and N % 16 == 0"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q = getattr(layer, self.w_q_name)
        w_s = getattr(layer, self.w_s_name)
        permute_param_layout_(w_q, input_dim=1, output_dim=0, packed_dim=1)
        permute_param_layout_(w_s, input_dim=1, output_dim=0)
        w = w_q.data.contiguous().view(torch.uint8).unsqueeze(0)  # [1, N, K/2]
        packed = torch.empty_like(w)
        if radeon_rq.enabled():
            packed = radeon_rq.rq_ops().pack_w4(w, radeon_rq.act_id())
        else:
            ops.moe_pack_w4_rdna4(w, packed)
        n, g = w_s.data.shape
        s = w_s.data.to(torch.bfloat16).reshape(n // 16, 16, g).permute(0, 2, 1).contiguous().unsqueeze(0)
        layer.register_buffer("radeon_w4_w", packed, persistent=False)
        layer.register_buffer("radeon_w4_s", s, persistent=False)
        dev = packed.device
        layer.register_buffer("radeon_w4_ids", torch.arange(MAX_M, dtype=torch.int32, device=dev), persistent=False)
        layer.register_buffer("radeon_w4_eids", torch.zeros(_MAX_MB, dtype=torch.int32, device=dev), persistent=False)
        layer.register_buffer(
            "radeon_w4_ntpp", (torch.arange(1, _MAX_MB + 1, dtype=torch.int32, device=dev) * 16), persistent=False
        )
        super().process_weights_after_loading(layer)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        c = self.config
        w_q, w_s, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.ops.vllm.radeon_w4a16_gemm(
            x_2d,
            layer.radeon_w4_w,
            layer.radeon_w4_s,
            layer.radeon_w4_ids,
            layer.radeon_w4_eids,
            layer.radeon_w4_ntpp,
            w_q,
            w_s,
            c.group_size,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (c.partition_weight_shape[1],))
