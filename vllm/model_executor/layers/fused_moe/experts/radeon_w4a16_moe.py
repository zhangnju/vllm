# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 (GPTQ symmetric int4, group size 128) MoE experts on the Radeon int8-WMMA kernel (RDNA4, gfx1200/gfx1201).

Weights are repacked once at load into the WMMA fragment order (see csrc/rocm/moe_w4a16_rdna4.cu); the bf16 activations are
quantised to int8 per 128-wide K group and multiplied with the unsigned nibbles on the int8 WMMA, the weight and activation
group scales being applied once per group. GEMM 1 fuses gate+up+SiLU*mul and writes the bf16 activation directly, GEMM 2 applies
the router weight in its epilogue; blocking is vLLM's own moe_align_block_size (block size 16).

Enabled with VLLM_ROCM_RADEON_MOE=1 on gfx1201 for symmetric, group-128 int4 checkpoints (asymmetric ones -- per-group zero points, e.g.
Quark UINT4 -- with VLLM_ROCM_RADEON_RQ=1: rq's int4 kernels take the zero points), hidden and per-partition
intermediate sizes that are multiples of 128, gated SiLU, no expert parallelism.
"""

import os

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts import radeon_rq
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey, kInt4Static, kInt4StaticAsym
from vllm.platforms import current_platform

_BLOCK = 16
_GROUP = 128


def radeon_w4a16_kernel_available() -> bool:
    if not current_platform.is_rocm() or not envs.VLLM_ROCM_RADEON_MOE:
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


_TRACE = [] if os.environ.get("VLLM_ROCM_RADEON_TRACE") else None


def save_routing_trace(path: str, num_layers: int, plen: int) -> int:
    """Write the recorded routing as ids [num_layers, sequences, positions, top_k] (int16; one prefill call per layer per sequence, eager, prompts of plen tokens;
    calls of other lengths -- the engine's profiling / warm-up runs -- are dropped)."""
    import numpy as np
    calls = [t.numpy() for t in (_TRACE or []) if t.shape[0] == plen]
    nseq = len(calls) // num_layers
    ids = np.stack([np.stack(calls[s * num_layers:(s + 1) * num_layers]) for s in range(nseq)], axis=1)   # [L, S, P, K]
    np.savez_compressed(path, ids=ids)
    return nseq


class RadeonW4A16Experts(mk.FusedMoEExpertsModular):
    """Symmetric g128 int4 experts on the Radeon int8-WMMA kernel (see module docstring)."""

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        hidden = moe_config.hidden_dim
        inter = moe_config.intermediate_size_per_partition
        if hidden % _GROUP != 0 or inter % _GROUP != 0:
            return False, f"hidden ({hidden}) and intermediate ({inter}) must be multiples of {_GROUP}"
        if moe_config.num_experts >= 1024:
            return False, "moe_align_block_size requires fewer than 1024 experts"
        return True, None

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return radeon_w4a16_kernel_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
        # asymmetric int4 (per-group zero points, e.g. Quark UINT4) on the rq path only: rq's int4 kernels take the zero points (`wzp`)
        return activation_key is None and (weight_key == kInt4Static or (weight_key == kInt4StaticAsym and radeon_rq.enabled()))

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_ep
            or moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        # w1 is the packed uint8 [E, 2N, K / 2] and w2 [E, K, N / 2]
        assert w1.dim() == 3 and w2.dim() == 3 and a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        return w1.size(0), a1.size(0), w1.size(1), a1.size(-1), topk_ids.size(1)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        act_dim = self.adjust_N_for_activation(N, activation)
        return ((M, topk, max(act_dim, K)), (M, topk, max(N, K)), (M, K))

    @staticmethod
    def _quant(x: torch.Tensor):
        pq = radeon_rq.prequant_take(x)
        if pq is not None:
            return pq[0].view(torch.int8), pq[1], pq[2]
        rows, K = x.shape
        xq = torch.empty(rows, K, dtype=torch.int8, device=x.device)
        xs = torch.empty(rows, K // _GROUP, dtype=torch.float32, device=x.device)
        xsum = torch.empty(rows, K // _GROUP, dtype=torch.int32, device=x.device)
        ops.moe_quant_x_g128_rdna4(x, xq, xs, xsum)
        return xq, xs, xsum

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert expert_map is None, "RadeonW4A16Experts does not support expert parallelism"
        assert not apply_router_weight_on_input, "router weight on input is not supported"
        assert hidden_states.is_contiguous() and hidden_states.dtype == torch.bfloat16
        assert self.w1_scale is not None and self.w2_scale is not None

        E, M, N2, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        inter = N2 // 2
        if global_num_experts == -1:
            global_num_experts = E
        num_slots = M * top_k

        if _TRACE is not None and M > 1:
            _TRACE.append(topk_ids.to(torch.int16).cpu())   # routing trace (VLLM_ROCM_RADEON_TRACE): one entry per MoE layer call, in layer order
        sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, _BLOCK, global_num_experts, None)
        if radeon_rq.enabled():
            self._topk_ids = topk_ids
            self._apply_rq(output, hidden_states, w1, w2, topk_weights, sorted_ids, expert_ids, num_post, inter, K, top_k, M, num_slots, workspace13, workspace2)
            return
        xq, xs, xsum = self._quant(hidden_states)
        act = workspace13.view(-1)[: num_slots * inter].view(num_slots, inter)
        # GEMM 1 writes the int8 activation (+ per-128 scales and sums) straight from its epilogue
        aq = torch.empty(num_slots, inter, dtype=torch.int8, device=hidden_states.device)
        a_s = torch.empty(num_slots, inter // _GROUP, dtype=torch.float32, device=hidden_states.device)
        a_sum = torch.empty(num_slots, inter // _GROUP, dtype=torch.int32, device=hidden_states.device)
        ops.moe_w4a16_gemm_rdna4(
            xq, xs, xsum, w1, self.w1_scale, act, None, sorted_ids, expert_ids, num_post, top_k, num_slots, True, aq, a_s, a_sum
        )
        down = workspace2.view(-1)[: num_slots * K].view(num_slots, K)
        tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
        ops.moe_w4a16_gemm_rdna4(
            aq, a_s, a_sum, w2, self.w2_scale, down, tw, sorted_ids, expert_ids, num_post, 1, num_slots, False
        )
        ops.moe_sum(down.view(M, top_k, K), output)

    def _apply_rq(self, output, hidden_states, w1, w2, topk_weights, sorted_ids, expert_ids, num_post, inter, K, top_k, M, num_slots, workspace13, workspace2):
        """Same pipeline on the rq core: quantise x, GEMM 1 (gate/up/SiLU + fused activation quantisation), GEMM 2 (router weight), sum over top-k."""
        rq = radeon_rq.rq_ops()
        act = radeon_rq.act_id()
        dev = hidden_states.device
        pq = radeon_rq.prequant_take(hidden_states) if act == 0 else None   # int8 activation: produced by the fused all-reduce + RMSNorm
        xq, xs, xsum = pq if pq is not None else rq.quant_x(hidden_states, act)
        aq = torch.empty(num_slots, inter * radeon_rq.act_bits(act) // 8, dtype=torch.uint8, device=dev)
        a_s = torch.empty(num_slots, inter // _GROUP, dtype=torch.float32, device=dev)
        a_sum = torch.empty(num_slots, inter // _GROUP, dtype=radeon_rq.act_sum_dtype(act), device=dev)
        act_out = workspace13.view(-1)[: num_slots * inter].view(num_slots, inter)   # shape donor only (the fused epilogue writes aq)
        E = w1.size(0)
        z1, z2 = self.w1_zp, self.w2_zp   # asymmetric checkpoints: uint8 zero points in the scale tile layout (int_wna16 RADEON conversion), else None
        up_split, up_lds, up_nw, down_lds, down_nw = (radeon_rq.moe_plan(w1.size(1), K, inter, M, E, top_k, act) if radeon_rq.plan_enabled()
                                                       else (False, False, 8, False, 8))
        if up_split:    # measured faster: the gate/up/SiLU GEMM with the bf16 GLU epilogue (streaming nw or the LDS kernel) + quant_x, bit-identical to the fused epilogue
            act_bf = workspace13.view(-1)[: num_slots * inter].view(num_slots, inter)
            if up_lds:
                sid64, eid64, np64 = moe_align_block_size(self._topk_ids, 64, E, None)
                rq.gemm_big(xq, xs, xsum, w1.view(torch.uint8), self.w1_scale, act_bf, None, sid64, eid64, np64, top_k, num_slots, radeon_rq.EPI_GLU, z1)
            else:
                rq.gemm(xq, xs, xsum, w1.view(torch.uint8), self.w1_scale, act_bf, None, sorted_ids, expert_ids, num_post, top_k, num_slots, radeon_rq.EPI_GLU,
                        None, None, None, up_nw, act, z1)
            aq, a_s, a_sum = rq.quant_x(act_bf, act)
        else:
            rq.gemm(xq, xs, xsum, w1.view(torch.uint8), self.w1_scale, act_out, None, sorted_ids, expert_ids, num_post,
                    top_k, num_slots, radeon_rq.EPI_GLUQ, aq, a_s, a_sum, 8, act, z1)
        down = workspace2.view(-1)[: num_slots * K].view(num_slots, K)
        tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
        if down_lds:
            sid128, eid128, np128 = moe_align_block_size(self._topk_ids, 128, E, None)
            rq.gemm_big(aq, a_s, a_sum, w2.view(torch.uint8), self.w2_scale, down, tw, sid128, eid128, np128, 1, num_slots, radeon_rq.EPI_PLAIN, z2)
        else:
            rq.gemm(aq, a_s, a_sum, w2.view(torch.uint8), self.w2_scale, down, tw, sorted_ids, expert_ids, num_post, 1, num_slots, radeon_rq.EPI_PLAIN, None,
                    None, None, down_nw, act, z2)
        ops.moe_sum(down.view(M, top_k, K), output)
