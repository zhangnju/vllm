# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 fused MoE experts on the Radeon packed-weight HIP kernel (RDNA4, gfx1200/gfx1201).

Both GEMMs run on expert weights that were re-laid ONCE, in place, into the WMMA fragment order
(`moe_pack_bf16_rdna4`, called from `convert_to_unquantized_kernel_format`): a wave load of 512 contiguous bytes
lands directly in the matrix fragment. GEMM 1 fuses gate+up+SiLU*mul and writes the bf16 activation directly
(no [M, topk, 2N] intermediate, no activation launch); GEMM 2 applies the router weight in its epilogue. The
token/expert blocking is vLLM's own `moe_align_block_size` with block size 16.

Enabled with VLLM_ROCM_RADEON_MOE=1 on gfx1200/gfx1201 for unquantized bf16 MoE with hidden size and per-partition
intermediate size multiples of 128, gated SiLU, no expert parallelism.
"""

import torch

import vllm._custom_ops as ops
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
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform

_BLOCK = 16


def radeon_moe_kernel_available() -> bool:
    """Whether the packed bf16 MoE kernel is built into this binary and runs on this GPU."""
    if not current_platform.is_rocm() or not envs.VLLM_ROCM_RADEON_MOE:
        return False
    from vllm.platforms.rocm import on_gfx1201

    return (
        on_gfx1201()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_packed_gemm_bf16_rdna4")
    )


class RadeonPackedExperts(mk.FusedMoEExpertsModular):
    """Unquantized bf16 experts on the packed-weight WMMA kernel (see module docstring)."""

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
        if moe_config.in_dtype != torch.bfloat16:
            return False, "bf16 activations only"
        hidden = moe_config.hidden_dim
        inter = moe_config.intermediate_size_per_partition
        if hidden % 128 != 0 or inter % 128 != 0:
            return False, f"hidden ({hidden}) and intermediate ({inter}) must be multiples of 128"
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
        return radeon_moe_kernel_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
        return weight_key is None and activation_key is None

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
        assert expert_map is None, "RadeonPackedExperts does not support expert parallelism"
        assert not apply_router_weight_on_input, "router weight on input is not supported"
        assert hidden_states.is_contiguous() and hidden_states.dtype == torch.bfloat16

        E, M, N2, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        inter = N2 // 2
        if global_num_experts == -1:
            global_num_experts = E
        num_slots = M * top_k

        sorted_ids, expert_ids, num_post = moe_align_block_size(
            topk_ids, _BLOCK, global_num_experts, None
        )
        act = workspace13.view(-1)[: num_slots * inter].view(num_slots, inter)
        ops.moe_packed_gemm_bf16_rdna4(
            hidden_states, w1, act, None, sorted_ids, expert_ids, num_post, top_k, num_slots, True
        )
        down = workspace2.view(-1)[: num_slots * K].view(num_slots, K)
        tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
        ops.moe_packed_gemm_bf16_rdna4(
            act, w2, down, tw, sorted_ids, expert_ids, num_post, 1, num_slots, False
        )
        ops.moe_sum(down.view(M, top_k, K), output)
