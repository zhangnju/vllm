# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OCP MXFP4 (e2m1 + e8m0 per 32) MoE experts on radeon_ops (RDNA4, gfx1201): W4A16 -- bf16 activations, the checkpoint's packed weights read as is.

torch.ops.radeon.moe_ffn_mxfp4 runs decode-sized batches (M <= VLLM_ROCM_RADEON_MXFP4_DECODE_MAX_M, default 48) on the fused MXFP4 decode GEMV, up to
VLLM_ROCM_RADEON_MXFP4_DG_MAX_M (default 896) tokens on the grouped weight-streaming WMMA kernels (16-row blocks) and larger ones on the large-M bf16-WMMA kernel
(gate-up + SiLU, down, top-k reduce); weights [E, 2N, K/2] / [E, K, N/2] uint8 (gate rows first), scales e8m0
[E, 2N, K/32] / [E, K, N/32].  Enabled with VLLM_ROCM_RADEON_MOE=1.  Checkpoints that also quantise activations (Quark w_mxfp4_a_mxfp4, W4A4) run with
unquantised bf16 activations here: the weights are exact, the activations more precise than the checkpoint's emulated MXFP4.
"""

import functools
import os

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey, kMxfp4Static
from vllm.platforms import current_platform

# R9700, E 256 / H 2048 / inter 512 (one MoE layer): GEMV < grouped from ~48 tokens, grouped < large-M from ~1024
_DECODE_MAX_M = int(os.environ.get("VLLM_ROCM_RADEON_MXFP4_DECODE_MAX_M", "48"))
_DG_MAX_M = int(os.environ.get("VLLM_ROCM_RADEON_MXFP4_DG_MAX_M", "896"))


@functools.cache
def _radeon_mxfp4_available() -> bool:
    try:
        from radeon_ops.torch_ops import load_radeon_torch_ops

        load_radeon_torch_ops()
        return hasattr(torch.ops.radeon, "moe_ffn_mxfp4")
    except Exception:
        return False


def radeon_mxfp4_kernel_available() -> bool:
    if not current_platform.is_rocm() or not envs.VLLM_ROCM_RADEON_MOE:
        return False
    from vllm.platforms.rocm import on_gfx1201

    return on_gfx1201() and _radeon_mxfp4_available()


class RadeonMxfp4Experts(mk.FusedMoEExpertsModular):
    """OCP MXFP4 W4A16 experts on radeon_ops (see module docstring)."""

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
        if hidden % 128 != 0 or inter % 64 != 0:
            return False, f"hidden ({hidden}) must be a multiple of 128 and intermediate ({inter}) of 64"
        if moe_config.num_experts > 512:
            return False, "at most 512 experts"
        return True, None

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return None   # bf16 activations (W4A16)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return radeon_mxfp4_kernel_available()

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True   # radeon_ops batch-invariant mode (on with VLLM_BATCH_INVARIANT): moe_ffn_mxfp4 skips its decode GEMV; the grouped and large-M kernels are bit-identical

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
        return weight_key == kMxfp4Static   # any checkpoint activation scheme: run as W4A16

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
        # w1 [E, 2N, K/2] and w2 [E, K, N/2] packed e2m1
        assert w1.dim() == 3 and w2.dim() == 3 and a1.dim() == 2
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
        return ((1,), (1,), (M, K))   # the radeon op keeps its intermediates in its own persistent scratch

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
        assert expert_map is None, "RadeonMxfp4Experts does not support expert parallelism"
        assert not apply_router_weight_on_input, "router weight on input is not supported"
        assert hidden_states.dtype == torch.bfloat16, f"bf16 activations expected, got {hidden_states.dtype}"
        # vLLM marks unrouted slots with expert id -1 (padding / warm-up batches): every radeon path skips ids < 0 in-kernel
        out = torch.ops.radeon.moe_ffn_mxfp4(
            hidden_states, w1.view(torch.uint8), self.w1_scale.view(torch.uint8), w2.view(torch.uint8), self.w2_scale.view(torch.uint8),
            topk_ids, topk_weights, 0, _DECODE_MAX_M, _DG_MAX_M,
        )
        output.copy_(out)
