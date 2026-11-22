# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-scaled FP8 (e4m3, 128x128 weight blocks, dynamic per-128 activation scales) MoE experts on the Radeon native
fp8-WMMA kernel (RDNA4, gfx1200/gfx1201), see csrc/rocm/moe_fp8_rdna4.cu.

Expert weights are re-laid once, at load time, into the WMMA fragment order. The block-quantised input arrives already in
fp8 from the prepare step; GEMM 1 fuses gate+up+SiLU*mul and writes the e4m3 activation of GEMM 2 (and its group scales)
straight from its epilogue, GEMM 2 applies the router weight. Blocking is vLLM's moe_align_block_size (block 16).

Enabled with VLLM_ROCM_RADEON_MOE=1 on gfx1201 for hidden / per-partition intermediate sizes that are multiples of 128,
gated SiLU, no expert parallelism.

VLLM_ROCM_RADEON_FP8_FFN=1 additionally runs batches with at least VLLM_ROCM_RADEON_FP8_FFN_MIN_ROWS (default 256) routed rows per expert (prefill)
on the radeon_ops large-M W8A8 expert FFN (torch.ops.radeon.moe_ffn_w8a8_packed: 128-row blocks, reads the same packed weights); never during
CUDA-graph capture (decode keeps the kernel above).  Qwen3-30B-A3B-FP8 TP=2 prefill (8192-token forwards): +3.2%, PPL within noise.
"""

import functools
import os

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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

_BLOCK = 16
_GROUP = 128
_FFN = os.environ.get("VLLM_ROCM_RADEON_FP8_FFN", "0").lower() in ("1", "true")
_FFN_MIN_ROWS = int(os.environ.get("VLLM_ROCM_RADEON_FP8_FFN_MIN_ROWS", "256"))   # below ~256 rows/expert the kernel above is as fast
_FFN_BM = 128


@functools.cache
def _radeon_ffn_available() -> bool:
    try:
        from radeon_ops.torch_ops import load_radeon_torch_ops

        load_radeon_torch_ops()
        return hasattr(torch.ops.radeon, "moe_ffn_w8a8_packed")
    except Exception:
        return False


def radeon_fp8_kernel_available() -> bool:
    if not current_platform.is_rocm() or not envs.VLLM_ROCM_RADEON_MOE:
        return False
    from vllm.platforms.rocm import on_gfx1201

    return on_gfx1201() and hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "moe_fp8_gemm_rdna4")


class RadeonFp8Experts(mk.FusedMoEExpertsModular):
    """Block-scaled fp8 experts on the Radeon fp8-WMMA kernel (see module docstring)."""

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
        return radeon_fp8_kernel_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
        return weight_key == kFp8Static128BlockSym and activation_key == kFp8Dynamic128Sym

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
        # w1 [E, 2N, K] and w2 [E, K, N] in the fragment order (same shapes as the checkpoint)
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
        assert expert_map is None, "RadeonFp8Experts does not support expert parallelism"
        assert not apply_router_weight_on_input, "router weight on input is not supported"
        assert self.w1_scale is not None and self.w2_scale is not None
        assert a1q_scale is not None and hidden_states.dtype == torch.float8_e4m3fn

        E, M, N2, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        inter = N2 // 2
        if global_num_experts == -1:
            global_num_experts = E
        num_slots = M * top_k
        dev = hidden_states.device

        xq = hidden_states.contiguous().view(torch.uint8)
        xs = a1q_scale.to(torch.float32).contiguous()
        if (
            _FFN
            and num_slots >= _FFN_MIN_ROWS * global_num_experts
            and not torch.cuda.is_current_stream_capturing()
            and _radeon_ffn_available()
        ):
            sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, _FFN_BM, global_num_experts, None)
            down = workspace2.view(-1)[: num_slots * K].view(num_slots, K)
            tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
            torch.ops.radeon.moe_ffn_w8a8_packed(
                xq, xs, w1.view(torch.uint8), self.w1_scale, w2.view(torch.uint8), self.w2_scale, tw, sorted_ids, expert_ids, num_post, top_k, _FFN_BM, down
            )
            ops.moe_sum(down.view(M, top_k, K), output)
            return
        sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, _BLOCK, global_num_experts, None)
        act = workspace13.view(-1)[: num_slots * inter].view(num_slots, inter)  # shape donor only, aq carries the data
        aq = torch.empty(num_slots, inter, dtype=torch.uint8, device=dev)
        a_s = torch.empty(num_slots, inter // _GROUP, dtype=torch.float32, device=dev)
        ops.moe_fp8_gemm_rdna4(
            xq, xs, w1.view(torch.uint8), self.w1_scale, act, None, sorted_ids, expert_ids, num_post, top_k, num_slots, True, aq, a_s
        )
        down = workspace2.view(-1)[: num_slots * K].view(num_slots, K)
        tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
        ops.moe_fp8_gemm_rdna4(
            aq, a_s, w2.view(torch.uint8), self.w2_scale, down, tw, sorted_ids, expert_ids, num_post, 1, num_slots, False
        )
        ops.moe_sum(down.view(M, top_k, K), output)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# W8A16 on RDNA3 (gfx11): no fp8 arithmetic there, so vLLM's Triton path runs the W8A8 scheme emulated (fp8 activations too) at ~1-27 ms per
# Qwen3.6 MoE layer on a W7900.  radeon_ops' W8A16 kernels keep the activations in bf16 and convert e4m3 in software inside the loads: the fused
# decode GEMV (gate-up + SiLU, down + weighted reduce) for small batches, the bf16-WMMA expert FFN (gate-up + SiLU, down into slot rows, token
# reduce) above.  W7900, E 256 H 2048 inter 512 top 8, cold weights: T 1 108 us vs 1037 (Triton W8A8), T 8 448 vs 3833, T 64 2337 vs 9763,
# T 1024 2950 vs 27074, T 4096 6522 vs 51056; error vs an fp32 reference 3-5e-3 vs 4.5e-2 (no activation quantisation).

_W8A16_GEMV_MAX_M = int(os.environ.get("VLLM_ROCM_RADEON_FP8_W8A16_GEMV_MAX_M", "32"))   # decode GEMV up to here (T 32 1674 us vs FFN 2060, T 64 2980 vs 2337)
_W8A16_BM64_ROWS = 96   # routed rows per expert from which the bm 64 FFN tiles beat bm 32 (T 4096: 6522 vs 7702 us; T 1024: 3027 vs 2950)


@functools.cache
def _radeon_w8a16_available() -> bool:
    try:
        from radeon_ops.backends.hip.native import moe as _m  # noqa: F401

        return True
    except Exception:
        return False


def radeon_fp8_w8a16_available() -> bool:
    if not current_platform.is_rocm() or not envs.VLLM_ROCM_RADEON_MOE:
        return False
    from vllm.platforms.rocm import on_gfx11

    return on_gfx11() and _radeon_w8a16_available()


class RadeonFp8W8A16Experts(mk.FusedMoEExpertsModular):
    """Block-scaled fp8 weights, bf16 activations (W8A16) on radeon_ops' gfx11 kernels (see the comment above)."""

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(cls, moe_config, weight_key, activation_key, activation_format)
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
        return radeon_fp8_w8a16_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def executed_activation_key(weight_key: QuantKey | None, activation_key: QuantKey | None) -> QuantKey | None:
        return None   # bf16 activations whatever the checkpoint's activation scheme

    @staticmethod
    def _supports_quant_scheme(weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
        return weight_key == kFp8Static128BlockSym and activation_key is None

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
        return ((1,), (1,), (M, K))   # the radeon_ops entry points keep their own intermediates

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
        from radeon_ops.backends.hip.native import moe as rmoe
        from radeon_ops.batch_invariant import enabled as batch_invariant

        assert expert_map is None, "RadeonFp8W8A16Experts does not support expert parallelism"
        assert not apply_router_weight_on_input, "router weight on input is not supported"
        assert self.w1_scale is not None and self.w2_scale is not None
        assert hidden_states.dtype == torch.bfloat16, "W8A16: bf16 activations"
        E, M, _, _, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts == -1:
            global_num_experts = E
        x = hidden_states.contiguous()
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        tw = topk_weights.to(torch.float32).contiguous()
        if M <= _W8A16_GEMV_MAX_M and not batch_invariant():   # the GEMV rounds differently from the WMMA FFN: not in batch-invariant mode
            out = rmoe.run_moe_decode_gemv_fp8(x, w1, self.w1_scale, w2, self.w2_scale, ids, tw)
        else:
            bm = 64 if M * top_k >= _W8A16_BM64_ROWS * global_num_experts and not batch_invariant() else 32
            out = rmoe.run_moe_prefill_ffn_fp8(x, w1, self.w1_scale, w2, self.w2_scale, ids, tw, top_k, bm=bm)
        output.copy_(out)
