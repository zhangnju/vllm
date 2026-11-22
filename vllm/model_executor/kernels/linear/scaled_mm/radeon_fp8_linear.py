# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-scaled FP8 dense linear (e4m3 weights, 128 x 128 block scales) as W8A16 on radeon_ops, for RDNA3 (gfx11), which has no fp8 arithmetic.

vLLM's block-fp8 path quantises the activation to fp8 and runs the Triton W8A8 GEMM, all emulated on gfx11: 0.5-2.4 ms per linear on a W7900 even at
M = 1 (Qwen3.6-35B-A3B-FP8 dense shapes), which is nearly the whole decode step.  radeon_ops keeps the activation in bf16 and converts e4m3 in
software: the GEMV at M = 1 (12-66 us), the tiled bf16-WMMA GEMM above (bm 32, bm 64 from 256 rows: 6-12x the Triton path at M = 2048); error vs an
fp32 reference 2.7e-3 vs 2.6e-2 (no activation quantisation).

Enabled with VLLM_ROCM_RADEON_FP8_LINEAR=1 on gfx11.
"""

import functools
import os

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel, FP8ScaledMMLinearLayerConfig

_ENABLED = os.environ.get("VLLM_ROCM_RADEON_FP8_LINEAR", "0").lower() in ("1", "true")
_BM64_MIN_M = 256   # bm 64 tiles from here (M 256: 367 vs 478 us on 12288 x 2048), bm 32 below


@functools.cache
def _radeon_available() -> bool:
    try:
        from radeon_ops.backends.hip.native import moe as _m  # noqa: F401

        return True
    except Exception:
        return False


def _radeon_fp8_w8a16_mm_impl(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    from radeon_ops.backends.hip.native import moe as rmoe
    from radeon_ops.batch_invariant import enabled as batch_invariant

    M = x.size(0)
    if M == 1 and not batch_invariant():   # the GEMV rounds differently from the tiled kernel: not in batch-invariant mode
        return rmoe.run_dense_gemv_fp8(x, weight, weight_scale)
    return rmoe.run_dense_prefill_gemm_fp8(x, weight, weight_scale, bm=64 if M >= _BM64_MIN_M and not batch_invariant() else 32)


def _radeon_fp8_w8a16_mm_fake(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.size(0), weight.size(0)), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="radeon_fp8_w8a16_mm",
    op_func=_radeon_fp8_w8a16_mm_impl,
    mutates_args=[],
    fake_impl=_radeon_fp8_w8a16_mm_fake,
)


class RadeonFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """FP8 W8A16 block-quantised linear on radeon_ops (gfx11, see the module docstring)."""

    apply_input_quant = False   # bf16 activations

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not current_platform.is_rocm() or not _ENABLED:
            return False, "needs ROCm and VLLM_ROCM_RADEON_FP8_LINEAR=1"
        from vllm.platforms.rocm import on_gfx11

        if not on_gfx11():
            return False, "gfx11 (RDNA3) only"
        if not _radeon_available():
            return False, "radeon_ops not importable"
        return True, None

    @classmethod
    def executed_activation_key(cls, activation_quant_key: QuantKey | None) -> QuantKey | None:
        return None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        gs = config.weight_quant_key.scale.group_shape
        if gs.row != 128 or gs.col != 128:
            return False, f"128 x 128 weight blocks only, got {gs.row} x {gs.col}"
        if config.out_dtype != torch.bfloat16:
            return False, "bf16 output only"
        if config.weight_shape is not None and config.weight_shape[-1] % 128 != 0:
            return False, "K % 128 == 0"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        # the kernels read dense [N, K] rows: drop vLLM's ROCm fp8 row padding (K 2048 stored with a 2304-element row stride)
        if not params.weight.is_contiguous():
            replace_parameter(layer, params.WEIGHT, params.weight.data.contiguous())
        # radeon_ops takes bf16 block scales [N / 128, K / 128] (exact for the checkpoint's bf16 scales, a 2^-9 relative rounding of f32 ones)
        replace_parameter(layer, params.block_scale_attr, params.block_scale.data.to(torch.bfloat16).contiguous())

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        params = self._get_layer_params(layer)
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.radeon_fp8_w8a16_mm(x2 if x2.is_contiguous() else x2.contiguous(), params.weight, params.block_scale)
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], params.weight.shape[0])

    def apply_block_scaled_mm(self, A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.radeon_fp8_w8a16_mm(A, B, Bs)
