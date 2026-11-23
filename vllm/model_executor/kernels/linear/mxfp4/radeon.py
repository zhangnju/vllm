# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 linear kernel for RDNA4 (gfx1201) and RDNA3 (gfx11) on radeon_ops: W4A16 -- bf16 activations, the checkpoint's packed weights ([N, K/2] uint8 E2M1 codes,
[N, K/32] uint8 E8M0 scales) read as is.

torch.ops.radeon.linear_mxfp4 routes on the number of activation rows M:
  M <= VLLM_ROCM_RADEON_MXFP4_LINEAR_GEMV_MAX_M (4; 8 on gfx11):  streaming GEMV (one MX group per 16-byte load, decoded once per token batch)
  M <= VLLM_ROCM_RADEON_MXFP4_LINEAR_DG_MAX_M (16; 32 on gfx11):  weight-streaming bf16 WMMA (16-token blocks)
  M <= VLLM_ROCM_RADEON_MXFP4_LINEAR_BIG_MAX_M (512): LDS-tiled bf16-WMMA kernel with in-register dequant
  otherwise (prefill):                                 dequantise into a persistent bf16 scratch copy + hipBLASLt
Enabled with VLLM_ROCM_RADEON_MXFP4_LINEAR=1.  W4A4 checkpoints (Quark w_mxfp4_a_mxfp4) run with unquantised bf16 activations, as the radeon MXFP4 MoE path.
"""

import functools
import os

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

# R9700, Qwen3.8-27B shapes (cold weights): GEMV 540-600 GB/s at M <= 4; WMMA streaming best at M 8-16; LDS-tiled to ~512; dequant + hipBLASLt above.
# W7900 (gfx11, radeon_ops >= 5dbce35): GEMV best to M 8, WMMA streaming at M 16-32 (1.4-2.1x the LDS-tiled kernel at 16), LDS-tiled from 64.
def _on_gfx11() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx11

    return on_gfx11()


_GEMV_MAX_M = int(os.environ.get("VLLM_ROCM_RADEON_MXFP4_LINEAR_GEMV_MAX_M", "8" if _on_gfx11() else "4"))
_DG_MAX_M = int(os.environ.get("VLLM_ROCM_RADEON_MXFP4_LINEAR_DG_MAX_M", "32" if _on_gfx11() else "16"))
_BIG_MAX_M = int(os.environ.get("VLLM_ROCM_RADEON_MXFP4_LINEAR_BIG_MAX_M", "512"))


@functools.cache
def _radeon_linear_mxfp4_available() -> bool:
    try:
        from radeon_ops.torch_ops import load_radeon_torch_ops

        load_radeon_torch_ops()
        return hasattr(torch.ops.radeon, "linear_mxfp4")
    except Exception:
        return False


def _radeon_mxfp4_linear_impl(x_2d: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    out = torch.ops.radeon.linear_mxfp4(x_2d, weight, weight_scale, _GEMV_MAX_M, _DG_MAX_M, _BIG_MAX_M)
    if bias is not None:
        out.add_(bias)
    return out


def _radeon_mxfp4_linear_fake(x_2d: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return torch.empty((x_2d.shape[0], weight.shape[0]), dtype=x_2d.dtype, device=x_2d.device)


direct_register_custom_op(
    op_name="radeon_mxfp4_linear_apply",
    op_func=_radeon_mxfp4_linear_impl,
    mutates_args=[],
    fake_impl=_radeon_mxfp4_linear_fake,
)


class RadeonMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 W4A16 linear on radeon_ops for RDNA4 / RDNA3 (see module docstring)."""

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "only targets ROCm"
        if os.environ.get("VLLM_ROCM_RADEON_MXFP4_LINEAR", "0") != "1":
            return False, "VLLM_ROCM_RADEON_MXFP4_LINEAR is not set"
        from vllm.platforms.rocm import on_gfx11, on_gfx1201

        if not (on_gfx1201() or on_gfx11()):
            return False, "requires RDNA4 (gfx1201) or RDNA3 (gfx11)"
        if not _radeon_linear_mxfp4_available():
            return False, "radeon_ops torch.ops.radeon.linear_mxfp4 is not available"
        return True, None

    @classmethod
    def _supports_quant_scheme(cls, weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
        return weight_key == kMxfp4Static and activation_key in (None, kMxfp4Dynamic)

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, weight_scale = layer.weight.data, layer.weight_scale.data
        N, k_half = weight.shape
        K = 2 * k_half
        if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8 or K % 128 != 0 or N % 8 != 0 or weight_scale.shape != (N, K // 32):
            raise ValueError(
                "RadeonMxFp4LinearKernel expects uint8 weight [N, K/2] and uint8 E8M0 weight_scale [N, K/32] with K % 128 == 0 and N % 8 == 0, got "
                f"{weight.dtype} {tuple(weight.shape)} and {weight_scale.dtype} {tuple(weight_scale.shape)}"
            )
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.contiguous(), requires_grad=False)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.dtype != torch.bfloat16:
            out = torch.ops.vllm.radeon_mxfp4_linear_apply(x_2d.to(torch.bfloat16), layer.weight, layer.weight_scale, None).to(x.dtype)
            if bias is not None:
                out.add_(bias)
        else:
            out = torch.ops.vllm.radeon_mxfp4_linear_apply(x_2d, layer.weight, layer.weight_scale, bias)
        return out.reshape(*x.shape[:-1], out.shape[-1])
