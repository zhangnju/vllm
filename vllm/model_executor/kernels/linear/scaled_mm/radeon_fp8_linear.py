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


def _batch_invariant() -> bool:
    from radeon_ops.batch_invariant import enabled

    return enabled()


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


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Per-token-activation x per-channel-weight fp8 linears (ptpc: Quark AttnFP8 / compressed-tensors channel-wise checkpoints) on RDNA4's native fp8 WMMA: radeon_ops'
# dense_gemm_fp8_token computes (A_fp8 @ W_fp8^T) * a_scale[m] * w_scale[n] (+ bias) with the scales, bias and cast as the kernel's epilogue.  vLLM's ROCm path without
# AITER is torch._scaled_mm rowwise (hipBLASLt).  R9700, Qwen3.5-9B dense shapes, op level: M = 1 on the W8A16 GEMV 1.2-2.6x (_scaled_mm + activation quant), M 96-256
# on the tiled kernel 1.03-2.26x, same error (1.7e-3); M 2-95 stays on _scaled_mm (a tie).  E2E Qwen3.5-9B online ptpc fp8 (quantization="fp8_per_channel"): bs 1
# 38.1 -> 50.1 tok/s, PPL 4.472 -> 4.466.
# Enabled with VLLM_ROCM_RADEON_FP8_LINEAR=1 on gfx1201.

from .ScaledMMLinearKernel import FP8ScaledMMLinearKernel  # noqa: E402

_PTPC_MIN_M = 96   # the tiled kernel from here (1.03-2.26x at M 96-192), torch._scaled_mm between the GEMV (M = 1) and here


def _radeon_fp8_ptpc_mm_impl(a: torch.Tensor, w_nk: torch.Tensor, a_scale: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    from radeon_ops.backends.hip.native.linear_q import run_dense_gemm_fp8_token

    if a.size(0) < _PTPC_MIN_M and not _batch_invariant():   # hipBLASLt (vLLM's RowWiseTorch path) ties the tiled kernel here: 0.90-1.25x over Qwen3.5-9B shapes at M 8-64, -3% E2E at bs 8 / 32
        out = torch._scaled_mm(a, w_nk.t(), scale_a=a_scale.reshape(-1, 1).float(), scale_b=w_scale.reshape(1, -1).float(), out_dtype=out_dtype)
        return out if bias is None else out + bias
    return run_dense_gemm_fp8_token(a, a_scale.reshape(-1).float(), w_nk, w_scale.reshape(-1).float(), bias=bias, out_dtype=out_dtype)


def _radeon_fp8_ptpc_mm_fake(a, w_nk, a_scale, w_scale, bias, out_dtype) -> torch.Tensor:
    return torch.empty((a.size(0), w_nk.size(0)), dtype=out_dtype, device=a.device)


direct_register_custom_op(
    op_name="radeon_fp8_ptpc_mm",
    op_func=_radeon_fp8_ptpc_mm_impl,
    mutates_args=[],
    fake_impl=_radeon_fp8_ptpc_mm_fake,
)


def _radeon_fp8_ptpc_gemv_impl(x: torch.Tensor, w_nk: torch.Tensor, ones_blk: torch.Tensor, w_scale_bf16: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    from radeon_ops.backends.hip.native import moe as rmoe

    out = rmoe.run_dense_gemv_fp8(x, w_nk, ones_blk) * w_scale_bf16
    return out if bias is None else out + bias


def _radeon_fp8_ptpc_gemv_fake(x, w_nk, ones_blk, w_scale_bf16, bias) -> torch.Tensor:
    return torch.empty((1, w_nk.size(0)), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="radeon_fp8_ptpc_gemv",
    op_func=_radeon_fp8_ptpc_gemv_impl,
    mutates_args=[],
    fake_impl=_radeon_fp8_ptpc_gemv_fake,
)


class RadeonPtpcFp8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """Per-token x per-channel W8A8 fp8 linear on radeon_ops' native fp8-WMMA GEMM (gfx1201, see the comment above)."""

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not current_platform.is_rocm() or not _ENABLED:
            return False, "needs ROCm and VLLM_ROCM_RADEON_FP8_LINEAR=1"
        from vllm.platforms.rocm import on_gfx1201

        if not on_gfx1201():
            return False, "gfx1201 (native fp8 WMMA) only"
        if not _radeon_available():
            return False, "radeon_ops not importable"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        a_gs, w_gs = c.activation_quant_key.scale.group_shape, c.weight_quant_key.scale.group_shape
        if a_gs.is_per_tensor() or w_gs.is_per_tensor() or a_gs.is_per_group() or w_gs.is_per_group():
            return False, "per-token activation and per-channel weight scales only"
        if c.activation_quant_key.scale.static:
            return False, "dynamic per-token activation scales only"
        if c.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "bf16 / fp16 output only"
        n, k = c.weight_shape   # (out, in)
        if k % 64 or n % 8:
            return False, "needs K % 64 == 0 and N % 8 == 0"
        return True, None

    def get_output_padding(self) -> int | None:
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_name = self.layer_param_names[0]
        w = getattr(layer, w_name)
        if not w.t().is_contiguous():   # the kernel reads [N, K] rows: keep the [K, N] parameter a transposed view of a contiguous [N, K] tensor
            replace_parameter(layer, w_name, w.data.t().contiguous().t())
        w = getattr(layer, w_name)
        n, k = w.shape[1], w.shape[0]
        # M = 1 (decode): W8A16 on the block-scale GEMV with unit block scales, the per-channel scale applied to its output -- bf16 activations, no
        # activation quantisation launch; R9700 Qwen3.5-9B shapes 1.2-2.6x torch._scaled_mm + quant (the tiled GEMM only ties hipBLASLt at M <= 32), error 2.7e-3 vs 2.6e-2.
        self._radeon_gemv = k % 128 == 0 and self.config.out_dtype in (None, torch.bfloat16)
        if self._radeon_gemv:
            w_s = getattr(layer, self.layer_param_names[1])
            layer.register_buffer("radeon_ones_blk", torch.ones((n + 127) // 128, k // 128, dtype=torch.bfloat16, device=w.device), persistent=False)
            layer.register_buffer("radeon_w_scale_bf16", w_s.reshape(-1).to(torch.bfloat16), persistent=False)

    def apply_weights(self, layer: torch.nn.Module, x, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self._radeon_gemv and isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 and x.numel() == x.shape[-1] and not _batch_invariant():
            w = getattr(layer, self.layer_param_names[0])
            out = torch.ops.vllm.radeon_fp8_ptpc_gemv(x.reshape(1, -1).contiguous(), w.t(), layer.radeon_ones_blk, layer.radeon_w_scale_bf16, bias)
            return out.view(*x.shape[:-1], out.shape[-1])
        return super().apply_weights(layer, x, bias)

    def apply_scaled_mm(self, *, A: torch.Tensor, B: torch.Tensor, out_dtype: torch.dtype, As: torch.Tensor, Bs: torch.Tensor, bias: torch.Tensor | None,
                        output_shape: list) -> torch.Tensor:
        out = torch.ops.vllm.radeon_fp8_ptpc_mm(A if A.is_contiguous() else A.contiguous(), B.t(), As, Bs, bias, out_dtype)
        return out.view(*output_shape)
