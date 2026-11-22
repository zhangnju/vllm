# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online int4 weight-only quantization.

`Int4PerGroup32AsymLMHeadMethod` quantizes an fp16/bf16 lm_head at load time
(a `quantization_config.targets` pattern naming the lm_head, with
`{"weight": "int4_per_group_32_asym"}`). Decode reads the
whole lm_head every step, so on bandwidth-bound GPUs a full-precision head is
one of the largest per-token costs even when the checkpoint quantizes every
other linear layer.
"""

import torch

from vllm.model_executor.layers.quantization.online.lm_head import (
    OnlineLMHeadMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kInt4Static32Asym,
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

_UINT4_MAX = 15
# Zero points pack 8 output rows per int32 word.
_PACK_FACTOR = 8


def quantize_int4_per_group_asym(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric uint4 quantization of the rows of `weight` [N, K].

    Returns the codes [N, K] uint8, the scales [N, K // group_size] in the
    weight dtype and the zero points [N, K // group_size] uint8, such that
    `(codes - zero_point) * scale` approximates `weight`. Each group's
    range is widened to include 0, so the zero point always fits in uint4.
    """
    n, k = weight.shape
    groups = weight.float().view(n, k // group_size, group_size)
    w_min = groups.amin(dim=-1).clamp(max=0)
    w_max = groups.amax(dim=-1).clamp(min=0)
    # Round against the stored scale, so the codes match what the kernel
    # dequantizes.
    scale = (
        ((w_max - w_min) / _UINT4_MAX)
        .clamp(min=torch.finfo(weight.dtype).tiny)
        .to(weight.dtype)
    )
    scale_f = scale.float()
    zero_point = (-w_min / scale_f).round().clamp(0, _UINT4_MAX)
    codes = ((groups / scale_f.unsqueeze(-1)).round() + zero_point.unsqueeze(-1)).clamp(
        0, _UINT4_MAX
    )
    return codes.to(torch.uint8).view(n, k), scale, zero_point.to(torch.uint8)


@torch.no_grad()
def quantize_int4_to_mp_layout(
    weight: torch.Tensor, group_size: int, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize `weight` [N, K] into the `MPLinearKernel` uint4 layout.

    Returns the codes packed along K (int32 [N, K // 8]), the scales
    [N, K // group_size] and the zero points packed along N
    (int32 [N // 8, K // group_size]), the layout of compressed-tensors
    asymmetric W4A16 checkpoints. Rows are processed in chunks to bound the
    fp32 working set.
    """
    n, k = weight.shape
    packed = torch.empty(n, k // _PACK_FACTOR, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group_size, dtype=weight.dtype, device=weight.device)
    zero_points = torch.empty(
        n, k // group_size, dtype=torch.uint8, device=weight.device
    )
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        codes, scales[rows], zero_points[rows] = quantize_int4_per_group_asym(
            weight[rows], group_size
        )
        packed[rows] = pack_quantized_values_into_int32(
            codes.to(torch.int32), scalar_types.uint4, packed_dim=1
        )
    packed_zero_points = pack_quantized_values_into_int32(
        zero_points.to(torch.int32), scalar_types.uint4, packed_dim=0
    )
    return packed, scales, packed_zero_points


class Int4PerGroup32AsymLMHeadMethod(OnlineLMHeadMethodBase):
    """Weight-only asymmetric uint4 lm_head, one scale and zero point per 32
    input channels."""

    scheme = "int4_per_group_32_asym"
    weight_quant_key = kInt4Static32Asym
    activation_quant_key = None
    weight_type = scalar_types.uint4
    group_size = 32
    zero_points = True
    n_align = _PACK_FACTOR
    k_align = group_size

    def quantize(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return quantize_int4_to_mp_layout(weight, self.group_size)


_INT4_MAX = 7
# Symmetric int4 with one scale per 128 input channels, stored as uint4b8 (code = q + 8): the GPTQ layout every W4A16 kernel takes,
# including the Radeon rq kernels (RDNA hybrid W4A16 with VLLM_ROCM_RADEON_HYBRID_BIG runs an N >= 8192 layer on rq at every M).
kInt4Static128Sym = QuantKey(scalar_types.uint4b8, scale=ScaleDesc(torch.float16, True, GroupShape(1, 128)), symmetric=True)


@torch.no_grad()
def quantize_int4_sym_to_mp_layout(weight: torch.Tensor, group_size: int, chunk_rows: int = 8192) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Symmetric int4 quantization of `weight` [N, K] into the `MPLinearKernel` uint4b8 layout: the codes (q + 8) packed along K
    (int32 [N, K // 8]) and the scales [N, K // group_size] (absmax / 7, in the weight dtype); no zero points."""
    n, k = weight.shape
    packed = torch.empty(n, k // _PACK_FACTOR, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group_size, dtype=weight.dtype, device=weight.device)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        groups = weight[rows].float().view(-1, k // group_size, group_size)
        scale = (groups.abs().amax(dim=-1) / _INT4_MAX).clamp(min=torch.finfo(weight.dtype).tiny).to(weight.dtype)
        scales[rows] = scale
        codes = (groups / scale.float().unsqueeze(-1)).round().clamp(-_INT4_MAX - 1, _INT4_MAX) + 8   # round against the stored scale
        packed[rows] = pack_quantized_values_into_int32(codes.to(torch.int32).view(-1, k), scalar_types.uint4b8, packed_dim=1)
    return packed, scales, None


class Int4PerGroup128SymLMHeadMethod(OnlineLMHeadMethodBase):
    """Weight-only symmetric int4 lm_head (uint4b8), one scale per 128 input channels -- e.g. a speculative drafter's private head
    (`draft_lm_head_quantization="int4"`): the target verifies with its own head, so only the acceptance length can change."""

    scheme = "int4_per_group_128_sym"
    weight_quant_key = kInt4Static128Sym
    activation_quant_key = None
    weight_type = scalar_types.uint4b8
    group_size = 128
    zero_points = False
    n_align = _PACK_FACTOR
    k_align = group_size

    def quantize(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        return quantize_int4_sym_to_mp_layout(weight, self.group_size)
