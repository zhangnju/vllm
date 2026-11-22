# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize the target model's unquantized lm_head at load time for Radeon (VLLM_ROCM_RADEON_LMHEAD=int4 | fp8).

Decode reads the whole lm_head every step: Qwen3.5 / 3.6's 248k x 2048 bf16 head is ~1 GB, 1.63 ms per call on an R9700 (13% of a batch-1 step) even when the
checkpoint quantizes every other linear layer.  Unlike `draft_lm_head_quantization` this changes the target's logits, so it is opt-in and has to pass a PPL check.
  * int4: weight-only symmetric int4, one scale per 128 input channels, on the platform's W4A16 kernel (RDNA4 + the Radeon hybrid W4A16 kernel: 0.43 ms);
  * fp8: e4m3 with 128 x 128 block scales on radeon_ops' W8A16 kernels (bf16 activations; both RDNA generations), half the bytes of bf16.
The quantized head replaces the bf16 one (no copy is kept), before a speculative drafter shares it (its MTP / EAGLE head then reads the quantized head too).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod

logger = init_logger(__name__)

_FP8_MAX = 448.0


class RadeonFp8LMHeadMethod:
    """lm_head.quant_method for an e4m3 head with 128 x 128 block scales: logits = x @ dequant(w)^T on radeon_ops (torch.ops.vllm.radeon_fp8_w8a16_mm)."""

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.radeon_fp8_w8a16_mm(x2 if x2.is_contiguous() else x2.contiguous(), layer.weight, layer.radeon_weight_scale)
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], out.shape[-1])

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("the fp8 lm_head is not an input embedding")


@torch.no_grad()
def _quantize_fp8_block(w: torch.Tensor, chunk_rows: int = 16384) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    assert k % 128 == 0, "hidden size % 128"
    nb = (n + 127) // 128
    q = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=w.device)
    scales = torch.empty(nb, k // 128, dtype=torch.bfloat16, device=w.device)
    for r0 in range(0, n, chunk_rows):   # 128-row aligned chunks bound the fp32 working set
        blk = w[r0:r0 + chunk_rows].float()
        rows = blk.shape[0]
        pad = (-rows) % 128
        if pad:
            blk = torch.cat([blk, blk.new_zeros(pad, k)])
        g = blk.view(-1, 128, k // 128, 128)
        s = (g.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny).to(torch.bfloat16)
        scales[r0 // 128:r0 // 128 + s.shape[0]] = s
        qs = (g / s.float()[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(-1, k)
        q[r0:r0 + rows] = qs[:rows]
    return q, scales


def _quantize_head_fp8(head: ParallelLMHead) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.radeon_fp8_linear  # noqa: F401  (registers torch.ops.vllm.radeon_fp8_w8a16_mm)

    q, s = _quantize_fp8_block(head.weight.data)
    head.weight = torch.nn.Parameter(q, requires_grad=False)
    head.register_buffer("radeon_weight_scale", s, persistent=False)
    head.quant_method = RadeonFp8LMHeadMethod()


def radeon_quantize_target_lm_heads(model: torch.nn.Module) -> None:
    """VLLM_ROCM_RADEON_LMHEAD=int4 | fp8: replace every unquantized fp16 / bf16 `lm_head` in `model` by its quantized form (see the module docstring)."""
    scheme = os.environ.get("VLLM_ROCM_RADEON_LMHEAD", "").lower()
    if scheme not in ("int4", "fp8"):
        return
    from vllm.model_executor.layers.quantization.online.lm_head import _quantize_copy, _LM_HEAD_METHODS

    for parent in list(model.modules()):
        for cname, child in list(parent.named_children()):
            if cname != "lm_head" or not isinstance(child, ParallelLMHead):
                continue
            if not isinstance(child.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)) or child.weight.dtype not in (torch.float16, torch.bfloat16):
                continue
            shape = tuple(child.weight.shape)
            if scheme == "fp8":
                _quantize_head_fp8(child)
            else:
                if "int4" not in _LM_HEAD_METHODS:
                    from vllm.model_executor.layers.quantization.online.int4 import Int4PerGroup128SymLMHeadMethod

                    _LM_HEAD_METHODS["int4"] = Int4PerGroup128SymLMHeadMethod
                setattr(parent, cname, _quantize_copy(child, "int4"))   # no _SOURCES entry: the bf16 head is released
            logger.info("Radeon: quantized the target lm_head %s to %s (VLLM_ROCM_RADEON_LMHEAD)", shape, scheme)
    torch.cuda.empty_cache()
