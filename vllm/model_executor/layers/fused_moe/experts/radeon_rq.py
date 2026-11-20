# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge from the Radeon vLLM backends to the engine-agnostic rq quantized-GEMM core (radeon_ops.qgemm, torch.ops.rq.*).

VLLM_ROCM_RADEON_RQ=1 routes the RADEON W4A16 MoE and dense kernels through rq instead of the in-tree kernels (same numerics for the int8
activation path: bit-exact); VLLM_ROCM_RADEON_ACT = int8 (default) | fp8 | int4 selects the activation format / WMMA atom.
"""
import functools
import os

import torch

import vllm.envs as envs

ACT_IDS = {"int8": 0, "fp8": 1, "int4": 2}
EPI_PLAIN, EPI_GLU, EPI_GLUQ = 0, 1, 2


_PREQUANT = os.environ.get("VLLM_ROCM_RADEON_PREQUANT", "0").lower() in ("1", "true")


def prequant_take(a: torch.Tensor):
    """(xq uint8, xs f32, xsum i32) if the fused all-reduce + RMSNorm just produced `a` together with its int8 group-128 quantisation, else None (radeon_ops.comm.prequant)."""
    if not _PREQUANT:
        return None
    from radeon_ops.comm import prequant

    return prequant.take(a)


# VLLM_ROCM_RADEON_BIG=1: dense W4A16 layers with M >= VLLM_ROCM_RADEON_BIG_MIN_M (default 512) run the LDS large-M rq kernel (rq.gemm_big, int8 activations);
# =2 additionally runs 64 < M < MIN_M on the streaming rq kernel instead of Triton.  Needs VLLM_ROCM_RADEON_RQ=1 (act int8).
_BIG = int(os.environ.get("VLLM_ROCM_RADEON_BIG", "0") or 0) if os.environ.get("VLLM_ROCM_RADEON_BIG", "0") not in ("true", "True") else 1
BIG_MIN_M = int(os.environ.get("VLLM_ROCM_RADEON_BIG_MIN_M", "512"))   # radeon_big


_PLAN = os.environ.get("VLLM_ROCM_RADEON_PLAN", "0").lower() in ("1", "true")


def plan_enabled() -> bool:
    return _PLAN


@functools.lru_cache(maxsize=4096)
def _moe_planned(r_up: int, k: int, inter: int, m_bucket: int, e: int, top_k: int, act: int) -> tuple[bool, bool, int]:
    from radeon_ops import qgemm

    a = qgemm.ActFormat(("int8", "fp8", "int4")[act])
    up = qgemm.plan(a, qgemm.Epilogue.GLU, qgemm.GemmType.MOE_SORTED, qgemm.Shape(R=r_up, K=k, M=m_bucket, E=e, top_k=top_k))
    dn = qgemm.plan(a, qgemm.Epilogue.PLAIN, qgemm.GemmType.MOE_SORTED, qgemm.Shape(R=k, K=inter, M=m_bucket, E=e, top_k=top_k))
    lds = qgemm.Datapath.LDS
    return up.config.tuning.datapath == lds, dn.config.tuning.datapath == lds, dn.config.tuning.nw


def moe_plan(r_up: int, k: int, inter: int, m: int, e: int, top_k: int, act: int) -> tuple[bool, bool, int]:
    """(gate-up on the LDS large-M kernel, down on the LDS kernel, nw of the streaming down GEMM) from radeon_ops.qgemm.plan
    (the measured per-arch tuning table) for the MoE GEMMs of a batch of m tokens."""
    return _moe_planned(r_up, k, inter, 1 << max(0, (m - 1).bit_length()), e, top_k, act)


def big_mode() -> int:
    return _BIG


def enabled() -> bool:
    return bool(envs.VLLM_ROCM_RADEON_RQ)


def act_id() -> int:
    return ACT_IDS[envs.VLLM_ROCM_RADEON_ACT.lower()]


@functools.lru_cache(maxsize=1)
def rq_ops():
    from radeon_ops.qgemm import torch_ops

    torch_ops.load()
    return torch.ops.rq


def act_bits(act: int) -> int:
    return 4 if act == 2 else 8


def act_sum_dtype(act: int) -> torch.dtype:
    return torch.float32 if act == 1 else torch.int32
