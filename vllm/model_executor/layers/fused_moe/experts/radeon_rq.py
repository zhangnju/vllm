# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge from the Radeon vLLM backends to the engine-agnostic rq quantized-GEMM core (radeon_ops.qgemm, torch.ops.rq.*).

VLLM_ROCM_RADEON_RQ=1 routes the RADEON W4A16 MoE and dense kernels through rq instead of the in-tree kernels (same numerics for the int8
activation path: bit-exact); VLLM_ROCM_RADEON_ACT = int8 (default) | fp8 | int4 selects the activation format / WMMA atom.
"""
import functools

import torch

import vllm.envs as envs

ACT_IDS = {"int8": 0, "fp8": 1, "int4": 2}
EPI_PLAIN, EPI_GLU, EPI_GLUQ = 0, 1, 2


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
