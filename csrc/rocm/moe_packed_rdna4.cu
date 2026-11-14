// Radeon packed-bf16 MoE GEMM for RDNA4 (gfx1200 / gfx1201, wave32 WMMA).
//
// The expert weights are re-laid ONCE (at load time, in place) into the WMMA fragment order: per 16-row tile and
// 16-wide K step, 32 chunks of 16 bytes, chunk L = row (L & 15), K offset (L >> 4) * 8 - so a wave load of 512
// contiguous bytes lands directly in the B fragment, no LDS round trip. A block of 8 warps computes 128 output rows
// for the 16 (token, expert) slots of one expert block of vLLM's moe_align_block_size (block size 16).
//
//   moe_pack_bf16_rdna4(w)                      in-place repack of w [E, R, K] (bf16)
//   moe_packed_gemm_bf16_rdna4(...)             out[slot, :] = (silu(W[:N] a) * (W[N:] a))  (glu)  or  W a  (down),
//                                               optionally scaled by the router weight of the slot; bf16 in / bf16 out,
//                                               fp32 accumulation.
//
// Source of the kernel design: the Radeon operator library (radeon_ops), benchmarks/llama_cpp (packed MoE for ggml).
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace {

[[maybe_unused]] constexpr int kBM = 16;      // slots per block row (one WMMA row tile)
constexpr int kWarps = 8;    // 8 warps x 16 output rows = 128 rows of W per block
constexpr int kKS = 8;       // K steps of 16 whose loads are in flight together

typedef unsigned __attribute__((ext_vector_type(4))) v4u;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float accf;
typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 v8h;

__device__ __forceinline__ unsigned f32_to_bf16_bits(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
  return u >> 16;
}

__global__ void pack_kernel(const char* __restrict__ src, v4u* __restrict__ dst,
                            int64_t n_chunks, int R, int K, size_t nb01) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_chunks) return;
  const int ksteps = K / 16;
  const int64_t tile = i / ((int64_t)ksteps * 32);
  const int64_t r = i % ((int64_t)ksteps * 32);
  const int ks = (int)(r / 32);
  const int q = (int)(r % 32);
  const int64_t row = tile * 16 + (q & 15);
  const int64_t k0 = (int64_t)ks * 16 + (q >> 4) * 8;
  dst[i] = *(const v4u*)(src + row * nb01 + k0 * 2);
}

template <bool GLU>
__global__ __launch_bounds__(kWarps * 32) void moe_packed_kernel(
    const uint16_t* __restrict__ a, int a_row_div, const uint16_t* __restrict__ wp,
    int tiles_per_expert, int up_tile_off, uint16_t* __restrict__ out,
    const float* __restrict__ topk_w, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ ntpp, int R, int K,
    int num_slots, int num_pid_n) {
#if defined(__gfx1201__) || defined(__gfx1200__)
  const int pid = blockIdx.x, pid_m = pid / num_pid_n, pid_n = pid % num_pid_n;
  if (pid_m * kBM >= *ntpp) return;
  const int e = expert_ids[pid_m];
  if (e < 0) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int half8 = (lane >= 16) ? 8 : 0;
  const int row = lane & 15;
  const int s = sorted_ids[pid_m * kBM + row];
  const bool valid = s < num_slots;
  const int arow = valid ? s / a_row_div : 0;
  const uint16_t* ap = a + (size_t)arow * K + half8;
  const int tile = min(pid_n * kWarps + warp, R / 16 - 1);
  const size_t ktiles = (size_t)(K / 16) * 256;
  const uint16_t* wptr = wp + ((size_t)e * tiles_per_expert + tile) * ktiles + lane * 8;
  const uint16_t* wptr2 =
      GLU ? wp + ((size_t)e * tiles_per_expert + tile + up_tile_off) * ktiles + lane * 8 : wptr;

  accf acc, acc2;
#pragma unroll
  for (int j = 0; j < 8; ++j) { acc[j] = 0.0f; acc2[j] = 0.0f; }
  for (int kb = 0; kb < K; kb += kKS * 16) {
    v4u av[kKS], wv[kKS], uv[kKS];
#pragma unroll
    for (int q = 0; q < kKS; ++q) {
      av[q] = *(const v4u*)(ap + kb + q * 16);
      wv[q] = *(const v4u*)(wptr + (size_t)(kb / 16 + q) * 256);
      if (GLU) uv[q] = *(const v4u*)(wptr2 + (size_t)(kb / 16 + q) * 256);
    }
#pragma unroll
    for (int q = 0; q < kKS; ++q) {
      v8h x, b, c;
      *(v4u*)&x = wv[q];  // (weights, activations): acc = C^T tile, a lane holds 8 consecutive output rows of one slot
      *(v4u*)&b = av[q];
      acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(x, b, acc);
      if (GLU) {
        *(v4u*)&c = uv[q];
        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(c, b, acc2);
      }
    }
  }
  const int n0 = pid_n * (kWarps * 16) + warp * 16 + half8;
  if (valid && n0 < R) {
    const float rw = topk_w ? topk_w[s] : 1.0f;
    unsigned r[4];
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      float v0 = GLU ? acc[j] / (1.0f + __expf(-acc[j])) * acc2[j] : acc[j];
      float v1 = GLU ? acc[j + 1] / (1.0f + __expf(-acc[j + 1])) * acc2[j + 1] : acc[j + 1];
      r[j / 2] = f32_to_bf16_bits(v0 * rw) | (f32_to_bf16_bits(v1 * rw) << 16);
    }
    v4u o;
    o.x = r[0]; o.y = r[1]; o.z = r[2]; o.w = r[3];
    *(v4u*)(out + (size_t)s * R + n0) = o;
  }
#endif
}

}  // namespace

// In-place repack of w [E, R, K] bf16, one expert at a time through a temporary of one expert's size.
void moe_pack_bf16_rdna4(torch::Tensor w) {
  TORCH_CHECK(w.dim() == 3 && w.scalar_type() == at::kBFloat16 && w.is_contiguous(),
              "moe_pack_bf16_rdna4: w must be a contiguous bf16 [E, R, K] tensor");
  const int64_t E = w.size(0), R = w.size(1), K = w.size(2);
  TORCH_CHECK(R % 16 == 0 && K % (kKS * 16) == 0, "moe_pack_bf16_rdna4: need R % 16 == 0 and K % 128 == 0");
  const at::cuda::OptionalCUDAGuard guard(device_of(w));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor tmp = torch::empty({R, K}, w.options());
  const int64_t n_chunks = R * K / 8;
  const int threads = 256;
  for (int64_t e = 0; e < E; ++e) {
    char* base = (char*)w.data_ptr() + e * R * K * 2;
    cudaMemcpyAsync(tmp.data_ptr(), base, R * K * 2, cudaMemcpyDeviceToDevice, stream);
    pack_kernel<<<(unsigned)((n_chunks + threads - 1) / threads), threads, 0, stream>>>(
        (const char*)tmp.data_ptr(), (v4u*)base, n_chunks, (int)R, (int)K, (size_t)K * 2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaStreamSynchronize(stream);
}

// a: [rows, K] bf16 (GEMM 1: the hidden states, slot s reads row s / top_k; GEMM 2: the activations, slot s reads row s)
// w: packed [E, Rtot, K]; glu: Rtot = 2N, out [num_slots, N]; otherwise out [num_slots, Rtot]
void moe_packed_gemm_bf16_rdna4(torch::Tensor a, torch::Tensor w, torch::Tensor out,
                                std::optional<torch::Tensor> topk_weights,
                                torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
                                torch::Tensor num_tokens_post_padded, int64_t a_row_div,
                                int64_t num_slots, bool glu) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16, "moe_packed_gemm_bf16_rdna4: bf16 only");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
  const int64_t E = w.size(0), Rtot = w.size(1), K = w.size(2);
  const int64_t R = glu ? Rtot / 2 : Rtot;
  TORCH_CHECK(a.size(1) == K && out.size(1) == R && out.size(0) >= num_slots, "shape mismatch");
  TORCH_CHECK(R % 16 == 0 && K % (kKS * 16) == 0 && R % 8 == 0, "need R % 16 == 0 and K % 128 == 0");
  TORCH_CHECK(!glu || (Rtot % 32 == 0), "glu needs 2N % 32 == 0");
  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int m_blocks = (int)expert_ids.size(0);
  const int num_pid_n = (int)((R + kWarps * 16 - 1) / (kWarps * 16));
  const dim3 grid((unsigned)(m_blocks * num_pid_n)), block(kWarps * 32);
  const float* tw = topk_weights.has_value() ? (const float*)topk_weights->data_ptr() : nullptr;
  const int tiles = (int)(Rtot / 16);
  if (glu) {
    moe_packed_kernel<true><<<grid, block, 0, stream>>>(
        (const uint16_t*)a.data_ptr(), (int)a_row_div, (const uint16_t*)w.data_ptr(), tiles, (int)(R / 16),
        (uint16_t*)out.data_ptr(), tw, (const int*)sorted_token_ids.data_ptr(), (const int*)expert_ids.data_ptr(),
        (const int*)num_tokens_post_padded.data_ptr(), (int)R, (int)K, (int)num_slots, num_pid_n);
  } else {
    moe_packed_kernel<false><<<grid, block, 0, stream>>>(
        (const uint16_t*)a.data_ptr(), (int)a_row_div, (const uint16_t*)w.data_ptr(), tiles, 0,
        (uint16_t*)out.data_ptr(), tw, (const int*)sorted_token_ids.data_ptr(), (const int*)expert_ids.data_ptr(),
        (const int*)num_tokens_post_padded.data_ptr(), (int)R, (int)K, (int)num_slots, num_pid_n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  (void)E;
}
