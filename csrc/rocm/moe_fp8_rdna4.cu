// Radeon FP8 (e4m3, 128x128 block-scaled) MoE GEMM for RDNA4 (gfx1200 / gfx1201) on the native fp8 WMMA.
//
// Same structure as moe_w4a16_rdna4.cu: weights are re-laid once into the WMMA fragment order (per 16-row tile and 128-K group,
// 32 lanes x 64 bytes: lane = row L & 15, K half L >> 4, K step s (16 wide) = 8 bytes at s * 8), the activations are quantised to
// e4m3 per 128-K group, the 8 WMMA steps of a group accumulate in f32 and the block scale of the weights (one per 128 rows x
// 128 K) times the activation group scale is applied once per group. GEMM 1 fuses gate/up/SiLU*mul and can write the next
// GEMM's e4m3 activation directly from its epilogue.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <type_traits>

namespace {

[[maybe_unused]] constexpr int kBM = 16;
constexpr int kGroup = 128;
constexpr float kFp8Max = 448.0f;

typedef unsigned __attribute__((ext_vector_type(4))) v4u;
typedef int __attribute__((ext_vector_type(2))) v2i;
typedef float __attribute__((ext_vector_type(8))) v8f;

__device__ __forceinline__ unsigned f32_to_bf16_bits(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float bf16_bits_to_f32(unsigned b) { return __uint_as_float(b << 16); }

__device__ __forceinline__ unsigned pack4_fp8(float a, float b, float c, float d) {
#if defined(__gfx1200__) || defined(__gfx1201__)
  unsigned w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);
  return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
#else
  return 0u;
#endif
}

// x: [rows, K] bf16 -> xq e4m3 bytes [rows, K], xs float [rows, G]; one warp per (row, group)
__global__ void quant_x_fp8_kernel(const uint16_t* __restrict__ x, uint8_t* __restrict__ xq, float* __restrict__ xs, int K,
                                   int64_t n_pairs) {
  const int lane = threadIdx.x & 31;
  const int64_t pair = (int64_t)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (pair >= n_pairs) return;
  const int G = K / kGroup;
  const int64_t row = pair / G;
  const int g = (int)(pair % G);
  const uint2 raw = *(const uint2*)(x + row * K + g * kGroup + lane * 4);
  float v[4] = {bf16_bits_to_f32(raw.x & 0xffffu), bf16_bits_to_f32(raw.x >> 16), bf16_bits_to_f32(raw.y & 0xffffu),
                bf16_bits_to_f32(raw.y >> 16)};
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
  const float d = amax / kFp8Max;
  const float inv = d > 0.0f ? 1.0f / d : 0.0f;
  *(unsigned*)(xq + row * K + g * kGroup + lane * 4) = pack4_fp8(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
  if (lane == 0) xs[row * G + g] = d;
}

// w_in: [E, R, K] e4m3 bytes; w_out: fragment order (see file header)
__global__ void pack_fp8_kernel(const uint8_t* __restrict__ w_in, uint8_t* __restrict__ w_out, int64_t E, int R, int K) {
  const int G = K / kGroup;
  const int64_t per_e = (int64_t)(R / 16) * G * 32;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= E * per_e) return;
  const int64_t e = i / per_e, c = i % per_e;
  const int lane = (int)(c % 32);
  const int64_t tg = c / 32;
  const int g = (int)(tg % G);
  const int64_t tile = tg / G;
  const int row = (int)(tile * 16 + (lane & 15)), h = lane >> 4;
  const uint8_t* src = w_in + (e * R + row) * (int64_t)K + g * kGroup + h * 8;
  uint8_t* dst = w_out + i * 64;
#pragma unroll
  for (int s = 0; s < 8; ++s) *(uint2*)(dst + s * 8) = *(const uint2*)(src + s * 16);
}

template <bool GLU, int NW>
__global__ __launch_bounds__(NW * 32) void moe_fp8_kernel(
    const uint8_t* __restrict__ xq, const float* __restrict__ xs, int a_row_div, const uint8_t* __restrict__ wp,
    const float* __restrict__ wsc, int tiles_per_expert, int up_tile_off, uint16_t* __restrict__ out,
    const float* __restrict__ topk_w, const int* __restrict__ sorted_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ ntpp, int R, int K, int num_slots, int num_pid_n, uint8_t* __restrict__ oq,
    float* __restrict__ ods) {
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
  const int G = K / kGroup;
  const uint8_t* xrow = xq + (size_t)arow * K + half8;
  const float* xsrow = xs + (size_t)arow * G;
  const int tile = min(pid_n * NW + warp, R / 16 - 1);
  const int Rtot = GLU ? 2 * R : R;
  const uint8_t* wb = wp + ((size_t)e * tiles_per_expert + tile) * G * 2048 + lane * 64;
  const uint8_t* wb2 = GLU ? wp + ((size_t)e * tiles_per_expert + tile + up_tile_off) * G * 2048 + lane * 64 : wb;
  const float* sb = wsc + ((size_t)e * (Rtot / 128) + (tile >> 3)) * G;
  const float* sb2 = GLU ? wsc + ((size_t)e * (Rtot / 128) + ((tile + up_tile_off) >> 3)) * G : sb;

  float acc[8], acc2[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) { acc[j] = 0.0f; acc2[j] = 0.0f; }
  const v8f zero = {0, 0, 0, 0, 0, 0, 0, 0};

  constexpr int GU = 2;
  for (int g0 = 0; g0 < G; g0 += GU) {
    v4u wa[GU][4], wc[GU][4];
    long long xb[GU][8];
    float xd[GU], sa[GU], sc[GU];
#pragma unroll
    for (int u = 0; u < GU; ++u) {
      const int g = g0 + u;
      if (g < G) {
#pragma unroll
        for (int q = 0; q < 4; ++q) wa[u][q] = *(const v4u*)(wb + (size_t)g * 2048 + q * 16);
        sa[u] = sb[g];
        if (GLU) {
#pragma unroll
          for (int q = 0; q < 4; ++q) wc[u][q] = *(const v4u*)(wb2 + (size_t)g * 2048 + q * 16);
          sc[u] = sb2[g];
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) xb[u][k] = *(const long long*)(xrow + g * kGroup + k * 16);
        xd[u] = xsrow[g];
      }
    }
#pragma unroll
    for (int u = 0; u < GU; ++u) {
      if (g0 + u >= G) continue;
      v8f tmp = zero;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const v4u w4 = wa[u][q];
        v2i a0 = {(int)w4.x, (int)w4.y}, a1 = {(int)w4.z, (int)w4.w};
        v2i x0, x1;
        *(long long*)&x0 = xb[u][2 * q];
        *(long long*)&x1 = xb[u][2 * q + 1];
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, x0, tmp);
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, x1, tmp);
      }
      const float f = sa[u] * xd[u];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += tmp[j] * f;
      if (GLU) {
        v8f tm2 = zero;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const v4u w4 = wc[u][q];
          v2i a0 = {(int)w4.x, (int)w4.y}, a1 = {(int)w4.z, (int)w4.w};
          v2i x0, x1;
          *(long long*)&x0 = xb[u][2 * q];
          *(long long*)&x1 = xb[u][2 * q + 1];
          tm2 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, x0, tm2);
          tm2 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, x1, tm2);
        }
        const float f2 = sc[u] * xd[u];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc2[j] += tm2[j] * f2;
      }
    }
  }
  const int n0 = pid_n * (NW * 16) + warp * 16 + half8;
  if (GLU && oq != nullptr) {
    // fused activation quantisation: this block's 128 output columns are one K group of the next GEMM
    __shared__ float smax[NW][16];
    float v[8], amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = bf16_bits_to_f32(f32_to_bf16_bits(acc[j] / (1.0f + __expf(-acc[j])) * acc2[j]));
      amax = fmaxf(amax, fabsf(v[j]));
    }
    amax = fmaxf(amax, __shfl_xor(amax, 16));
    if (lane < 16) smax[warp][row] = amax;
    __syncthreads();
    float m = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) m = fmaxf(m, smax[w][row]);
    const float d = m / kFp8Max, inv = d > 0.0f ? 1.0f / d : 0.0f;
    const unsigned p0 = pack4_fp8(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
    const unsigned p1 = pack4_fp8(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv);
    if (valid) *(uint2*)(oq + (size_t)s * R + n0) = make_uint2(p0, p1);
    if (warp == 0 && lane < 16 && valid) ods[(size_t)s * (R / kGroup) + pid_n] = d;
    return;
  }
  if (valid && n0 < R) {
    const float rw = topk_w ? topk_w[s] : 1.0f;
    unsigned r[4];
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      float v0 = GLU ? acc[j] / (1.0f + __expf(-acc[j])) * acc2[j] : acc[j];
      float v1 = GLU ? acc[j + 1] / (1.0f + __expf(-acc[j + 1])) * acc2[j + 1] : acc[j + 1];
      r[j / 2] = f32_to_bf16_bits(v0 * rw) | (f32_to_bf16_bits(v1 * rw) << 16);
    }
    v4u o; o.x = r[0]; o.y = r[1]; o.z = r[2]; o.w = r[3];
    *(v4u*)(out + (size_t)s * R + n0) = o;
  }
#endif
}

}  // namespace

void moe_quant_x_fp8_rdna4(torch::Tensor x, torch::Tensor xq, torch::Tensor xs) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x must be contiguous bf16 [rows, K]");
  const int64_t rows = x.size(0), K = x.size(1);
  TORCH_CHECK(K % kGroup == 0, "K must be a multiple of 128");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t pairs = rows * (K / kGroup);
  quant_x_fp8_kernel<<<(unsigned)((pairs + 7) / 8), 256, 0, stream>>>((const uint16_t*)x.data_ptr(), (uint8_t*)xq.data_ptr(),
                                                                      (float*)xs.data_ptr(), (int)K, pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// w_in [E, R, K] uint8 (e4m3 bytes) -> fragment order, same shape
void moe_pack_fp8_rdna4(torch::Tensor w_in, torch::Tensor w_out) {
  TORCH_CHECK(w_in.scalar_type() == at::kByte && w_in.dim() == 3 && w_in.is_contiguous() && w_out.sizes() == w_in.sizes(), "bad shapes");
  const int64_t E = w_in.size(0), R = w_in.size(1), K = w_in.size(2);
  TORCH_CHECK(R % 128 == 0 && K % kGroup == 0, "need R % 128 == 0 and K % 128 == 0");
  const at::cuda::OptionalCUDAGuard guard(device_of(w_in));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = E * (R / 16) * (K / kGroup) * 32;
  pack_fp8_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>((const uint8_t*)w_in.data_ptr(), (uint8_t*)w_out.data_ptr(), E, (int)R, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// glu: w [E, 2N, K] packed, ws [E, 2N/128, K/128] f32, out [num_slots, N]; otherwise out [num_slots, R]
void moe_fp8_gemm_rdna4(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor ws, torch::Tensor out,
                        std::optional<torch::Tensor> topk_weights, torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
                        torch::Tensor num_tokens_post_padded, int64_t a_row_div, int64_t num_slots, bool glu,
                        std::optional<torch::Tensor> oq, std::optional<torch::Tensor> ods, int64_t nw) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && ws.scalar_type() == at::kFloat && w.scalar_type() == at::kByte, "dtypes");
  const int64_t Rtot = w.size(1), K = w.size(2);
  const int64_t R = glu ? Rtot / 2 : Rtot;
  TORCH_CHECK(xq.size(1) == K && out.size(1) == R && out.size(0) >= num_slots, "shape mismatch");
  const bool fuse_q = oq.has_value();
  TORCH_CHECK(!fuse_q || (glu && R % kGroup == 0 && ods.has_value()), "fused quant needs glu, R % 128 == 0");
  TORCH_CHECK(R % 128 == 0 && K % kGroup == 0, "need R % 128 == 0 and K % 128 == 0");
  TORCH_CHECK(ws.dim() == 3 && ws.size(1) == Rtot / 128 && ws.size(2) == K / kGroup, "scale layout");
  const at::cuda::OptionalCUDAGuard guard(device_of(xq));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int m_blocks = (int)expert_ids.size(0);
  TORCH_CHECK(nw == 8 || nw == 4 || nw == 2 || nw == 1, "nw must be 1, 2, 4 or 8");
  TORCH_CHECK(!fuse_q || nw == 8, "fused quant needs nw == 8");
  const int num_pid_n = (int)((R + nw * 16 - 1) / (nw * 16));
  const dim3 grid((unsigned)(m_blocks * num_pid_n)), block((unsigned)(nw * 32));
  const float* tw = topk_weights.has_value() ? (const float*)topk_weights->data_ptr() : nullptr;
  const int tiles = (int)(Rtot / 16);
  auto launch = [&](auto glu_tag, auto nw_tag) {
    constexpr bool GLU = decltype(glu_tag)::value;
    constexpr int NW = decltype(nw_tag)::value;
    moe_fp8_kernel<GLU, NW><<<grid, block, 0, stream>>>(
        (const uint8_t*)xq.data_ptr(), (const float*)xs.data_ptr(), (int)a_row_div, (const uint8_t*)w.data_ptr(),
        (const float*)ws.data_ptr(), tiles, GLU ? (int)(R / 16) : 0, (uint16_t*)out.data_ptr(), tw,
        (const int*)sorted_token_ids.data_ptr(), (const int*)expert_ids.data_ptr(),
        (const int*)num_tokens_post_padded.data_ptr(), (int)R, (int)K, (int)num_slots, num_pid_n,
        fuse_q ? (uint8_t*)oq->data_ptr() : nullptr, fuse_q ? (float*)ods->data_ptr() : nullptr);
  };
  using std::integral_constant;
  if (glu) {
    TORCH_CHECK(nw == 8 || nw == 4 || nw == 2, "glu supports nw 8, 4, 2");
    if (nw == 8) launch(std::true_type{}, integral_constant<int, 8>{});
    else if (nw == 4) launch(std::true_type{}, integral_constant<int, 4>{});
    else launch(std::true_type{}, integral_constant<int, 2>{});
  } else {
    if (nw == 8) launch(std::false_type{}, integral_constant<int, 8>{});
    else if (nw == 4) launch(std::false_type{}, integral_constant<int, 4>{});
    else if (nw == 2) launch(std::false_type{}, integral_constant<int, 2>{});
    else launch(std::false_type{}, integral_constant<int, 1>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
