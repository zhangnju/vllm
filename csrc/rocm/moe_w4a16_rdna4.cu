// Radeon W4A16 MoE GEMM for RDNA4 (gfx1200 / gfx1201): GPTQ-style symmetric int4 weights (value = (q - 8) * scale, one bf16
// scale per output row and 128-wide K group) times activations quantised to int8 per 128-K group, on the int8 WMMA.
//
//   * Weights are re-laid ONCE into the WMMA fragment order: per 16-row tile and per 128-K group, 32 lanes x 32 bytes
//     (lane = row L & 15, K half L >> 4; per K step of 16 the lane's 8 weights are 4 bytes: byte i = element i (low nibble) |
//     element 4 + i (high nibble)), i.e. a wave reads 1 KB contiguous and unpacks 8 int8 per step with two ANDs.
//   * The weights stay UNSIGNED (q, 0..15): sum (q - 8) x = sum q x - 8 sum x, the second term comes from the per-group
//     activation sums, so no sign handling is needed in the inner loop.
//   * One scale per (row, group) and one per (activation row, group) is applied after the 8 K steps of a group: the int32
//     accumulator never needs rescaling inside the group (128 = the weight group size = the activation group size).
//
// Source of the design: the Radeon operator library (radeon_ops), benchmarks/llama_cpp (Q4_K / MXFP4 / NVFP4 packed kernels for ggml).
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <type_traits>

namespace {

[[maybe_unused]] constexpr int kBM = 16;   // slots per block row (one WMMA column tile)
constexpr int kWarps = 8;                  // 8 warps x 16 output rows = 128 rows of W per block
constexpr int kGroup = 128;                // K group = weight scale group = activation scale group

typedef unsigned __attribute__((ext_vector_type(4))) v4u;
typedef int __attribute__((ext_vector_type(2))) v2i;
typedef __attribute__((__vector_size__(8 * sizeof(int)))) int acci;

__device__ __forceinline__ unsigned f32_to_bf16_bits(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float bf16_bits_to_f32(unsigned b) { return __uint_as_float(b << 16); }

// x: [rows, K] bf16 -> xq int8 [rows, K], xs float [rows, G], xsum int32 [rows, G]; one warp per (row, group)
__global__ void quant_x_kernel(const uint16_t* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xs,
                               int* __restrict__ xsum, int K, int64_t n_pairs) {
  const int lane = threadIdx.x & 31;
  const int64_t pair = (int64_t)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (pair >= n_pairs) return;
  const int G = K / kGroup;
  const int64_t row = pair / G;
  const int g = (int)(pair % G);
  const uint16_t* src = x + row * K + g * kGroup + lane * 4;
  const uint2 raw = *(const uint2*)src;
  float v[4] = {bf16_bits_to_f32(raw.x & 0xffffu), bf16_bits_to_f32(raw.x >> 16),
                bf16_bits_to_f32(raw.y & 0xffffu), bf16_bits_to_f32(raw.y >> 16)};
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
  const float d = amax / 127.0f;
  const float inv = d > 0.0f ? 1.0f / d : 0.0f;
  int q[4], sum = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) { q[i] = __float2int_rn(v[i] * inv); sum += q[i]; }
  *(unsigned*)(xq + row * K + g * kGroup + lane * 4) =
      (unsigned)(q[0] & 0xff) | ((unsigned)(q[1] & 0xff) << 8) | ((unsigned)(q[2] & 0xff) << 16) | ((unsigned)(q[3] & 0xff) << 24);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o);
  if (lane == 0) { xs[row * G + g] = d; xsum[row * G + g] = sum; }
}

// w_in: N-first [E, R, K/2] (byte j of a row = q[2j] | q[2j+1] << 4); w_out: fragment order (see file header)
__global__ void pack_kernel(const uint8_t* __restrict__ w_in, uint8_t* __restrict__ w_out, int64_t E, int R, int K) {
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
  const uint8_t* src = w_in + (e * R + row) * (K / 2) + g * (kGroup / 2);
  uint8_t* dst = w_out + i * 32;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const unsigned u = *(const unsigned*)(src + s * 8 + h * 4);   // 8 consecutive elements (K step s, half h)
    const unsigned b0 = u & 0xff, b1 = (u >> 8) & 0xff, b2 = (u >> 16) & 0xff, b3 = u >> 24;
    dst[s * 4 + 0] = (uint8_t)((b0 & 15) | ((b2 & 15) << 4));
    dst[s * 4 + 1] = (uint8_t)((b0 >> 4) | ((b2 >> 4) << 4));
    dst[s * 4 + 2] = (uint8_t)((b1 & 15) | ((b3 & 15) << 4));
    dst[s * 4 + 3] = (uint8_t)((b1 >> 4) | ((b3 >> 4) << 4));
  }
}

template <bool GLU>
__global__ __launch_bounds__(kWarps * 32) void moe_w4_kernel(
    const int8_t* __restrict__ xq, const float* __restrict__ xs, const int* __restrict__ xsum, int a_row_div,
    const uint8_t* __restrict__ wp, const uint16_t* __restrict__ wsc, int tiles_per_expert, int up_tile_off,
    uint16_t* __restrict__ out, const float* __restrict__ topk_w, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ ntpp, int R, int K, int num_slots, int num_pid_n) {
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
  const int8_t* xrow = xq + (size_t)arow * K + half8;
  const float* xsrow = xs + (size_t)arow * G;
  const int* xsumrow = xsum + (size_t)arow * G;
  const int tile = min(pid_n * kWarps + warp, R / 16 - 1);
  const uint8_t* wb = wp + ((size_t)e * tiles_per_expert + tile) * G * 1024 + lane * 32;
  const uint8_t* wb2 = GLU ? wp + ((size_t)e * tiles_per_expert + tile + up_tile_off) * G * 1024 + lane * 32 : wb;
  const uint16_t* sb = wsc + ((size_t)e * tiles_per_expert + tile) * G * 16 + half8;
  const uint16_t* sb2 = GLU ? wsc + ((size_t)e * tiles_per_expert + tile + up_tile_off) * G * 16 + half8 : sb;

  float acc[8], acc2[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) { acc[j] = 0.0f; acc2[j] = 0.0f; }
  acci zero;
#pragma unroll
  for (int j = 0; j < 8; ++j) zero[j] = 0;
  const unsigned M4 = 0x0f0f0f0fu;

  constexpr int GU = 2;   // groups whose loads are in flight together
  for (int g0 = 0; g0 < G; g0 += GU) {
    v4u wa[GU][2], wc[GU][2], sa[GU], sc[GU];
    long long xb[GU][8];
    float xd[GU]; int xm[GU];
#pragma unroll
    for (int u = 0; u < GU; ++u) {
      const int g = g0 + u;
      if (g < G) {
        wa[u][0] = *(const v4u*)(wb + (size_t)g * 1024);
        wa[u][1] = *(const v4u*)(wb + (size_t)g * 1024 + 16);
        sa[u] = *(const v4u*)(sb + (size_t)g * 16);
        if (GLU) {
          wc[u][0] = *(const v4u*)(wb2 + (size_t)g * 1024);
          wc[u][1] = *(const v4u*)(wb2 + (size_t)g * 1024 + 16);
          sc[u] = *(const v4u*)(sb2 + (size_t)g * 16);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) xb[u][k] = *(const long long*)(xrow + g * kGroup + k * 16);
        xd[u] = xsrow[g];
        xm[u] = xsumrow[g];
      }
    }
#pragma unroll
    for (int u = 0; u < GU; ++u) {
      if (g0 + u >= G) continue;
      const unsigned wv[8] = {wa[u][0].x, wa[u][0].y, wa[u][0].z, wa[u][0].w, wa[u][1].x, wa[u][1].y, wa[u][1].z, wa[u][1].w};
      acci tmp = zero;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        v2i a = {(int)(wv[k] & M4), (int)((wv[k] >> 4) & M4)};
        v2i x; *(long long*)&x = xb[u][k];
        tmp = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, x, tmp, false);
      }
      const float corr = 8.0f * (float)xm[u], xsc = xd[u];
      const unsigned sw[4] = {sa[u].x, sa[u].y, sa[u].z, sa[u].w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float swj = bf16_bits_to_f32((sw[j >> 1] >> (16 * (j & 1))) & 0xffffu);
        acc[j] += swj * xsc * ((float)tmp[j] - corr);
      }
      if (GLU) {
        const unsigned vv[8] = {wc[u][0].x, wc[u][0].y, wc[u][0].z, wc[u][0].w, wc[u][1].x, wc[u][1].y, wc[u][1].z, wc[u][1].w};
        acci tm2 = zero;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          v2i a = {(int)(vv[k] & M4), (int)((vv[k] >> 4) & M4)};
          v2i x; *(long long*)&x = xb[u][k];
          tm2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, x, tm2, false);
        }
        const unsigned sv[4] = {sc[u].x, sc[u].y, sc[u].z, sc[u].w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float swj = bf16_bits_to_f32((sv[j >> 1] >> (16 * (j & 1))) & 0xffffu);
          acc2[j] += swj * xsc * ((float)tm2[j] - corr);
        }
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
    v4u o; o.x = r[0]; o.y = r[1]; o.z = r[2]; o.w = r[3];
    *(v4u*)(out + (size_t)s * R + n0) = o;
  }
#endif
}

}  // namespace

void moe_quant_x_g128_rdna4(torch::Tensor x, torch::Tensor xq, torch::Tensor xs, torch::Tensor xsum) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x must be contiguous bf16 [rows, K]");
  const int64_t rows = x.size(0), K = x.size(1);
  TORCH_CHECK(K % kGroup == 0, "K must be a multiple of 128");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t pairs = rows * (K / kGroup);
  quant_x_kernel<<<(unsigned)((pairs + 7) / 8), 256, 0, stream>>>((const uint16_t*)x.data_ptr(), (int8_t*)xq.data_ptr(),
                                                                (float*)xs.data_ptr(), (int*)xsum.data_ptr(), (int)K, pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// w_in [E, R, K/2] uint8 (N-first GPTQ nibbles) -> w_out fragment order (same shape)
void moe_pack_w4_rdna4(torch::Tensor w_in, torch::Tensor w_out) {
  TORCH_CHECK(w_in.scalar_type() == at::kByte && w_in.dim() == 3 && w_in.is_contiguous() && w_out.sizes() == w_in.sizes(), "bad shapes");
  const int64_t E = w_in.size(0), R = w_in.size(1), K = w_in.size(2) * 2;
  TORCH_CHECK(R % 16 == 0 && K % kGroup == 0, "need R % 16 == 0 and K % 128 == 0");
  const at::cuda::OptionalCUDAGuard guard(device_of(w_in));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = E * (R / 16) * (K / kGroup) * 32;
  pack_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>((const uint8_t*)w_in.data_ptr(), (uint8_t*)w_out.data_ptr(), E, (int)R, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// glu: w [E, 2N, K/2] packed, ws [E, 2N/16, G, 16] bf16, out [num_slots, N]; otherwise out [num_slots, R]
void moe_w4a16_gemm_rdna4(torch::Tensor xq, torch::Tensor xs, torch::Tensor xsum, torch::Tensor w, torch::Tensor ws,
                          torch::Tensor out, std::optional<torch::Tensor> topk_weights, torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded, int64_t a_row_div,
                          int64_t num_slots, bool glu) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && ws.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kByte, "dtypes");
  const int64_t Rtot = w.size(1), K = w.size(2) * 2;
  const int64_t R = glu ? Rtot / 2 : Rtot;
  TORCH_CHECK(xq.size(1) == K && out.size(1) == R && out.size(0) >= num_slots, "shape mismatch");
  TORCH_CHECK(R % 16 == 0 && K % kGroup == 0 && R % 8 == 0, "need R % 16 == 0 and K % 128 == 0");
  TORCH_CHECK(ws.dim() == 4 && ws.size(1) == Rtot / 16 && ws.size(2) == K / kGroup && ws.size(3) == 16, "scale layout");
  const at::cuda::OptionalCUDAGuard guard(device_of(xq));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int m_blocks = (int)expert_ids.size(0);
  const int num_pid_n = (int)((R + kWarps * 16 - 1) / (kWarps * 16));
  const dim3 grid((unsigned)(m_blocks * num_pid_n)), block(kWarps * 32);
  const float* tw = topk_weights.has_value() ? (const float*)topk_weights->data_ptr() : nullptr;
  const int tiles = (int)(Rtot / 16);
  auto launch = [&](auto glu_tag) {
    constexpr bool GLU = decltype(glu_tag)::value;
    moe_w4_kernel<GLU><<<grid, block, 0, stream>>>(
        (const int8_t*)xq.data_ptr(), (const float*)xs.data_ptr(), (const int*)xsum.data_ptr(), (int)a_row_div,
        (const uint8_t*)w.data_ptr(), (const uint16_t*)ws.data_ptr(), tiles, GLU ? (int)(R / 16) : 0,
        (uint16_t*)out.data_ptr(), tw, (const int*)sorted_token_ids.data_ptr(), (const int*)expert_ids.data_ptr(),
        (const int*)num_tokens_post_padded.data_ptr(), (int)R, (int)K, (int)num_slots, num_pid_n);
  };
  if (glu) launch(std::true_type{}); else launch(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
