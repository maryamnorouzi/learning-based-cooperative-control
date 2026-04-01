#include "rbf_cuda.hpp"
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>
#include <cstdio>

using T = float;               // <-- GPU math in float precision
__constant__ T c_x[8];         // 8-D query point on device (float), normalized 
__constant__ T c_z2[4];        // <-- NEW: z2 (same for all threads)
__constant__ T c_lo[8];
__constant__ T c_step[8];


// --- RBF kernel: fills S -> 8D grid coordinate (base-ne mixed radix), then RBF.
// __global__ void rbf_kernel(T* __restrict__ S,
//                            int ne, T lo, T step, T inv_lambda,
//                            uint64_t N)
// {
//     uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     uint64_t q = i;
//     T d2 = 0.0f;

// // **************** --- S computation --- **************** 
// #pragma unroll 8
//     for (int d = 0; d < 8; ++d) {
//         int digit = static_cast<int>(q % (uint64_t)ne);
//         q /= (uint64_t)ne;
//         T z = lo + T(digit) * step;   // evenly spaced in [lo, hi]
//         T e = c_x[d] - z;
//         d2 += e * e;
//     }
//     // float-precision exp
//     S[i] = __expf(-d2 * inv_lambda);
// }

__global__ void rbf_kernel(T* S, int ne, T inv_lambda, uint64_t N)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint64_t q = i;
    T d2 = 0.0f;

    #pragma unroll 8
    for (int d = 0; d < 8; ++d) {
        int digit = (int)(q % (uint64_t)ne);
        q /= (uint64_t)ne;

        T z = c_lo[d] + T(digit) * c_step[d];   // ✅ per-dimension
        T e = c_x[d] - z;
        d2 += e * e;
    }

    S[i] = __expf(-d2 * inv_lambda);
}




// ---------------- Centers kernel ----------------
// __global__ void centers_kernel(float* __restrict__ C8,
//                                int ne, float lo, float step,
//                                uint64_t N)
__global__ void centers_kernel(float* C8, int ne, uint64_t N)
{
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  uint64_t q = i;

  #pragma unroll 8
  for (int d = 0; d < 8; ++d) {
    int digit = (int)(q % (uint64_t)ne);
    q /= (uint64_t)ne;
    // C8[i * 8 + d] = lo + (float)digit * step;
    C8[i * 8 + d] = c_lo[d] + (float)digit * c_step[d];

  }
}

// ---------------- C-callable wrappers ----------------
extern "C" cudaError_t rbf_upload_bounds8f_async(const float lo8_host[8],
                                                const float step8_host[8],
                                                cudaStream_t stream)
{
    cudaError_t e;
    e = cudaMemcpyToSymbolAsync(c_lo, lo8_host, 8*sizeof(T), 0,
                                cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;

    e = cudaMemcpyToSymbolAsync(c_step, step8_host, 8*sizeof(T), 0,
                                cudaMemcpyHostToDevice, stream);
    return e;
}



extern "C" void rbf_build_centers8f(void* dC8, int ne,
                                   uint64_t N, cudaStream_t stream)
{
    if (!dC8) return;

    int threads = 256;
    int blocks  = (int)((N + threads - 1) / threads);

    centers_kernel<<<blocks, threads, 0, stream>>>((float*)dC8, ne, N);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("centers_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


extern "C" {

cudaError_t rbf_alloc_centers8f(void** dC8, uint64_t N)
{
  if (!dC8) return cudaErrorInvalidValue;
  return cudaMalloc(dC8, (size_t)N * 8 * sizeof(float));
}

// void rbf_build_centers8f(void* dC8, int ne, float lo, float hi,
//                          uint64_t N, cudaStream_t stream)
// {
//   if (!dC8) return;

//   float step = (ne == 1) ? 0.0f : (hi - lo) / float(ne - 1);

//   int threads = 256;
//   int blocks  = (int)((N + threads - 1) / threads);

//   centers_kernel<<<blocks, threads, 0, stream>>>((float*)dC8, ne, lo, step, N);

//   cudaError_t err = cudaPeekAtLastError();
//   if (err != cudaSuccess) {
//     printf("centers_kernel launch error: %s\n", cudaGetErrorString(err));
//   }
// }





cudaError_t rbf_download_centers8f(void* dC8, float* C_host, uint64_t N)
{
  if (!dC8 || !C_host) return cudaErrorInvalidValue;
  return cudaMemcpy(C_host, dC8, (size_t)N * 8 * sizeof(float),
                    cudaMemcpyDeviceToHost);
}

cudaError_t rbf_download_center8f(void* dC8, uint64_t idx, float out8_host[8])
{
  if (!dC8 || !out8_host) return cudaErrorInvalidValue;
  return cudaMemcpy(out8_host,
                    ((float*)dC8) + idx * 8,
                    8 * sizeof(float),
                    cudaMemcpyDeviceToHost);
}

} // extern "C"





// **************** --- Updating law --- **************** 
// --- update w1..w6: dw_i[j] = gamma1*( S[j]*z2[i] - sigma*w_i[j] ) --- 
__global__ void update_w6_kernel(T* __restrict__ W6,  // size = 4*N
                                 const T* __restrict__ S,
                                 T gamma1, T sigma,
                                 uint64_t N)  //  const T* __restrict__ z2, // 4
{
    uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    T s = S[j];

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        T* wi  = W6 + uint64_t(i) * N;
        T wj   = wi[j];
        wi[j] += -gamma1 * (s * c_z2[i] + sigma * wj);
    }
}

// --- dot reduce: F[i] = sum_j w_i[j] * S[j], i=0..5 ---
// S is a vector of length N.
// W6 holds six weight vectors, each length N, laid out back-to-back in memory:
// W6 = [w0(0..N-1), w1(0..N-1), …, w5(0..N-1)].
// F6[4] is the output: F6[i] = wi · S.

// simple grid-stride + atomicAdd to 4 accumulators
__global__ void dot_w6_kernel(const T* __restrict__ W6,
                              const T* __restrict__ S,
                              uint64_t N,
                              T* __restrict__ F6) // 4
{
    T acc[4] = {0,0,0,0};
    for (uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
         j < N;
         j += uint64_t(blockDim.x) * gridDim.x)
    {
        T sj = S[j];
#pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            const T* wi = W6 + uint64_t(i) * N;
            acc[i] += wi[j] * sj;
        }
    }
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        atomicAdd(&F6[i], acc[i]);
    }
}

void CudaRBF::copy_S_to_host(float* out) const {
  if (!dS_ || !out) return;
  // Optionally you can store the status in last_status_ if you want
  rbf_download_Sf(dS_, out, N_);
}

void CudaRBF::copy_W_to_host(std::vector<float>& out) const {
  if (!dW6_) return;
  out.resize(4 * N_);
  cudaMemcpy(out.data(), dW6_, 4 * N_ * sizeof(scalar_t), cudaMemcpyDeviceToHost);
}

// ---- C-callable helpers (used by C++ wrapper) ----
extern "C" cudaError_t rbf_init_and_allocf(void** dS, uint64_t N) {
    return cudaMalloc(dS, N * sizeof(T));
}

extern "C" cudaError_t rbf_freef(void* dS) {
    return cudaFree(dS);
}

extern "C" cudaError_t rbf_upload_xf(const float x_host[8]) {
    return cudaMemcpyToSymbol(c_x, x_host, 8 * sizeof(T), 0, cudaMemcpyHostToDevice);
}

// extern "C" void rbf_launchf(void* dS,
//                            int ne, T lo, T hi, T lambda,
//                            uint64_t N, cudaStream_t stream)
// {
//     const T step = (ne == 1) ? T(0) : (hi - lo) / T(ne - 1);
//     // const T inv_lambda = T(1) / lambda;
//     const T inv_lambda = (lambda > T(0)) ? (T(1)/lambda) : T(0);

//     const int threads = 256;
//     const int blocks  = (int)((N + threads - 1) / threads);
//     rbf_kernel<<<blocks, threads, 0, stream>>>((T*)dS, ne, lo, step, inv_lambda, N);
    
//     // --- optional error checks right AFTER the launch ---
//     cudaError_t err = cudaPeekAtLastError();         // catches launch config errors
//     if (err != cudaSuccess) {
//         printf("rbf_kernel launch error: %s\n", cudaGetErrorString(err));
//     }
// }

extern "C" void rbf_launchf(void* dS, int ne, float lambda,
                           uint64_t N, cudaStream_t stream)
{
    const T inv_lambda = (lambda > 0.0f) ? (T(1)/T(lambda)) : T(0);

    const int threads = 256;
    const int blocks  = (int)((N + threads - 1) / threads);

    rbf_kernel<<<blocks, threads, 0, stream>>>((T*)dS, ne, inv_lambda, N);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("rbf_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


// --- weights memory helpers (4*N floats) ---
extern "C" cudaError_t rbf_alloc_w6(void** dW6, uint64_t N) {
    return cudaMalloc(dW6, 4 * N * sizeof(T));
}
extern "C" cudaError_t rbf_zero_w6(void* dW6, uint64_t N) {
    return cudaMemset(dW6, 0, 4 * N * sizeof(T));
}

extern "C" void rbf_update_w6(void* dW6, const void* dS,
                              const float z2_host[4],
                              float gamma1, float sigma,
                              uint64_t N, cudaStream_t stream)
{

    // Upload z2 to constant memory (broadcast-friendly)
    cudaMemcpyToSymbolAsync(c_z2, z2_host, 4*sizeof(T), 0, cudaMemcpyHostToDevice, stream);

    const int threads = 256;
    const int blocks  = (int)((N + threads - 1) / threads);

    update_w6_kernel<<<blocks, threads, 0, stream>>>(
        (T*)dW6, (const T*)dS, (T)gamma1, (T)sigma, N);

    cudaPeekAtLastError(); // optional: check launch errors

}

extern "C" cudaError_t rbf_dot6(void* dW6, const void* dS,
                                uint64_t N, float F_host[4],
                                cudaStream_t stream)
{
    using T = float;
    T* dF = nullptr;
    cudaError_t err = cudaMalloc(&dF, 4*sizeof(T));
    if (err != cudaSuccess) return err;

    cudaMemsetAsync(dF, 0, 4*sizeof(T), stream);

    const int threads = 256;
    const int blocks  = 256;
    dot_w6_kernel<<<blocks, threads, 0, stream>>>(
        (const T*)dW6, (const T*)dS, N, dF);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) { cudaFree(dF); return err; }

    T F_dev[4];
    err = cudaMemcpyAsync(F_dev, dF, 4*sizeof(T),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { cudaFree(dF); return err; }

    cudaStreamSynchronize(stream);  // ensure results ready
    cudaFree(dF);

    #pragma unroll 4
    for (int i=0;i<4;++i) F_host[i] = (float)F_dev[i];
    return cudaSuccess;
}



extern "C" cudaError_t rbf_download_Sf(void* dS, float* S_host, uint64_t N) {
    return cudaMemcpy(S_host, dS, N * sizeof(T), cudaMemcpyDeviceToHost);
}

extern "C" cudaError_t rbf_download_w6(void* dW6, float* W_host, uint64_t N)
{
    // 4*N floats
    return cudaMemcpy(W_host, dW6, 4 * N * sizeof(float), cudaMemcpyDeviceToHost);
}
