#include "rbf_cuda.hpp"

#include <cuda_runtime.h>
#include <stdint.h>

#include <cmath>
#include <cstdio>

using T = float;

// ---------------------------------------------------------------------------
// Device Constants
// ---------------------------------------------------------------------------

__constant__ T c_x[8];
__constant__ T c_z2[4];
__constant__ T c_lo[8];
__constant__ T c_step[8];

// RBF update flow:
// 1. `rbf_kernel` computes the current basis activations S(x).
// 2. `dot_w4_kernel` computes the learned output F = W4 * S.
// 3. The caller chooses one update path per step:
//    - `update_w4_kernel` for local adaptation only
//    - `update_w4_cooperative_kernel` for local adaptation plus consensus
// 4. The cooperative kernel already contains the full local update term, so
//    the caller should not run both kernels in the same controller step.

// ---------------------------------------------------------------------------
// Device Kernels
// ---------------------------------------------------------------------------

// rbf_kernel
// Compute the current activation vector S(x) over the full 8-D center grid.
__global__ void rbf_kernel(
    T* S,
    int ne,
    T inv_lambda,
    uint64_t N) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint64_t q = i;
    T d2 = 0.0f;

    #pragma unroll 8
    for (int d = 0; d < 8; ++d) {
        int digit = (int)(q % (uint64_t)ne);
        q /= (uint64_t)ne;

        T z = c_lo[d] + T(digit) * c_step[d];
        T e = c_x[d] - z;
        d2 += e * e;
    }

    S[i] = __expf(-d2 * inv_lambda);
}

// centers_kernel
// Expand the implicit 8-D grid definition into an explicit center table.
__global__ void centers_kernel(
    float* C8,
    int ne,
    uint64_t N) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint64_t q = i;

    #pragma unroll 8
    for (int d = 0; d < 8; ++d) {
        int digit = (int)(q % (uint64_t)ne);
        q /= (uint64_t)ne;
        C8[i * 8 + d] = c_lo[d] + (float)digit * c_step[d];
    }
}

// update_w4_kernel
// Apply the local-only weight update using the current tracking error z2:
// dw_i[j] = -gamma1 * (S[j] * z2[i] + sigma * w_i[j]).
__global__ void update_w4_kernel(T* __restrict__ W4,
                                const T* __restrict__ S,
                                T gamma1,
                                T sigma,
                                uint64_t N) {
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    T s = S[j];

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        T* wi = W4 + uint64_t(i) * N;
        T wj = wi[j];
        wi[j] += -gamma1 * (s * c_z2[i] + sigma * wj);
    }
}

// update_w4_cooperative_kernel
// Apply the cooperative update. This includes the same local term as
// `update_w4_kernel` plus a neighbor-consensus term for each weight.
__global__ void update_w4_cooperative_kernel(T* __restrict__ W4,
                                            const T* __restrict__ S,
                                            const T* __restrict__ neighbor_sum,
                                            T degree_k,
                                            T gamma1,
                                            T gamma2,
                                            T sigma,
                                            uint64_t N) {
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    T s = S[j];

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        T* wi = W4 + uint64_t(i) * N;
        T wj = wi[j];
        const T neighbor_term = neighbor_sum[uint64_t(i) * N + j];
        wi[j] += -gamma1 * (s * c_z2[i] + sigma * wj)
                -gamma2 * (degree_k * wj - neighbor_term);
    }
}

// dot_w4_kernel
// Accumulate F = W4 * S for all four controller output channels.
__global__ void dot_w4_kernel(const T* __restrict__ W4,
                                const T* __restrict__ S,
                                uint64_t N,
                                T* __restrict__ F4) {
    T acc[4] = {0, 0, 0, 0};

    for (uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
        j < N;
        j += uint64_t(blockDim.x) * gridDim.x) {
        T sj = S[j];

        #pragma unroll 4
        for (int i = 0; i < 4; ++i) {
        const T* wi = W4 + uint64_t(i) * N;
        acc[i] += wi[j] * sj;
        }
    }

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        atomicAdd(&F4[i], acc[i]);
    }
}

// ---------------------------------------------------------------------------
// C API: Grid And Activation Helpers
// ---------------------------------------------------------------------------

// rbf_upload_bounds8f_async
// Upload the per-dimension lower bounds and step sizes to constant memory.
extern "C" cudaError_t rbf_upload_bounds8f_async(const float lo8_host[8],
                                                const float step8_host[8],
                                                cudaStream_t stream) {
    cudaError_t e = cudaMemcpyToSymbolAsync(c_lo, lo8_host, 8 * sizeof(T), 0,
                                                cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;

    return cudaMemcpyToSymbolAsync(c_step, step8_host, 8 * sizeof(T), 0,
                                    cudaMemcpyHostToDevice, stream);
}

// rbf_build_centers8f
// Launch the kernel that builds the explicit center table on the device.
extern "C" void rbf_build_centers8f(void* dC8,
                                    int ne,
                                    uint64_t N,
                                    cudaStream_t stream) {
    if (!dC8) return;

    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);

    centers_kernel<<<blocks, threads, 0, stream>>>((float*)dC8, ne, N);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("centers_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// rbf_alloc_centers8f
// Allocate device storage for the full 8-D center table.
extern "C" cudaError_t rbf_alloc_centers8f(void** dC8, uint64_t N) {
    if (!dC8) return cudaErrorInvalidValue;
    return cudaMalloc(dC8, (size_t)N * 8 * sizeof(float));
}

// rbf_download_centers8f
// Download the full center table from device memory to the host.
extern "C" cudaError_t rbf_download_centers8f(void* dC8,
                                                float* C_host,
                                                uint64_t N) {
    if (!dC8 || !C_host) return cudaErrorInvalidValue;
    return cudaMemcpy(C_host, dC8, (size_t)N * 8 * sizeof(float),
                        cudaMemcpyDeviceToHost);
}

// rbf_download_center8f
// Download a single 8-D center vector from device memory to the host.
extern "C" cudaError_t rbf_download_center8f(void* dC8,
                                            uint64_t idx,
                                            float out8_host[8]) {
    if (!dC8 || !out8_host) return cudaErrorInvalidValue;
    return cudaMemcpy(out8_host,
                        ((float*)dC8) + idx * 8,
                        8 * sizeof(float),
                        cudaMemcpyDeviceToHost);
}

// rbf_init_and_allocf
// Allocate device storage for the activation vector S.
extern "C" cudaError_t rbf_init_and_allocf(void** dS, uint64_t N) {
  return cudaMalloc(dS, N * sizeof(T));
}

// rbf_freef
// Free a device allocation created by this file.
extern "C" cudaError_t rbf_freef(void* dS) {
    return cudaFree(dS);
}

// rbf_upload_xf
// Upload the current normalized query vector x to constant memory.
extern "C" cudaError_t rbf_upload_xf(const float x_host[8]) {
  return cudaMemcpyToSymbol(c_x, x_host, 8 * sizeof(T), 0,
                            cudaMemcpyHostToDevice);
}

// rbf_launchf
// Launch the kernel that computes the current activation vector S(x).
extern "C" void rbf_launchf(void* dS,
                            int ne,
                            float lambda,
                            uint64_t N,
                            cudaStream_t stream) {
    const T inv_lambda = (lambda > 0.0f) ? (T(1) / T(lambda)) : T(0);
    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);

    rbf_kernel<<<blocks, threads, 0, stream>>>((T*)dS, ne, inv_lambda, N);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("rbf_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// rbf_download_Sf
// Download the activation vector S from device memory to the host.
extern "C" cudaError_t rbf_download_Sf(void* dS, float* S_host, uint64_t N) {
  return cudaMemcpy(S_host, dS, N * sizeof(T), cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// C API: Weight Helpers
// ---------------------------------------------------------------------------

// rbf_alloc_w4
// Allocate device storage for four back-to-back weight vectors [w0..w3].
extern "C" cudaError_t rbf_alloc_w4(void** dW4, uint64_t N) {
  return cudaMalloc(dW4, 4 * N * sizeof(T));
}

// rbf_zero_w4
// Zero the full W4 device buffer.
extern "C" cudaError_t rbf_zero_w4(void* dW4, uint64_t N) {
  return cudaMemset(dW4, 0, 4 * N * sizeof(T));
}

// rbf_update_w4
// Launch the local-only weight update for one controller step.
extern "C" void rbf_update_w4(void* dW4,
                                const void* dS,
                                const float z2_host[4],
                                float gamma1,
                                float sigma,
                                uint64_t N,
                                cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(c_z2, z2_host, 4 * sizeof(T), 0,
                            cudaMemcpyHostToDevice, stream);

    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);

    update_w4_kernel<<<blocks, threads, 0, stream>>>(
        (T*)dW4, (const T*)dS, (T)gamma1, (T)sigma, N);

    cudaPeekAtLastError();
}

// rbf_update_w4_cooperative
// Launch the cooperative weight update for one controller step.
extern "C" cudaError_t rbf_update_w4_cooperative(void* dW4,
                                                const void* dS,
                                                const float z2_host[4],
                                                const float* neighbor_sum_host,
                                                float degree_k,
                                                float gamma1,
                                                float gamma2,
                                                float sigma,
                                                uint64_t N,
                                                cudaStream_t stream) {
    if (!dW4 || !dS || !neighbor_sum_host) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaMemcpyToSymbolAsync(
        c_z2, z2_host, 4 * sizeof(T), 0, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }

    T* d_neighbor_sum = nullptr;
    err = cudaMalloc(&d_neighbor_sum, 4 * N * sizeof(T));
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMemcpyAsync(
        d_neighbor_sum,
        neighbor_sum_host,
        4 * N * sizeof(T),
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        cudaFree(d_neighbor_sum);
        return err;
    }

    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);

    update_w4_cooperative_kernel<<<blocks, threads, 0, stream>>>(
        (T*)dW4,
        (const T*)dS,
        d_neighbor_sum,
        (T)degree_k,
        (T)gamma1,
        (T)gamma2,
        (T)sigma,
        N);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        cudaFree(d_neighbor_sum);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaError_t free_err = cudaFree(d_neighbor_sum);

    if (err != cudaSuccess) {
        return err;
    }

    return free_err;
}

// rbf_dot4
// Compute F = W4 * S and copy the four controller outputs back to the host.
extern "C" cudaError_t rbf_dot4(void* dW4,
                                const void* dS,
                                uint64_t N,
                                float F_host[4],
                                cudaStream_t stream) {
    T* dF = nullptr;
    cudaError_t err = cudaMalloc(&dF, 4 * sizeof(T));
    if (err != cudaSuccess) return err;

    cudaMemsetAsync(dF, 0, 4 * sizeof(T), stream);

    const int threads = 256;
    const int blocks = 256;
    dot_w4_kernel<<<blocks, threads, 0, stream>>>(
        (const T*)dW4, (const T*)dS, N, dF);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        cudaFree(dF);
        return err;
    }

    T F_host_tmp[4];
    err = cudaMemcpyAsync(F_host_tmp, dF, 4 * sizeof(T),
                            cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dF);
        return err;
    }

    cudaStreamSynchronize(stream);  // Ensure results are ready on the host.
    cudaFree(dF);

    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        F_host[i] = (float)F_host_tmp[i];
    }

    return cudaSuccess;
}

// rbf_upload_w4
// Upload the full W4 weight buffer from host memory to device memory.
extern "C" cudaError_t rbf_upload_w4(void* dW4, const float* W_host, uint64_t N) {
  return cudaMemcpy(dW4, W_host, 4 * N * sizeof(float), cudaMemcpyHostToDevice);
}

// rbf_download_w4
// Download the full W4 weight buffer from device memory to the host.
extern "C" cudaError_t rbf_download_w4(void* dW4, float* W_host, uint64_t N) {
  return cudaMemcpy(W_host, dW4, 4 * N * sizeof(float),
                    cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// CudaRBF Convenience Methods
// ---------------------------------------------------------------------------

// CudaRBF::copy_S_to_host
// Copy the current activation vector S from the device into a host buffer.
void CudaRBF::copy_S_to_host(float* out) const {
    if (!dS_ || !out) return;
    rbf_download_Sf(dS_, out, N_);
}

// CudaRBF::copy_W_to_host
// Copy the current W4 weight buffer from the device into a host vector.
void CudaRBF::copy_W_to_host(std::vector<float>& out) const {
    if (!dW4_) return;
    out.resize(4 * N_);
    cudaMemcpy(out.data(), dW4_, 4 * N_ * sizeof(scalar_t),
                cudaMemcpyDeviceToHost);
}
