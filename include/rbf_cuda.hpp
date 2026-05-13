#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

constexpr int kRbfDim = 6;

// ---------------------------------------------------------------------------
// C API Declarations
// ---------------------------------------------------------------------------

extern "C" {

  // -------------------------------------------------------------------------
  // Activation Helpers
  // -------------------------------------------------------------------------

  // rbf_init_and_allocf
  // Allocate device storage for the activation vector S.
  cudaError_t rbf_init_and_allocf(void** dS, uint64_t N);

  // rbf_freef
  // Free a device allocation created by the CUDA RBF layer.
  cudaError_t rbf_freef(void* dS);

  // rbf_upload_xf
  // Upload the current normalized query vector x.
  cudaError_t rbf_upload_xf(const float x_host[kRbfDim]);

  // rbf_upload_boundsf_async
  // Upload the per-dimension lower bounds and step sizes.
  cudaError_t rbf_upload_boundsf_async(const float lo_host[kRbfDim],
                                       const float step_host[kRbfDim],
                                       cudaStream_t stream);

  // rbf_launchf
  // Launch the kernel that computes the current activation vector S(x).
  void rbf_launchf(void* dS,
                    int ne,
                    float lambda,
                    uint64_t N,
                    cudaStream_t stream);

  // rbf_download_Sf
  // Download the current activation vector S from device memory.
  cudaError_t rbf_download_Sf(void* dS, float* S_host, uint64_t N);

  // -------------------------------------------------------------------------
  // Weight Helpers
  // -------------------------------------------------------------------------

  // rbf_alloc_w4
  // Allocate device storage for four back-to-back weight vectors [w0..w3].
  cudaError_t rbf_alloc_w4(void** dW4, uint64_t N);

  // rbf_zero_w4
  // Zero the full W4 device buffer.
  cudaError_t rbf_zero_w4(void* dW4, uint64_t N);

  // rbf_update_w4
  // Launch the local-only weight update for one controller step.
  void rbf_update_w4(void* dW4,
                      const void* dS,
                      const float z2_host[4],
                      float gamma1,
                      float sigma,
                      uint64_t N,
                      cudaStream_t stream);

  // rbf_update_w4_cooperative
  // Launch the cooperative weight update for one controller step.
  cudaError_t rbf_update_w4_cooperative(void* dW4,
                                        const void* dS,
                                        const float z2_host[4],
                                        const float* neighbor_sum_host,
                                        float degree_k,
                                        float gamma1,
                                        float gamma2,
                                        float sigma,
                                        uint64_t N,
                                        cudaStream_t stream);

  // rbf_dot4
  // Compute F = W4 * S and copy the four controller outputs to the host.
  cudaError_t rbf_dot4(void* dW4,
                        const void* dS,
                        uint64_t N,
                        float F_host[4],
                        cudaStream_t stream);

  // rbf_upload_w4
  // Upload the full W4 weight buffer from host memory to device memory.
  cudaError_t rbf_upload_w4(void* dW4, const float* W_host, uint64_t N);

  // rbf_download_w4
  // Download the full W4 weight buffer from device memory.
  cudaError_t rbf_download_w4(void* dW4, float* W_host, uint64_t N);

  // -------------------------------------------------------------------------
  // Center Table Helpers
  // -------------------------------------------------------------------------

  // rbf_alloc_centersf
  // Allocate device storage for the explicit center table.
  cudaError_t rbf_alloc_centersf(void** dC, uint64_t N);

  // rbf_build_centersf
  // Build the explicit center table from the implicit grid definition.
  void rbf_build_centersf(void* dC,
                          int ne,
                          uint64_t N,
                          cudaStream_t stream);

  // rbf_download_centersf
  // Download the full center table from device memory.
  cudaError_t rbf_download_centersf(void* dC,
                                    float* C_host,
                                    uint64_t N);

  // rbf_download_centerf
  // Download a single center vector from device memory.
  cudaError_t rbf_download_centerf(void* dC,
                                   uint64_t idx,
                                   float out_host[kRbfDim]);
}

class CudaRBF {
 public:
  using scalar_t = float;  // GPU storage/compute type.

  // CudaRBF
  // Construct the CUDA-backed RBF helper and allocate device resources.
  CudaRBF(int ne,
          const float lo[kRbfDim],
          const float hi[kRbfDim],
          float lambda)
      : ne_(ne), lambda_(lambda) {
    for (int d = 0; d < kRbfDim; ++d) {
      lo_[d] = lo[d];
      hi_[d] = hi[d];
      step_[d] =
          (ne_ <= 1) ? 0.0f : (hi_[d] - lo_[d]) / float(ne_ - 1);
    }

    N_ = 1;
    for (int d = 0; d < kRbfDim; ++d) {
      N_ *= static_cast<uint64_t>(ne_);
    }

    bytes_S_ = static_cast<size_t>(N_) * sizeof(scalar_t);
    bytes_W4_ =
        static_cast<size_t>(4) * static_cast<size_t>(N_) * sizeof(scalar_t);
    bytes_C_ = static_cast<size_t>(N_) * kRbfDim * sizeof(scalar_t);

    cudaError_t e = cudaStreamCreate(&stream_);
    if (e != cudaSuccess) {
      stream_ = nullptr;
      last_status_ = e;
      return;
    }

    last_status_ = rbf_upload_boundsf_async(lo_, step_, stream_);
    if (last_status_ != cudaSuccess) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
      return;
    }

    last_C_ = rbf_alloc_centersf(&dC_, N_);
    if (last_C_ != cudaSuccess) {
      dC_ = nullptr;
    }

    if (dC_) {
      rbf_build_centersf(dC_, ne_, N_, stream_);
      cudaStreamSynchronize(stream_);
    }

    last_S_ = rbf_init_and_allocf(&dS_, N_);
    if (last_S_ != cudaSuccess) {
      dS_ = nullptr;
    }

    last_W4_ = rbf_alloc_w4(&dW4_, N_);
    if (last_W4_ != cudaSuccess) {
      dW4_ = nullptr;
    }

    if (dW4_) {
      rbf_zero_w4(dW4_, N_);
    }
  }

  // ~CudaRBF
  // Release device allocations and the CUDA stream owned by this object.
  ~CudaRBF() {
    if (dS_) rbf_freef(dS_);
    if (dW4_) rbf_freef(dW4_);
    if (dC_) rbf_freef(dC_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  // download_center
  // Download one center vector from the explicit device center table.
  cudaError_t download_center(uint64_t idx, float out[kRbfDim]) const {
    if (!dC_) return cudaErrorUnknown;
    return rbf_download_centerf(dC_, idx, out);
  }

  // download_centers
  // Download the full explicit center table into a host vector.
  cudaError_t download_centers(std::vector<float>& out) const {
    if (!dC_) return cudaErrorUnknown;
    out.resize((size_t)N_ * kRbfDim);
    return rbf_download_centersf(dC_, out.data(), N_);
  }

  // ready
  // Report whether the CUDA resources required for this helper are ready.
  bool ready() const {
    return (last_status_ == cudaSuccess) && dS_ && dW4_ && dC_;
  }

  // last_status
  // Return the last initialization-level CUDA status observed by this object.
  cudaError_t last_status() const { return last_status_; }

  // download_W
  // Download the full W4 weight buffer into a caller-provided host buffer.
  cudaError_t download_W(float* host_buf) const {
    if (!dW4_) return cudaErrorUnknown;
    return rbf_download_w4(dW4_, host_buf, N_);
  }

  // num_points
  // Return the number of RBF points in the implicit grid.
  std::uint64_t num_points() const { return N_; }

  // copy_W_to_host
  // Copy the current W4 weight buffer into a host vector.
  void copy_W_to_host(std::vector<float>& out) const;

  // upload_W
  // Copy a host-side W4 weight buffer into device memory.
  cudaError_t upload_W(const float* host_buf) {
    if (!dW4_ || !host_buf) return cudaErrorInvalidValue;
    return rbf_upload_w4(dW4_, host_buf, N_);
  }

  // bytes_S
  // Return the size in bytes of the activation vector buffer.
  std::size_t bytes_S() const { return bytes_S_; }

  // bytes_W4
  // Return the size in bytes of the stacked W4 weight buffer.
  std::size_t bytes_W4() const { return bytes_W4_; }

  // last_status_S
  // Return the CUDA status from allocating the S buffer.
  cudaError_t last_status_S() const { return last_S_; }

  // last_status_W4
  // Return the CUDA status from allocating the W4 buffer.
  cudaError_t last_status_W4() const { return last_W4_; }

  // dS
  // Expose the raw device pointer for the activation vector S.
  void* dS() const { return dS_; }

  // dW4
  // Expose the raw device pointer for the stacked W4 weight buffer.
  void* dW4() const { return dW4_; }

  // S_device
  // Expose the activation buffer as a const raw device pointer.
  const void* S_device() const { return dS_; }

  // W4_device
  // Expose the weight buffer as a const raw device pointer.
  const void* W4_device() const { return dW4_; }

  // compute_S
  // Compute S(x) for the current normalized query.
  void compute_S(const float x_host[kRbfDim]) {
    rbf_upload_xf(x_host);
    rbf_launchf(dS_, ne_, lambda_, N_, stream_);
  }

  // dot_F
  // Compute F = W4 * S and return the four controller outputs on the host.
  void dot_F(float F_host[4]) {
    rbf_dot4(dW4_, dS_, N_, F_host, stream_);
  }

  // update_w
  // Apply the local-only weight update using the current z2 error vector.
  void update_w(const float z2[4], float gamma1, float sigma) {
    rbf_update_w4(dW4_, dS_, z2, gamma1, sigma, N_, stream_);
  }

  // update_w_cooperative
  // Apply the cooperative weight update using the current z2 error vector
  // and the aggregated neighbor weights.
  cudaError_t update_w_cooperative(const float z2[4],
                                    const float* neighbor_sum_host,
                                    float degree_k,
                                    float gamma1,
                                    float gamma2,
                                    float sigma) {
    return rbf_update_w4_cooperative(dW4_,
                                      dS_,
                                      z2,
                                      neighbor_sum_host,
                                      degree_k,
                                      gamma1,
                                      gamma2,
                                      sigma,
                                      N_,
                                      stream_);
  }

  // copy_S_to_host
  // Copy the current activation vector S into a host buffer.
  void copy_S_to_host(float* out) const;

 private:
  // Grid configuration.
  int ne_{0};
  float lo_[kRbfDim] = {0};
  float hi_[kRbfDim] = {0};
  float step_[kRbfDim] = {0};
  scalar_t lambda_{1.5f};
  std::uint64_t N_{0};

  // Device buffers.
  void* dS_{nullptr};   // N floats
  void* dW4_{nullptr};  // 4 * N floats (w0..w3)
  void* dC_{nullptr};   // N * kRbfDim floats

  // Buffer sizes and allocation status.
  std::size_t bytes_S_{0};
  std::size_t bytes_W4_{0};
  std::size_t bytes_C_{0};
  cudaError_t last_S_{cudaSuccess};
  cudaError_t last_W4_{cudaSuccess};
  cudaError_t last_C_{cudaSuccess};
  cudaError_t last_status_{cudaSuccess};

  // CUDA stream used by this helper.
  cudaStream_t stream_{nullptr};
};
