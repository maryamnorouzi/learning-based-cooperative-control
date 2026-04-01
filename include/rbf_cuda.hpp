#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>  


extern "C" {
  cudaError_t rbf_init_and_allocf(void** dS, uint64_t N);
  cudaError_t rbf_freef(void* dS);

  cudaError_t rbf_upload_xf(const float x_host[8]);

  //upload per-dimension lo and step
  cudaError_t rbf_upload_bounds8f_async(const float lo8_host[8],
                                        const float step8_host[8],
                                        cudaStream_t stream);

  void        rbf_launchf(void* dS, int ne,
                          float lambda, uint64_t N, cudaStream_t stream);

  cudaError_t rbf_alloc_w6(void** dW6, uint64_t N);
  cudaError_t rbf_zero_w6(void* dW6, uint64_t N);

  void        rbf_update_w6(void* dW6, const void* dS,
                            const float z2_host[4],
                            float gamma1, float sigma,
                            uint64_t N, cudaStream_t stream);

  cudaError_t rbf_dot6(void* dW6, const void* dS,
                       uint64_t N, float F_host[4], cudaStream_t stream);

  cudaError_t rbf_download_Sf(void* dS, float* S_host, uint64_t N);
  cudaError_t rbf_download_w6(void* dW6, float* W_host, uint64_t N);

  cudaError_t rbf_alloc_centers8f(void** dC8, uint64_t N);

  void        rbf_build_centers8f(void* dC8, int ne,
                                  uint64_t N, cudaStream_t stream);

  cudaError_t rbf_download_centers8f(void* dC8, float* C_host, uint64_t N);
  cudaError_t rbf_download_center8f(void* dC8, uint64_t idx, float out8_host[8]);
}




// ---- C++ wrapper class ----
class CudaRBF {
public:
  using scalar_t = float; // GPU storage/compute type
  
  CudaRBF(int ne, const float lo8[8], const float hi8[8], float lambda)
  : ne_(ne), lambda_(lambda)
  {
    // store bounds + compute per-dim step
    for (int d = 0; d < 8; ++d) {
      lo8_[d] = lo8[d];
      hi8_[d] = hi8[d];
      step8_[d] = (ne_ <= 1) ? 0.0f : (hi8_[d] - lo8_[d]) / float(ne_ - 1);
    }

    N_ = 1;
    for (int d = 0; d < 8; ++d) N_ *= static_cast<uint64_t>(ne_);

    bytes_S_  = static_cast<size_t>(N_) * sizeof(scalar_t);
    bytes_W6_ = static_cast<size_t>(4) * static_cast<size_t>(N_) * sizeof(scalar_t);
    bytes_C8_ = static_cast<size_t>(N_) * 8 * sizeof(scalar_t);

    // create stream FIRST
    cudaError_t e = cudaStreamCreate(&stream_);
    if (e != cudaSuccess) {
      stream_ = nullptr;
      last_status_ = e;
      return;
    }

    // upload per-dimension bounds ONCE
    // last_status_ = rbf_upload_bounds8f_async(lo8_, step8_, stream_);
    // if (last_status_ != cudaSuccess) return;

    last_status_ = rbf_upload_bounds8f_async(lo8_, step8_, stream_);
    if (last_status_ != cudaSuccess) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
      return;
}


    // Centers
    last_C8_ = rbf_alloc_centers8f(&dC8_, N_);
    if (last_C8_ != cudaSuccess) dC8_ = nullptr;

    if (dC8_) {
      rbf_build_centers8f(dC8_, ne_, N_, stream_);
      cudaStreamSynchronize(stream_);
    }

    // S
    last_S_ = rbf_init_and_allocf(&dS_, N_);
    if (last_S_ != cudaSuccess) dS_ = nullptr;

    // W
    last_W6_ = rbf_alloc_w6(&dW6_, N_);
    if (last_W6_ != cudaSuccess) dW6_ = nullptr;

    if (dW6_) rbf_zero_w6(dW6_, N_);
  }

  

  ~CudaRBF() { 
    if (dS_) rbf_freef(dS_); 
    if (dW6_) rbf_freef(dW6_);
    if (dC8_) rbf_freef(dC8_);
    if (stream_) cudaStreamDestroy(stream_);
  }


  // Download ONE center (8 floats) by index
  cudaError_t download_center(uint64_t idx, float out8[8]) const {
    if (!dC8_) return cudaErrorUnknown;
    return rbf_download_center8f(dC8_, idx, out8);
  }  

  // Download ALL centers (N*8 floats) to host vector
  cudaError_t download_centers(std::vector<float>& out) const {
    if (!dC8_) return cudaErrorUnknown;
    out.resize((size_t)N_ * 8);
    return rbf_download_centers8f(dC8_, out.data(), N_);
  }

  // bool ready() const { return dS_ && dW6_ && dC8_; }
  // bool ready() const { return dS_ && dW6_; }
  bool ready() const { return (last_status_ == cudaSuccess) && dS_ && dW6_ && dC8_; }

  cudaError_t last_status() const { return last_status_; }

  // Copy all 4*N weights from device to host
  cudaError_t download_W(float* host_buf) const {
      // host_buf must have size >= 4 * N_
      if (!dW6_) return cudaErrorUnknown;
      return rbf_download_w6(dW6_, host_buf, N_);
  }

  // // Host -> Device (copy 4*N floats into dW6_)
  // cudaError_t upload_W(const float* W_host) {
  //   if (!W_host) return cudaErrorInvalidValue;
  //   if (!dW6_)   return cudaErrorUnknown;
  //   return cudaMemcpy(dW6_, W_host, bytes_W6_, cudaMemcpyHostToDevice);
  // }


  std::uint64_t num_points()   const { return N_; }
  // Copy all weights (4×N) to host
  void copy_W_to_host(std::vector<float>& out) const;

  std::size_t   bytes_S()      const { return bytes_S_; }
  std::size_t   bytes_W6()     const { return bytes_W6_; }
  cudaError_t   last_status_S() const { return last_S_; }
  cudaError_t   last_status_W6() const { return last_W6_; }

  void* dS()  const { return dS_;  }
  void* dW6() const { return dW6_; }

  const void* S_device() const { return dS_; }
  const void* W6_device() const { return dW6_; }

  // Compute S for given 12D x (float)
  void compute_S(const float x_host[8]) {
    rbf_upload_xf(x_host);
    // rbf_launchf(dS_, ne_, lo_, hi_, lambda_, N_, stream_);
    rbf_launchf(dS_, ne_, lambda_, N_, stream_);

  }

  // Dot: F[i] = w_i^T S  -> writes 6 floats to F_host
  void dot_F(float F_host[4]) {
    rbf_dot6(dW6_, dS_, N_, F_host, stream_);
  }

  // Update weights using z2[6], gamma1, sigma
  void update_w(const float z2[4], float gamma1, float sigma) {
    rbf_update_w6(dW6_, dS_, z2, gamma1, sigma, N_, stream_);
  }


  // Copy S(x) from device to host
  void copy_S_to_host(float* out) const;

private:
  int ne_{0};
  float lo8_[8] = {0};
  float hi8_[8] = {0};
  float step8_[8] = {0};
  scalar_t  lambda_{1.5f}; //lo_{-1.0f}, hi_{1.0f},
  std::uint64_t N_{0};
  void* dS_{nullptr};   // N floats
  void* dW6_{nullptr};  // 6*N floats (w1..w6)

  std::size_t bytes_S_{0};
  std::size_t bytes_W6_{0};
  cudaError_t last_S_{cudaSuccess};
  cudaError_t last_W6_{cudaSuccess};
  cudaError_t last_status_{cudaSuccess};

  cudaStream_t stream_{nullptr}; 

  void* dC8_{nullptr};      // centers table: N*8 floats
  std::size_t bytes_C8_{0};
  cudaError_t last_C8_{cudaSuccess};

};
