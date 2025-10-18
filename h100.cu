// gemm_h100_bench.cu
// Target: NVIDIA H100 (SM_90a)
// Build: nvcc -O3 -std=c++17 -arch=sm_90a gemm_h100_bench.cu -lcublas -lcublasLt
// NOTE: Requires CUDA 12.x

#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cublasLt.lib")
#pragma comment(lib,"cudart.lib")

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(x) do{auto _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__, cudaGetErrorString(_e)); exit(1);} }while(0)
#define CHECK_CUBLAS(x) do{auto _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n", __FILE__,__LINE__, int(_s)); exit(1);} }while(0)

// Switch problem size if you want larger matrices
static constexpr int M = 8192;
static constexpr int N = 8192;
static constexpr int K = 8192;
static constexpr int ITERS = 20;

// GELU pieces for fallback/post-kernel
__device__ __forceinline__ float gelu(float x){
  const float kAlpha=0.79788456f, kBeta=0.044715f;
  float u = x + kBeta*x*x*x;
  return 0.5f * x * (1.0f + tanhf(kAlpha*u));
}
__global__ void k_gelu_add_residual(float* __restrict__ Y,
                                    const float* __restrict__ R,
                                    int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<size){
    float x = Y[i];
    const float kAlpha=0.79788456f, kBeta=0.044715f;
    float u = x + kBeta*x*x*x;
    float g = 0.5f * x * (1.0f + tanhf(kAlpha*u));
    Y[i] = g + R[i];
  }
}
__global__ void k_add_bias(float* __restrict__ Y,
                           const float* __restrict__ bias,
                           int M, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int size = M * N;
  if(i<size){
    int c = i % N;
    Y[i] += bias[c];
  }
}
__global__ void k_add_residual(float* __restrict__ Y,
                               const float* __restrict__ R,
                               int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<size) Y[i] += R[i];
}

float elapsed(cudaEvent_t a, cudaEvent_t b){ float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,a,b)); return ms; }

// Try to set GELU+BIASe pilogue; fallback to BIAS-only
bool try_set_bias_gelu_epilogue(cublasLtMatmulDesc_t opDesc, const void* biasDevPtr){
  cublasStatus_t s;
#ifdef CUBLASLT_EPILOGUE_BIAS_GELU
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS_GELU;
  s = cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  if(s==CUBLAS_STATUS_SUCCESS){
    s = cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasDevPtr, sizeof(biasDevPtr));
    if(s==CUBLAS_STATUS_SUCCESS) return true;
  }
#endif
#ifdef CUBLASLT_EPILOGUE_GELU_BIAS
  cublasLtEpilogue_t epi2 = CUBLASLT_EPILOGUE_GELU_BIAS;
  s = cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi2, sizeof(epi2));
  if(s==CUBLAS_STATUS_SUCCESS){
    s = cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasDevPtr, sizeof(biasDevPtr));
    if(s==CUBLAS_STATUS_SUCCESS) return true;
  }
#endif
  return false;
}
bool try_set_bias_only_epilogue(cublasLtMatmulDesc_t opDesc, const void* biasDevPtr){
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
  if(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)) != CUBLAS_STATUS_SUCCESS) return false;
  if(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasDevPtr, sizeof(biasDevPtr)) != CUBLAS_STATUS_SUCCESS) return false;
  return true;
}

// Simple autotune over a handful of Lt candidates
int pick_best_algo(cublasLtHandle_t lt,
                   cublasLtMatmulDesc_t opDesc,
                   cublasLtMatrixLayout_t layoutA,
                   cublasLtMatrixLayout_t layoutB,
                   cublasLtMatrixLayout_t layoutC,
                   const void* A, const void* B, void* C, void* D,
                   void* workspace, size_t workSize){
  const int maxAlgo = 32;
  std::vector<cublasLtMatmulHeuristicResult_t> results(maxAlgo);
  int returned = 0;

  cublasLtMatmulPreference_t pref; CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSize, sizeof(workSize)));
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      lt, opDesc, layoutA, layoutB, layoutC, layoutC,
      pref, maxAlgo, results.data(), &returned));
  cublasLtMatmulPreferenceDestroy(pref);

  if(returned == 0){ fprintf(stderr,"No cuBLASLt heuristic found.\n"); exit(1); }

  float alpha = 1.0f, beta = 0.0f;
  cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
  float best_ms = 1e30f; int best_idx = 0;

  // Warm + time each
  for(int i=0;i<returned;++i){
    CHECK_CUBLAS(cublasLtMatmul(lt, opDesc, &alpha, B, layoutB, A, layoutA, &beta,
                                C, layoutC, D, layoutC, &results[i].algo, workspace, workSize, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(s));
    CHECK_CUBLAS(cublasLtMatmul(lt, opDesc, &alpha, B, layoutB, A, layoutA, &beta,
                                C, layoutC, D, layoutC, &results[i].algo, workspace, workSize, 0));
    CHECK_CUDA(cudaEventRecord(e)); CHECK_CUDA(cudaEventSynchronize(e));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));
    if(ms<best_ms){ best_ms=ms; best_idx=i; }
  }
  CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
  return best_idx;
}

int main(){
  // Host buffers (FP32 host side)
  std::vector<float> hA(M*K), hB(K*N), hBias(N), hRes(M*N);
  for(int i=0;i<M*K;++i) hA[i]=0.5f+0.5f*sinf(0.001f*i);
  for(int i=0;i<K*N;++i) hB[i]=0.5f+0.5f*cosf(0.0013f*i);
  for(int i=0;i<N;++i)   hBias[i]=0.1f*sinf(0.00077f*i);
  for(int i=0;i<M*N;++i) hRes[i]=0.02f*((i%5)-2);

  // Device storage (we keep FP32 outputs for easy diff and post kernels)
  float *dY_base=nullptr, *dY_fused=nullptr, *dBias_f32=nullptr, *dRes=nullptr;
  CHECK_CUDA(cudaMalloc(&dY_base, sizeof(float)*M*N));
  CHECK_CUDA(cudaMalloc(&dY_fused, sizeof(float)*M*N));
  CHECK_CUDA(cudaMalloc(&dBias_f32, sizeof(float)*N));
  CHECK_CUDA(cudaMalloc(&dRes, sizeof(float)*M*N));
  CHECK_CUDA(cudaMemcpy(dBias_f32, hBias.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dRes,      hRes.data(),  sizeof(float)*M*N, cudaMemcpyHostToDevice));

  // For GEMM inputs use BF16 if available, else FP16
  bool use_bf16 = false;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // At runtime on H100 we will still create the BF16 layouts; fallback below if creation fails.
  use_bf16 = true;
#endif

  // Allocate A,B in chosen format; keep host staging in FP32 then cast
  void *dA=nullptr, *dB=nullptr;
  cudaDataType_t aType, bType, cType, dType;
  cublasComputeType_t computeType;

  if(use_bf16){
#ifdef CUDA_R_16BF
    aType = CUDA_R_16BF; bType = CUDA_R_16BF;
    computeType = CUBLAS_COMPUTE_32F; // Accumulate in FP32
    cType = CUDA_R_32F; dType = CUDA_R_32F;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(__nv_bfloat16)*M*K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(__nv_bfloat16)*K*N));
#else
    use_bf16 = false;
#endif
  }
  if(!use_bf16){
    aType = CUDA_R_16F; bType = CUDA_R_16F;
    computeType = CUBLAS_COMPUTE_32F;
    cType = CUDA_R_32F; dType = CUDA_R_32F;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(__half)*M*K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(__half)*K*N));
  }

  // Cast host FP32 -> device (BF16/FP16)
  // Simple cast kernels to avoid huge host-side conversions
  auto cast_fp32_to_half = [] __global__ (const float* src, __half* dst, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){ dst[i] = __float2half(src[i]); }
  };
#ifdef CUDA_R_16BF
  auto cast_fp32_to_bf16 = [] __global__ (const float* src, __nv_bfloat16* dst, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){ dst[i] = __float2bfloat16(src[i]); }
  };
#endif

  // Temp device buffers for staging
  float *dA_f32=nullptr, *dB_f32=nullptr;
  CHECK_CUDA(cudaMalloc(&dA_f32, sizeof(float)*M*K));
  CHECK_CUDA(cudaMalloc(&dB_f32, sizeof(float)*K*N));
  CHECK_CUDA(cudaMemcpy(dA_f32, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_f32, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));
  {
    int threads=256;
    int nA=M*K, nB=K*N;
    int blocksA=(nA+threads-1)/threads, blocksB=(nB+threads-1)/threads;
    if(use_bf16){
#ifdef CUDA_R_16BF
      cast_fp32_to_bf16<<<blocksA,threads>>>(dA_f32, ( __nv_bfloat16*)dA, nA);
      cast_fp32_to_bf16<<<blocksB,threads>>>(dB_f32, ( __nv_bfloat16*)dB, nB);
#endif
    }else{
      cast_fp32_to_half<<<blocksA,threads>>>(dA_f32, ( __half*)dA, nA);
      cast_fp32_to_half<<<blocksB,threads>>>(dB_f32, ( __half*)dB, nB);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(dA_f32); cudaFree(dB_f32);
  }

  // cuBLAS/cuBLASLt handles
  cublasHandle_t blas; CHECK_CUBLAS(cublasCreate(&blas));
  // H100: allow TF32/TC path if used; our computeType is 32F accumulate, inputs 16-bit use TC
  CHECK_CUBLAS(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH));
  cublasLtHandle_t lt; CHECK_CUBLAS(cublasLtCreate(&lt));

  // Descriptors
  cublasLtMatmulDesc_t opDesc;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, computeType, cType));
  cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));

  // Layouts
  cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, aType, K, M, K)); // lda=K
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, bType, N, K, N)); // ldb=N
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, cType, N, M, N)); // ldc=N

  // Workspace for autotune
  void* dWork=nullptr;
  size_t workSize = 512*1024*1024; // 512 MB helps Lt pick wide kernels
  CHECK_CUDA(cudaMalloc(&dWork, workSize));

  // Baseline (cuBLAS GemmEx + bias + gelu + residual)
  float alpha=1.0f, beta0=0.0f;
  cudaEvent_t t0,t1,t2,t3; CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaEventCreate(&t2)); CHECK_CUDA(cudaEventCreate(&t3));

  // Warm
  for(int w=0; w<2; ++w){
    CHECK_CUBLAS(cublasGemmEx(
      blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
      &alpha, dB, bType, N, dA, aType, K, &beta0, dY_base, cType, N,
      computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  int sizeMN=M*N, threads=256, blocks=(sizeMN+threads-1)/threads;
  CHECK_CUDA(cudaEventRecord(t0));
  for(int it=0; it<ITERS; ++it){
    CHECK_CUBLAS(cublasGemmEx(
      blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
      &alpha, dB, bType, N, dA, aType, K, &beta0, dY_base, cType, N,
      computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    k_add_bias<<<blocks,threads>>>(dY_base, dBias_f32, M, N);
    k_gelu_add_residual<<<blocks,threads>>>(dY_base, dRes, sizeMN);
  }
  CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
  float ms_base = elapsed(t0,t1);

  // Fused via cuBLASLt with epilogue
  bool have_bias_gelu = try_set_bias_gelu_epilogue(opDesc, dBias_f32);
  if(!have_bias_gelu){
    fprintf(stderr,"[Info] GELU+BIAS epilogue not available; falling back to BIAS-only.\n");
    if(!try_set_bias_only_epilogue(opDesc, dBias_f32)){ fprintf(stderr,"Epilogue set failed.\n"); exit(1); }
  }
  float beta = 0.0f;

  // Autotune to get fastest algo
  int best = pick_best_algo(lt, opDesc, layoutA, layoutB, layoutC,
                            dA, dB, dY_fused, dY_fused, dWork, workSize);

  // Re-fetch best algo object
  cublasLtMatmulHeuristicResult_t chosen{};
  {
    const int maxAlgo=32;
    std::vector<cublasLtMatmulHeuristicResult_t> results(maxAlgo);
    int ret=0;
    cublasLtMatmulPreference_t pref; CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSize, sizeof(workSize)));
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      lt, opDesc, layoutA, layoutB, layoutC, layoutC, pref, maxAlgo, results.data(), &ret));
    cublasLtMatmulPreferenceDestroy(pref);
    if(ret==0){ fprintf(stderr,"No heuristics on second fetch.\n"); exit(1); }
    chosen = results[best];
  }

  // Timed fused loop
  CHECK_CUDA(cudaEventRecord(t2));
  for(int it=0; it<ITERS; ++it){
    CHECK_CUBLAS(cublasLtMatmul(lt, opDesc, &alpha,
      dB, layoutB, dA, layoutA, &beta,
      /*C*/ dY_fused, layoutC,
      /*D*/ dY_fused, layoutC,
      &chosen.algo, dWork, workSize, 0));
    if(!have_bias_gelu){
      // need GELU; fuse with residual in one kernel
      k_gelu_add_residual<<<blocks,threads>>>(dY_fused, dRes, sizeMN);
    }else{
      // only residual left
      k_add_residual<<<blocks,threads>>>(dY_fused, dRes, sizeMN);
    }
  }
  CHECK_CUDA(cudaEventRecord(t3)); CHECK_CUDA(cudaEventSynchronize(t3));
  float ms_fused = elapsed(t2,t3);

  // Diff
  std::vector<float> hBase(M*N), hFused(M*N);
  CHECK_CUDA(cudaMemcpy(hBase.data(),  dY_base,  sizeof(float)*M*N, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hFused.data(), dY_fused, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
  double max_abs=0.0, mean_abs=0.0;
  for(size_t i=0;i<hBase.size();++i){
    double d = std::abs(double(hBase[i])-double(hFused[i]));
    if(d>max_abs) max_abs=d;
    mean_abs += d;
  }
  mean_abs /= double(hBase.size());

  printf("Datatype: %s, Compute: FP32 accumulate on TC\n", use_bf16? "BF16" : "FP16");
  printf("Baseline: %.3f ms total / %d iters -> %.3f ms/iter\n",
         ms_base, ITERS, ms_base/ITERS);
  printf("Fused   : %.3f ms total / %d iters -> %.3f ms/iter\n",
         ms_fused, ITERS, ms_fused/ITERS);
  printf("Speedup : %.2f %%\n", 100.0f*(ms_base - ms_fused)/ms_base);
  printf("Diff    : max_abs=%.3e, mean_abs=%.3e\n", max_abs, mean_abs);

  // Cleanup
  if(dWork) cudaFree(dWork);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatrixLayoutDestroy(layoutB);
  cublasLtMatrixLayoutDestroy(layoutC);
  cublasLtMatmulDescDestroy(opDesc);
  cublasLtDestroy(lt);
  cublasDestroy(blas);
  cudaFree(dA); cudaFree(dB); cudaFree(dBias_f32); cudaFree(dRes);
  cudaFree(dY_base); cudaFree(dY_fused);
  return 0;
}
