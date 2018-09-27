// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"

// *****************************************************************************
#ifdef __NVCC__
__inline__ __device__ double4 operator*(double4 a, double4 b) {
  return make_double4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
#include <cub/cub.cuh>

// *****************************************************************************
static double cub_vector_dot(const int N,
                             const double* __restrict vec1,
                             const double* __restrict vec2) {
  static double *h_dot = NULL;
  if (!h_dot) h_dot = (double*)mfem::rmalloc<double>::operator new(1,true);
  static double *d_dot = NULL;
  if (!d_dot) d_dot=(double*)mfem::rmalloc<double>::operator new(1);
  static void *d_storage = NULL;
  static size_t storage_bytes = 0;
  if (!d_storage){
    cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
    d_storage = mfem::rmalloc<char>::operator new(storage_bytes);
  }
  cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
  mfem::rmemcpy::rDtoH(h_dot,d_dot,sizeof(double));
  return *h_dot;
}
#endif // __NVCC__

#ifdef __HIPCC__

extern "C" __global__ void vector_dot0(const int N,
                                   const int Nblocks,
                                   const double* __restrict vec1,
                                   const double* __restrict vec2,
                                   double *storage){

  __shared__ volatile double s_dot[CUDA_BLOCK_SIZE];

  int t = threadIdx.x;
  int b = blockIdx.x;
  int id = CUDA_BLOCK_SIZE * b + t;
  s_dot[t] = 0.0;
  while (id<N) {
    s_dot[t] += vec1[id]*vec2[id];
    id += CUDA_BLOCK_SIZE*Nblocks;
  }

  __syncthreads();

#if CUDA_BLOCK_SIZE>512
  if(t<512) s_dot[t] += s_dot[t+512];
  __syncthreads();
#endif

#if CUDA_BLOCK_SIZE>256
  if(t<256) s_dot[t] += s_dot[t+256];
  __syncthreads();
#endif

  if(t<128) s_dot[t] += s_dot[t+128];
  __syncthreads();

  if(t< 64) s_dot[t] += s_dot[t+64];
  __syncthreads();

  if(t< 32) s_dot[t] += s_dot[t+32];
  //    __syncthreads();

  if(t< 16) s_dot[t] += s_dot[t+16];
  //    __syncthreads();

  if(t<  8) s_dot[t] += s_dot[t+8];
  //    __syncthreads();

  if(t<  4) s_dot[t] += s_dot[t+4];
  //    __syncthreads();

  if(t<  2) s_dot[t] += s_dot[t+2];
  //    __syncthreads();

  if(t<  1) storage[b] = s_dot[1] + s_dot[0];
}


extern "C" __global__ void vector_dot1(const int N,
                                   const int Nblocks,
                                   const double* __restrict storage,
                                   double *sum){

  __shared__ volatile double s_dot[CUDA_BLOCK_SIZE];

  int t = threadIdx.x;
  int b = blockIdx.x;
  int id = CUDA_BLOCK_SIZE * b + t;
  s_dot[t] = 0.0;
  while (id<N) {
    s_dot[t] += storage[id];
    id += CUDA_BLOCK_SIZE*Nblocks;
  }

  __syncthreads();

#if CUDA_BLOCK_SIZE>512
  if(t<512) s_dot[t] += s_dot[t+512];
  __syncthreads();
#endif

#if CUDA_BLOCK_SIZE>256
  if(t<256) s_dot[t] += s_dot[t+256];
  __syncthreads();
#endif

  if(t<128) s_dot[t] += s_dot[t+128];
  __syncthreads();

  if(t< 64) s_dot[t] += s_dot[t+64];
  __syncthreads();

  if(t< 32) s_dot[t] += s_dot[t+32];
  //    __syncthreads();

  if(t< 16) s_dot[t] += s_dot[t+16];
  //    __syncthreads();

  if(t<  8) s_dot[t] += s_dot[t+8];
  //    __syncthreads();

  if(t<  4) s_dot[t] += s_dot[t+4];
  //    __syncthreads();

  if(t<  2) s_dot[t] += s_dot[t+2];
  //    __syncthreads();

  if(t<  1) sum[b] = s_dot[1] + s_dot[0];
}

static double hip_vector_dot(const int N,
                             const double* __restrict vec1,
                             const double* __restrict vec2) {
  static double *h_min = NULL;
  if (!h_min) h_min = (double*)mfem::rmalloc<double>::operator new(1,true);
  static double *d_min = NULL;
  if (!d_min) d_min=(double*)mfem::rmalloc<double>::operator new(1);
  static void *d_storage = NULL;
  static size_t storage_bytes = 0;

  int NmaxBlocks = 128;
  int Nthreads = CUDA_BLOCK_SIZE;

  if (!d_storage){
    storage_bytes = NmaxBlocks*sizeof(double);
    d_storage = mfem::rmalloc<char>::operator new(storage_bytes);
  }

  int Nblocks = (N+Nthreads-1)/Nthreads;
  Nblocks = (Nblocks > NmaxBlocks) ? NmaxBlocks : Nblocks;

  //two phase reduction
  hipLaunchKernelGGL((vector_dot0),dim3(Nblocks),dim3(Nthreads),0,0,N,Nblocks,vec1,vec2,d_storage);
  hipLaunchKernelGGL((vector_dot1),dim3(1),      dim3(Nthreads),0,0,Nblocks,1,d_storage,d_min);
  mfem::rmemcpy::rDtoH(h_min,d_min,sizeof(double));
  return *h_min;
}
#endif //__HIPCC__

// *****************************************************************************
double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
  push(dot,Cyan);
#ifdef __NVCC__
  if (mfem::rconfig::Get().Cuda()){
    const double result = cub_vector_dot(N,vec1,vec2);
    pop();
    return result;
  }
#endif
#ifdef __HIPCC__
  if (mfem::rconfig::Get().Hip()){
    const double result = hip_vector_dot(N,vec1,vec2);
    pop();
    return result;
  }
#endif
  ReduceDecl(Sum,dot,0.0);
  ReduceForall(i,N,dot += vec1[i]*vec2[i];);
  pop();
  return dot;
}
