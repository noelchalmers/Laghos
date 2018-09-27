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
#include <cub/cub.cuh>

// *****************************************************************************
static double cub_vector_min(const int N,
                             const double* __restrict vec) {
  static double *h_min = NULL;
  if (!h_min) h_min = (double*)mfem::rmalloc<double>::operator new(1,true);
  static double *d_min = NULL;
  if (!d_min) d_min=(double*)mfem::rmalloc<double>::operator new(1);
  static void *d_storage = NULL;
  static size_t storage_bytes = 0;
  if (!d_storage){
    cub::DeviceReduce::Min(d_storage, storage_bytes, vec, d_min, N);
    d_storage = mfem::rmalloc<char>::operator new(storage_bytes);
  }
  cub::DeviceReduce::Min(d_storage, storage_bytes, vec, d_min, N);
  mfem::rmemcpy::rDtoH(h_min,d_min,sizeof(double));
  return *h_min;
}
#endif // __NVCC__

#ifdef __HIPCC__

extern "C" __global__ void vector_min0(const int N,
                                   const int Nblocks,
                                   const double* __restrict v0,
                                   double *storage){

  __shared__ volatile double s_min[CUDA_BLOCK_SIZE];

  int t = threadIdx.x;
  int b = blockIdx.x;
  int id = CUDA_BLOCK_SIZE * b + t;
  s_min[t] = v0[CUDA_BLOCK_SIZE * b];
  id += CUDA_BLOCK_SIZE*Nblocks;
  while (id<N) {
    const double v0n = v0[id];
    s_min[t] = (v0n < s_min[t]) ? v0n : s_min[t];
    id += CUDA_BLOCK_SIZE*Nblocks;
  }

  __syncthreads();

#if CUDA_BLOCK_SIZE>512
  if(t<512) s_min[t] = (s_min[t+512] < s_min[t]) ? s_min[t+512] : s_min[t];
  __syncthreads();
#endif

#if CUDA_BLOCK_SIZE>256
  if(t<256) s_min[t] = (s_min[t+256] < s_min[t]) ? s_min[t+256] : s_min[t];
  __syncthreads();
#endif

  if(t<128) s_min[t] = (s_min[t+128] < s_min[t]) ? s_min[t+128] : s_min[t];
  __syncthreads();

  if(t< 64) s_min[t] = (s_min[t+64] < s_min[t]) ? s_min[t+64] : s_min[t];
  __syncthreads();

  if(t< 32) s_min[t] = (s_min[t+32] < s_min[t]) ? s_min[t+32] : s_min[t];
  //    __syncthreads();

  if(t< 16) s_min[t] = (s_min[t+16] < s_min[t]) ? s_min[t+16] : s_min[t];
  //    __syncthreads();

  if(t<  8) s_min[t] = (s_min[t+8] < s_min[t]) ? s_min[t+8] : s_min[t];
  //    __syncthreads();

  if(t<  4) s_min[t] = (s_min[t+4] < s_min[t]) ? s_min[t+4] : s_min[t];
  //    __syncthreads();

  if(t<  2) s_min[t] = (s_min[t+2] < s_min[t]) ? s_min[t+2] : s_min[t];
  //    __syncthreads();

  if(t<  1) storage[b] = (s_min[1] < s_min[0]) ? s_min[1] : s_min[0];
}

static double hip_vector_min(const int N,
                             const double* __restrict vec) {
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
  hipLaunchKernelGGL((vector_min0),dim3(Nblocks),dim3(Nthreads),0,0,N,Nblocks,vec,d_storage);
  hipLaunchKernelGGL((vector_min0),dim3(1),      dim3(Nthreads),0,0,Nblocks,1,d_storage,d_min);
  mfem::rmemcpy::rDtoH(h_min,d_min,sizeof(double));
  return *h_min;
}
#endif

// *****************************************************************************
double vector_min(const int N,
                  const double* __restrict vec) {
  push(min,Cyan);
#ifdef __NVCC__
  if (mfem::rconfig::Get().Cuda()){
    const double result = cub_vector_min(N,vec);
    pop();
    return result;
  }
#endif
#ifdef __HIPCC__
  if (mfem::rconfig::Get().Hip()){
    const double result = hip_vector_min(N,vec);
    pop();
    return result;
  }
#endif
  ReduceDecl(Min,red,vec[0]);
  ReduceForall(i,N,red.min(vec[i]););
  pop();
  return red;
}

