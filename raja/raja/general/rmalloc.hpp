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
/////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018,2019 Advanced Micro Devices, Inc.
/////////////////////////////////////////////////////////////////////////////////
#ifndef LAGHOS_RAJA_MALLOC
#define LAGHOS_RAJA_MALLOC

namespace mfem
{

// ***************************************************************************
template<class T> struct rmalloc: public rmemcpy
{

   // *************************************************************************
   inline void* operator new (size_t n, bool lock_page = false)
   {
#if defined(RAJA_ENABLE_CUDA)
      if (!rconfig::Get().Cuda()) { return ::new T[n]; }
      void *ptr;
      if (!rconfig::Get().Uvm())
      {
         if (lock_page) { cuMemHostAlloc(&ptr, n*sizeof(T), CU_MEMHOSTALLOC_PORTABLE); }
         else { cuMemAlloc((CUdeviceptr*)&ptr, n*sizeof(T)); }
      }
      else
      {
         cuMemAllocManaged((CUdeviceptr*)&ptr, n*sizeof(T),CU_MEM_ATTACH_GLOBAL);
      }
#elif defined(RAJA_ENABLE_HIP)
      if (!rconfig::Get().Hip()) { return ::new T[n]; }
      void *ptr;

      if (lock_page) { hipHostMalloc(&ptr, n*sizeof(T), hipHostMallocMapped); }
      else { hipMalloc((void**)&ptr, n*sizeof(T)); }
#endif
      return ptr;
   }

   // ***************************************************************************
   inline void operator delete (void *ptr)
   {
#if defined(RAJA_ENABLE_CUDA)
      if (!rconfig::Get().Cuda())
      {
         if (ptr)
         {
            ::delete[] static_cast<T*>(ptr);
         }
      }
      else
      {
         cuMemFree((CUdeviceptr)ptr); // or cuMemFreeHost if page_locked was used
      }
#elif defined(RAJA_ENABLE_HIP)
      if (!rconfig::Get().Hip())
      {
         if (ptr)
         {
            ::delete[] static_cast<T*>(ptr);
         }
      }
      else
      {
         hipFree(ptr); // or hipHostFree if page_locked was used
      }
#endif
      ptr = nullptr;
   }
};

} // mfem

#endif // LAGHOS_RAJA_MALLOC
