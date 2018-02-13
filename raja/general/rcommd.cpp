// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#include "../raja.hpp"

namespace mfem {

  // ***************************************************************************
  // * RajaCommD
  // ***************************************************************************
  RajaCommD::RajaCommD(ParFiniteElementSpace &pfes):
    GroupCommunicator(pfes.GroupComm()),
    d_group_buf(NULL) {}

  
  // ***************************************************************************
  // * ~RajaCommD
  // ***************************************************************************
  RajaCommD::~RajaCommD(){ }

  // ***************************************************************************
  // * d_CopyGroupToBuffer
  // ***************************************************************************
  template <class T>
  T *RajaCommD::d_CopyGroupToBuffer(const T *d_ldata,
                                           T *d_buf, int group,
                                           int layout) const  {
    push(Yellow);
    dbg("\t\033[33m[d_CopyGroupToBuffer]");
    switch (layout) {
    case 1:
      {
        dbg("\t\t\033[33m[d_CopyGroupToBuffer] layout 1");
#ifndef __NVCC__ 
        pop();
        return std::copy(d_ldata + group_ldof.GetI()[group],
                         d_ldata + group_ldof.GetI()[group+1],
                         d_buf);
#else
        CUdeviceptr dest = (CUdeviceptr)d_buf;
        const CUdeviceptr src = (CUdeviceptr)(d_ldata + group_ldof.GetI()[group]);
        const size_t sz = group_ldof.GetI()[group+1]-group_ldof.GetI()[group];
        checkCudaErrors(cuMemcpyDtoD(dest,src,sz*sizeof(T)));
#endif
      }
    case 2:
      {
        dbg("\t\t\033[33m[d_CopyGroupToBuffer] layout 2");
        const int nltdofs = group_ltdof.RowSize(group);
        const int *ltdofs = group_ltdof.GetRow(group);
        for (int j = 0; j < nltdofs; j++)
        {
#ifndef __NVCC__
          d_buf[j] = d_ldata[ltdofs[j]];
#else
          CUdeviceptr dest = (CUdeviceptr)(d_buf+j);
          const CUdeviceptr src = (CUdeviceptr)(d_ldata+ltdofs[j]);
          checkCudaErrors(cuMemcpyDtoD(dest,src,sizeof(T)));
#endif
        }
        pop();
        return d_buf + nltdofs;
      }
    default:
      {
        dbg("\t\t\033[33m[d_CopyGroupToBuffer] default");
        const int nldofs = group_ldof.RowSize(group);
        const int *ldofs = group_ldof.GetRow(group);
        for (int j = 0; j < nldofs; j++)
        {
#ifndef __NVCC__
          d_buf[j] = d_ldata[ldofs[j]];
#else
          CUdeviceptr dest = (CUdeviceptr)(d_buf+j);
          const CUdeviceptr src = (CUdeviceptr)(d_ldata+ldofs[j]);
          checkCudaErrors(cuMemcpyDtoD(dest,src,sizeof(T)));
#endif
        }
        dbg("\t\t\033[33m[d_CopyGroupToBuffer] done");
        pop();
        return d_buf + nldofs;
      }
    }
  }

  // ***************************************************************************
  // ***************************************************************************
  template <class T>
  const T *RajaCommD::d_CopyGroupFromBuffer(const T *d_buf, T *d_ldata,
                                                   int group, int layout) const{
    push(Gold);
    dbg("\t\033[33m[d_CopyGroupFromBuffer]");
    const int nldofs = group_ldof.RowSize(group);
    assert(layout==0);
    dbg("\t\t\033[33m[d_CopyGroupFromBuffer] default");
    const int *ldofs = group_ldof.GetRow(group);
    for (int j = 0; j < nldofs; j++)
    {
#ifndef __NVCC__
      d_ldata[ldofs[j]] = d_buf[j];
#else
      CUdeviceptr dest = (CUdeviceptr)(d_ldata+ldofs[j]);
      const CUdeviceptr src = (CUdeviceptr)(d_buf+j);
      checkCudaErrors(cuMemcpyDtoD(dest,src,sizeof(T)));
#endif
    }
    dbg("\t\t\033[33m[d_CopyGroupFromBuffer] done");
    pop();
    return d_buf + nldofs;
  }

  // ***************************************************************************
  // * kAtomicAdd
  // ***************************************************************************
  template <class T>
  static __global__ void kAtomicAdd(T* adrs, T *value){
    //const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("\033[33m %d\033[m",tid);
    //atomicAdd(adrs, *value);
    *adrs += *value;
  }
  template __global__ void kAtomicAdd<int>(int*, int*);
  template __global__ void kAtomicAdd<double>(double*, double*);

  // ***************************************************************************
  // * ReduceGroupFromBuffer
  // ***************************************************************************
  template <class T>
  const T *RajaCommD::d_ReduceGroupFromBuffer(const T *d_buf, T *d_ldata,
                                                     int group, int layout,
                                                     void (*Op)(OpData<T>)) const  {
    push(PaleGoldenrod);
    dbg("\t\033[33m[d_ReduceGroupFromBuffer]");
    OpData<T> opd;
    opd.ldata = d_ldata;
    opd.nldofs = group_ldof.RowSize(group);
    opd.nb = 1;
    opd.buf = const_cast<T*>(d_buf);
    dbg("\t\t\033[33m[d_ReduceGroupFromBuffer] layout 2");
    opd.ldofs = const_cast<int*>(group_ltdof.GetRow(group));
#ifndef __NVCC__
    Op(opd);
#else
    assert(opd.nb == 1);
    
    // this is the operation to perform:
    // opd.ldata[opd.ldofs[i]] += opd.buf[i];
    // mfem/general/communication.cpp, line 1008
    //T adrs,value;
    //for (int i = 0; i < opd.nldofs; i++){
    //  const CUdeviceptr src = (CUdeviceptr)(opd.ldata+opd.ldofs[i]);
    //  checkCudaErrors(cuMemcpyDtoH(&adrs,src,sizeof(T)));
    //  const CUdeviceptr sval = (CUdeviceptr)(opd.buf+i);
    //  checkCudaErrors(cuMemcpyDtoH(&value,sval,sizeof(T)));
    //  // Do the +=
    //  adrs += value;//opd.buf[i];
    //  // Push back the answer          
    //  checkCudaErrors(cuMemcpyHtoD(src,&adrs,sizeof(T)));
    //}

    // this is the operation to perform: opd.ldata[opd.ldofs[i]] += opd.buf[i];
    // mfem/general/communication.cpp, line 1008
    for (int i = 0; i < opd.nldofs; i++)
      kAtomicAdd<<<1,1>>>(opd.ldata+opd.ldofs[i],opd.buf+i);
#endif // __NVCC__
    dbg("\t\t\033[33m[d_ReduceGroupFromBuffer] done");
    pop();
    return d_buf + opd.nldofs;
  }


  // ***************************************************************************
  // * d_BcastBegin
  // ***************************************************************************
  template <class T>
  void RajaCommD::d_BcastBegin(T *d_ldata, int layout) {
    MFEM_VERIFY(comm_lock == 0, "object is already in use");
    if (group_buf_size == 0) { return; }
    
    push(Moccasin);
    assert(layout==2);
    const int rnk = rconfig::Get().Rank();
    dbg("\033[33;1m[%d-d_BcastBegin]",rnk);
    int request_counter = 0;
    group_buf.SetSize(group_buf_size*sizeof(T));
    T *buf = (T *)group_buf.GetData();
#ifdef __NVCC__
    assert(d_group_buf);
    T *d_buf = (T*)d_group_buf;
#endif
    for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
    {
      const int num_send_groups = nbr_send_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
        T *buf_start = buf;
        T *d_buf_start = d_buf;
        const int *grp_list = nbr_send_groups.GetRow(nbr);
        for (int i = 0; i < num_send_groups; i++)
        {
          T *d_buf_ini = d_buf;
          assert(layout==2);
          d_buf = d_CopyGroupToBuffer(d_ldata, d_buf, grp_list[i], 2);
          buf += d_buf - d_buf_ini;
        }
        push(MPI_Isend,Orange);
#ifdef __NVCC__
        checkCudaErrors(cuMemcpyDtoH(buf_start,
                                     (CUdeviceptr)d_buf_start,
                                     (buf-buf_start)*sizeof(T)));
#endif
        MPI_Isend(buf_start,
                  buf - buf_start,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  40822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
        request_marker[request_counter] = -1; // mark as send request
        request_counter++;
      }

      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
        const int *grp_list = nbr_recv_groups.GetRow(nbr);
        int recv_size = 0;
        for (int i = 0; i < num_recv_groups; i++)
        {
          recv_size += group_ldof.RowSize(grp_list[i]);
        }
        push(MPI_Irecv,Orange);
        MPI_Irecv(buf,
                  recv_size,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  40822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
        request_marker[request_counter] = nbr;
        request_counter++;
        buf_offsets[nbr] = buf - (T*)group_buf.GetData();
        buf += recv_size;
        d_buf += recv_size;
      }
    }
    assert(buf - (T*)group_buf.GetData() == group_buf_size);
    comm_lock = 1; // 1 - locked for Bcast
    num_requests = request_counter;
    dbg("\033[33;1m[%d-d_BcastBegin] done",rnk);
    pop();
  }

  // ***************************************************************************
  // * d_BcastEnd
  // ***************************************************************************
  template <class T>
  void RajaCommD::d_BcastEnd(T *d_ldata, int layout) {
    if (comm_lock == 0) { return; }
    push(PeachPuff);
    const int rnk = rconfig::Get().Rank();
    dbg("\033[33;1m[%d-d_BcastEnd]",rnk);
    // The above also handles the case (group_buf_size == 0).
    assert(comm_lock == 1);
    // copy the received data from the buffer to d_ldata, as it arrives
    int idx;
    while (MPI_Waitany(num_requests, requests, &idx, MPI_STATUS_IGNORE),
           idx != MPI_UNDEFINED)
    {
      int nbr = request_marker[idx];
      if (nbr == -1) { continue; } // skip send requests

      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
        const int *grp_list = nbr_recv_groups.GetRow(nbr);
        int recv_size = 0;
        for (int i = 0; i < num_recv_groups; i++)
        {
          recv_size += group_ldof.RowSize(grp_list[i]);
        }
#ifndef __NVCC__
        const T *d_buf = (T*)group_buf.GetData() + buf_offsets[nbr];
#else
        const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
        const T *d_buf = (T*)d_group_buf + buf_offsets[nbr];
        checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)d_buf,buf,recv_size*sizeof(T)));
#endif
        for (int i = 0; i < num_recv_groups; i++)
        {
          d_buf = d_CopyGroupFromBuffer(d_buf, d_ldata, grp_list[i], layout);
        }
      }
    }
    comm_lock = 0; // 0 - no lock
    num_requests = 0;
    dbg("\033[33;1m[%d-d_BcastEnd] done",rnk);
    pop();
  }

  // ***************************************************************************
  // * d_ReduceBegin
  // ***************************************************************************
  template <class T>
  void RajaCommD::d_ReduceBegin(const T *d_ldata) {
    MFEM_VERIFY(comm_lock == 0, "object is already in use");
    if (group_buf_size == 0) { return; }
    push(PapayaWhip);
    const int rnk = rconfig::Get().Rank();
    dbg("\033[33;1m[%d-d_ReduceBegin]",rnk);

    int request_counter = 0;
    group_buf.SetSize(group_buf_size*sizeof(T));
    T *buf = (T *)group_buf.GetData();
#ifdef __NVCC__
    if (!d_group_buf){
      dbg("\n\033[31;1m[%d-d_ReduceBegin] d_buf cuMemAlloc\033[m",rnk);
      checkCudaErrors(cuMemAlloc((CUdeviceptr*)&d_group_buf,group_buf_size*sizeof(T)));
    }
    T *d_buf = (T*)d_group_buf;
#endif
    for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
    {
      const int num_send_groups = nbr_recv_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
        T *buf_start = buf;
        T *d_buf_start = d_buf;
        const int *grp_list = nbr_recv_groups.GetRow(nbr);
        for (int i = 0; i < num_send_groups; i++)
        {
          T *d_buf_ini = d_buf;
          d_buf = d_CopyGroupToBuffer(d_ldata, d_buf, grp_list[i], 0);
          buf += d_buf - d_buf_ini;
        }
        dbg("\033[33;1m[%d-d_ReduceBegin] MPI_Isend",rnk);
#ifdef __NVCC__
        push(DtoH,Red);
        checkCudaErrors(cuMemcpyDtoH(buf_start,
                                     (CUdeviceptr)d_buf_start,
                                     (buf-buf_start)*sizeof(T)));
        pop();
#endif
        push(MPI_Isend,Orange);
        MPI_Isend(buf_start,
                  buf - buf_start,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  43822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
        request_marker[request_counter] = -1; // mark as send request
        request_counter++;
      }

      // In Reduce operation: send_groups <--> recv_groups
      const int num_recv_groups = nbr_send_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
        const int *grp_list = nbr_send_groups.GetRow(nbr);
        int recv_size = 0;
        for (int i = 0; i < num_recv_groups; i++)
        {
          recv_size += group_ldof.RowSize(grp_list[i]);
        }
        dbg("\033[33;1m[%d-d_ReduceBegin] MPI_Irecv",rnk);
        push(MPI_Irecv,Orange);
        MPI_Irecv(buf,
                  recv_size,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  43822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
        request_marker[request_counter] = nbr;
        request_counter++;
        buf_offsets[nbr] = buf - (T*)group_buf.GetData();
        buf += recv_size;
        d_buf += recv_size;
      }
    }
    assert(buf - (T*)group_buf.GetData() == group_buf_size);
    comm_lock = 2;
    num_requests = request_counter;
    dbg("\033[33;1m[%d-d_ReduceBegin] done",rnk);
    pop();
  }

  // ***************************************************************************
  // * d_ReduceEnd
  // ***************************************************************************
  template <class T>
  void RajaCommD::d_ReduceEnd(T *d_ldata, int layout,
                                     void (*Op)(OpData<T>)){
    if (comm_lock == 0) { return; }
    push(LavenderBlush);
    const int rnk = rconfig::Get().Rank();
    dbg("\033[33;1m[%d-d_ReduceEnd]",rnk);
    // The above also handles the case (group_buf_size == 0).
    assert(comm_lock == 2);
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
    
    for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
    {
      // In Reduce operation: send_groups <--> recv_groups
      const int num_recv_groups = nbr_send_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
        const int *grp_list = nbr_send_groups.GetRow(nbr);
        int recv_size = 0;
        for (int i = 0; i < num_recv_groups; i++)
          recv_size += group_ldof.RowSize(grp_list[i]);
        const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
#ifdef __NVCC__
        const T *d_buf = (T*)d_group_buf + buf_offsets[nbr];
        checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)d_buf,buf,recv_size*sizeof(T)));
#endif       
        for (int i = 0; i < num_recv_groups; i++)
        {
          d_buf = d_ReduceGroupFromBuffer(d_buf, d_ldata, grp_list[i], layout, Op);
        }
      }
    }
    comm_lock = 0; // 0 - no lock
    num_requests = 0;
    dbg("\033[33;1m[%d-d_ReduceEnd] end",rnk);
    pop();
  }

  // ***************************************************************************
  // * instantiate RajaCommD::Bcast and Reduce for int and double
  // ***************************************************************************
  template void RajaCommD::d_BcastBegin<int>(int*, int);
  template void RajaCommD::d_BcastEnd<int>(int*, int);
  template void RajaCommD::d_ReduceBegin<int>(const int*);
  template void RajaCommD::d_ReduceEnd<int>(int*,int,void (*)(OpData<int>));

  template void RajaCommD::d_BcastBegin<double>(double*, int);
  template void RajaCommD::d_BcastEnd<double>(double*, int);
  template void RajaCommD::d_ReduceBegin<double>(const double *);
  template void RajaCommD::d_ReduceEnd<double>(double*,int,void (*)(OpData<double>));

} // namespace mfem