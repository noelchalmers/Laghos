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
#ifndef LAGHOS_HIP_VECTOR
#define LAGHOS_HIP_VECTOR

namespace mfem
{

class HipVector : public rmalloc<double>
{
private:
   size_t size = 0;
   double* data = NULL;
   bool own = true;
public:
   HipVector(): size(0),data(NULL),own(true) {}
   HipVector(const HipVector&);
   HipVector(const HipVector*);
   HipVector(const size_t);
   HipVector(const size_t,double);
   HipVector(const Vector& v);
   HipVector(HipArray<double>& v);
   operator Vector();
   operator Vector() const;
   double* alloc(const size_t);
   inline double* ptr() const { return data;}
   inline double* GetData() const { return data;}
   inline operator double* () { return data; }
   inline operator const double* () const { return data; }
   void Print(std::ostream& = std::cout, int = 8) const;
   void SetSize(const size_t,const void* =NULL);
   inline size_t Size() const { return size; }
   inline size_t bytes() const { return size*sizeof(double); }
   double operator* (const HipVector& v) const;
   HipVector& operator = (const HipVector& v);
   HipVector& operator = (const Vector& v);
   HipVector& operator = (double value);
   HipVector& operator -= (const HipVector& v);
   HipVector& operator += (const HipVector& v);
   HipVector& operator += (const Vector& v);
   HipVector& operator *=(const double d);
   HipVector& Add(const double a, const HipVector& Va);
   void Neg();
   HipVector* GetRange(const size_t, const size_t) const;
   void SetSubVector(const HipArray<int> &, const double, const int);
   double Min() const;
   ~HipVector();
};

// ***************************************************************************
void add(const HipVector&,const double,const HipVector&,HipVector&);
void add(const HipVector&,const HipVector&,HipVector&);
void add(const double,const HipVector&,const double,const HipVector&,
         HipVector&);
void subtract(const HipVector&,const HipVector&,HipVector&);

}

#endif // LAGHOS_HIP_VECTOR
