// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "../laghos_assembly.hpp"

#include "kMassPAOperator.hpp"
#include "backends/kernels/kernels.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
kMassPAOperator::kMassPAOperator(QuadratureData *qd_,
                                 ParFiniteElementSpace &fes_,
                                 const IntegrationRule &ir_) :
   AbcMassPAOperator(*fes_.GetTrueVLayout()),
   dim(fes_.GetMesh()->Dimension()),
   nzones(fes_.GetMesh()->GetNE()),
   quad_data(qd_),
   fes(fes_),
   ir(ir_),
   ess_tdofs_count(0),
   ess_tdofs(0),
   bilinearForm(new kernels::kBilinearForm(&fes.Get_PFESpace()->As<kernels::kFiniteElementSpace>())) { }

// *****************************************************************************
void kMassPAOperator::Setup()
{
   dbg("engine");
   const mfem::Engine &engine = fes.GetMesh()->GetEngine();
   dbg("massInteg");
   kernels::KernelsMassIntegrator &massInteg = *(new kernels::KernelsMassIntegrator(engine));
   dbg("SetIntegrationRule");
   massInteg.SetIntegrationRule(ir);
   dbg("SetOperator");
   massInteg.SetOperator(quad_data->rho0DetJ0w);
   dbg("AddDomainIntegrator");
   bilinearForm->AddDomainIntegrator(&massInteg);
   dbg("Assemble");
   bilinearForm->Assemble();
   dbg("FormOperator");
   bilinearForm->FormOperator(mfem::Array<int>(), massOperator);
   dbg("done");
   pop();
}

// *************************************************************************
void kMassPAOperator::SetEssentialTrueDofs(mfem::Array<int> &dofs)
{
   push(Wheat);
   dbg("ess_tdofs_count=%d, ess_tdofs.Size()=%d & dofs.Size()=%d",ess_tdofs_count, ess_tdofs.Size(), dofs.Size());
   ess_tdofs_count = dofs.Size();
  
   if (ess_tdofs.Size()==0){
      dbg("ess_tdofs.Size()==0");
#ifdef MFEM_USE_MPI
      dbg("MPI_Allreduce");
      int global_ess_tdofs_count;
      const MPI_Comm comm = fes.GetParMesh()->GetComm();
      MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                    1, MPI_INT, MPI_SUM, comm);
      assert(global_ess_tdofs_count>0);
      
      dbg("ess_tdofs.Resize");      
      //ess_tdofs.Resize(global_ess_tdofs_count);
      const mfem::Engine &engine = fes.GetMesh()->GetEngine();
      ess_tdofs.Resize(engine.MakeLayout(global_ess_tdofs_count));
      ess_tdofs.Fill(0);
      dbg("ess_tdofs.Pull(false)");
      ess_tdofs.Pull(false);
#else
      assert(ess_tdofs_count>0);
      ess_tdofs.Resize(ess_tdofs_count);
#endif
   } else{
      dbg("ess_tdofs_count==%d",ess_tdofs_count);
      dbg("ess_tdofs.Size()==%d",ess_tdofs.Size());      
      assert(ess_tdofs_count<=ess_tdofs.Size());
   }

   assert(ess_tdofs>0);
  
   if (ess_tdofs_count == 0) { 
      dbg("(0) done");
      pop();
      return;
   }
  
   {
      assert(ess_tdofs_count>0);
      assert(dofs.GetData());
      //memcpy((void*)ess_tdofs.GetData(), dofs.GetData(), ess_tdofs_count*sizeof(int));
      int *d_ess_tdofs = (int*) mfem::kernels::kmalloc<int>::operator new(ess_tdofs_count);
      kernels::kmemcpy::rHtoD((void*)d_ess_tdofs,
                              dofs.GetData(),
                              ess_tdofs_count*sizeof(int));
      ess_tdofs.MakeRef(d_ess_tdofs,ess_tdofs_count);
      pop();
   }
   dbg("done");
   pop();
}

// *****************************************************************************
void kMassPAOperator::EliminateRHS(mfem::Vector &b)
{
   push(Wheat);
   if (ess_tdofs_count > 0){
      kernels::Vector kb = b.Get_PVector()->As<kernels::Vector>();
      kb.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
   pop();
}

// *************************************************************************
void kMassPAOperator::Mult(const mfem::Vector &x,
                                 mfem::Vector &y) const
{
   push();
   dbg("mx");

   mfem::Vector mx(fes.GetTrueVLayout());
   mx.PushData(x.GetData());
   
   kernels::Vector &kx = mx.Get_PVector()->As<kernels::Vector>();
   kernels::Vector &ky = y.Get_PVector()->As<kernels::Vector>();

   if (ess_tdofs_count)
   {
      dbg("kx.SetSubVector");
      kx.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
   
   dbg("massOperator->Mult(mx, y);");
   //massOperator->Mult(mx, y);
   massOperator->Mult(x, y);
   //while(true);
   //assert(false);

   if (ess_tdofs_count)
   {
      dbg("ky.SetSubVector");
      ky.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
   dbg("done");
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
