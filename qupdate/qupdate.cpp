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

#include "../laghos_solver.hpp"
#include "qupdate.hpp"

namespace mfem {

namespace hydrodynamics {
   
   // **************************************************************************
   //__attribute__((unused))
   static void getL2Values(const int dim,
                           const int nL2dof1D,
                           const int nqp1D,
                           const double* __restrict__ vecL2,
                           double* __restrict__ vecQ){     
      assert(dim == 2);      
      double LQ[nL2dof1D*nqp1D];         
      // LQ_j2_k1 = vecL2_j1_j2 LQs_j1_k1  -- contract in x direction.
      multAtB(nL2dof1D, nL2dof1D, tensors1D->LQshape1D.Width(),
              vecL2, tensors1D->LQshape1D.Data(), LQ);      
      // QQ_k1_k2 = LQ_j2_k1 LQs_j2_k2 -- contract in y direction.
      multAtB(nL2dof1D, nqp1D, tensors1D->LQshape1D.Width(),
              LQ, tensors1D->LQshape1D.Data(), vecQ);
   }

   // **************************************************************************
   static void getVectorGrad(const int dim,
                             const int nH1dof1D,
                             const int nqp1D,
                             const Array<int> &dof_map,
                             const DenseMatrix &vec,
                             DenseTensor &J) {
      assert(dim == 2);
      
      const int nH1dof = nH1dof1D * nH1dof1D;
      
      double X[nH1dof];
      double HQ[nH1dof1D * nqp1D];
      double QQ[nqp1D * nqp1D];
      
      for (int c = 0; c < 2; c++) {
         
         // Transfer from the mfem's H1 local numbering to the tensor structure
         for (int j = 0; j < nH1dof; j++) X[j] = vec(dof_map[j], c);         

         // HQ_i2_k1  = X_i1_i2 HQg_i1_k1  -- gradients in x direction.
         // QQ_k1_k2  = HQ_i2_k1 HQs_i2_k2 -- contract  in y direction.
         multAtB(nH1dof1D, nH1dof1D, tensors1D->HQgrad1D.Width(),
                 X, tensors1D->HQgrad1D.Data(), HQ);
         multAtB(nH1dof1D, nqp1D, tensors1D->HQshape1D.Width(),
                 HQ, tensors1D->HQshape1D.Data(), QQ);
         // Set the (c,0) component of the Jacobians at all quadrature points.
         for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int k2 = 0; k2 < nqp1D; k2++) {
               const int idx = k2 * nqp1D + k1;
               J(idx)(c, 0) = QQ[idx];
            }
         }

         // HQ_i2_k1  = X_i1_i2 HQs_i1_k1  -- contract  in x direction.
         // QQ_k1_k2  = HQ_i2_k1 HQg_i2_k2 -- gradients in y direction.
         multAtB(nH1dof1D, nH1dof1D, tensors1D->HQshape1D.Width(),
                 X, tensors1D->HQshape1D.Data(), HQ);
         multAtB(nH1dof1D, nqp1D, tensors1D->HQgrad1D.Width(),
                 HQ, tensors1D->HQgrad1D.Data(), QQ);
         // Set the (c,1) component of the Jacobians at all quadrature points.
         for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int k2 = 0; k2 < nqp1D; k2++) {
               const int idx = k2 * nqp1D + k1;
               J(idx)(c, 1) = QQ[idx];
            }
         }
      }
   } 
   
   // **************************************************************************
   void QUpdate(const int dim,
                const int nzones,
                const int l2dofs_cnt,
                const int h1dofs_cnt,
                const bool use_viscosity,
                const bool p_assembly,
                const double cfl,
                TimingData &timer,
                Coefficient *material_pcf,
                const IntegrationRule &integ_rule,
                ParFiniteElementSpace &H1FESpace,
                ParFiniteElementSpace &L2FESpace,
                const Vector &S,
                bool &quad_data_is_current,
                QuadratureData &quad_data) {
      push();
      assert(p_assembly);
      assert(material_pcf);
      
      ElementTransformation *T = H1FESpace.GetElementTransformation(0);
      const IntegrationPoint &ip = integ_rule.IntPoint(0);
      const double gamma = material_pcf->Eval(*T,ip);

      // ***********************************************************************
      if (quad_data_is_current) return;
      timer.sw_qdata.Start();

      const int nqp = integ_rule.GetNPoints();
      dbg("nqp=%d",nqp);

      ParGridFunction x, velocity, energy;
      Vector* sptr = (Vector*) &S;
      x.MakeRef(&H1FESpace, *sptr, 0);
//#warning x for(int i=0;i<x.Size();i+=1) x[i] = 1.123456789*drand48();
         //dbg("x (size=%d)",x.Size());//x.Print();
      velocity.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
      //dbg("velocity (size=%d)",velocity.Size());//velocity.Print();
      energy.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
      //dbg("energy (size=%d)",energy.Size());//energy.Print();
      
      Vector e_loc(l2dofs_cnt), vector_loc(h1dofs_cnt * dim);
      DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);
      DenseMatrix vector_loc_mtx(vector_loc.GetData(), h1dofs_cnt, dim);
      DenseTensor grad_v_ref(dim, dim, nqp);
      Array<int> L2dofs, H1dofs;

      const H1_QuadrilateralElement *fe =
         dynamic_cast<const H1_QuadrilateralElement *>(H1FESpace.GetFE(0));
      const Array<int> &h1_dof_map = fe->GetDofMap();
      
      const int nqp1D    = tensors1D->LQshape1D.Width();
      const int nL2dof1D = tensors1D->LQshape1D.Height();
      const int nH1dof1D = tensors1D->HQshape1D.Height();
      
      // Energy values at quadrature point *************************************
      const bool use_external_e = false;
      Vector e_vals(nqp);
      Vector e_quads(nzones * nqp);
      if (use_external_e)
         dof2quad(L2FESpace, integ_rule, energy.GetData(), e_quads.GetData());

      // Jacobian **************************************************************
      DenseTensor Jpr;
      Jpr.SetSize(dim, dim, nqp);
      qGeometry *geom = qGeometry::Get(H1FESpace,integ_rule);
/*
      //const size_t Jsz = dim*dim*nqp*nzones;
      //for(int k=0; k<Jsz; k++) dbg("%f",geom->J[k]);
      for(int z=0; z < nzones; z++){
         printf("\nzone %d",z);
         for(int q=0; q < nqp; q++) {
            printf("\n\tquad %d",q);
            const double J00 = geom->J[(z*nqp+q)*nzones+0];
            const double J10 = geom->J[(z*nqp+q)*nzones+1];
            const double J01 = geom->J[(z*nqp+q)*nzones+2];
            const double J11 = geom->J[(z*nqp+q)*nzones+3];
            printf("\n\t\tJ(%d,%d) %f %f",z,q,J00,J01);
            printf("\n\t\tJ(%d,%d) %f %f",z,q,J10,J11);
         }
      }
      assert(false);
      
      for(int z=0; z < nzones; z++) {
         printf("\nzone %d",z);
         ElementTransformation *T = H1FESpace.GetElementTransformation(z);
         H1FESpace.GetElementVDofs(z, H1dofs);
         x.GetSubVector(H1dofs, vector_loc);
         getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, Jpr);
         for(int q=0; q < nqp; q++) {
            printf("\n\tquad %d",q);
            const double J00 = Jpr(q)(0,0);
            const double J10 = Jpr(q)(1,0);
            const double J01 = Jpr(q)(0,1);
            const double J11 = Jpr(q)(1,1);
            printf("\n\t\tJpr(%d,%d) %f %f",z,q,J00,J01);
            printf("\n\t\tJpr(%d,%d) %f %f",z,q,J10,J11);
         }
      }
      assert(false);
      */
      
      // ***********************************************************************
      const bool use_external_J = true;
/*
      {
         for(int z=0; z < nzones; z++) {
            ElementTransformation *T = H1FESpace.GetElementTransformation(z);
            H1FESpace.GetElementVDofs(z, H1dofs);
            //dbg("\nH1dofs:\n");H1dofs.Print();
            x.GetSubVector(H1dofs, vector_loc);
            //dbg("\nvector_loc (z=%d):\n",z);vector_loc.Print();
            getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, Jpr);
            for(int q=0; q < nqp; q++) {
               const DenseMatrix &J = DenseMatrix(&geom->J[(z*nqp+q)*nzones],dim,dim);
               //dbg("Jpr (z=%d, q=%d):",z,q); Jpr(q).Print();
               //dbg("vs J:"); J.Print();
               if (fabs(Jpr(q)(0,0)-J(0,0))>1.e-14){
                  dbg("Jpr(q)(0,0)=%.21e %.21e=J(0,0):",Jpr(q)(0,0),J(0,0));
                  assert(false);
               }
               if (fabs(Jpr(q)(1,0)-J(1,0))>1.e-14){
                  dbg("Jpr(q)(1,0)=%.21e %.21e=J(1,0):",Jpr(q)(1,0),J(1,0));
                  assert(false);
               }
               if (fabs(Jpr(q)(0,1)-J(0,1))>1.e-14){
                  dbg("Jpr(q)(0,1)=%.21e %.21e=J(0,1):",Jpr(q)(0,1),J(0,1));
                  assert(false);
               }
               if (fabs(Jpr(q)(1,1)-J(1,1))>1.e-14){
                  dbg("Jpr(q)(1,1)=%.21e %.21e=J(1,1):",Jpr(q)(1,1),J(1,1));
                  assert(false);
               }
           }
         }
         assert(false);
      }
*/
      const double h1order = (double) H1FESpace.GetOrder(0);
      const double infinity = std::numeric_limits<double>::infinity();
      double min_detJ = infinity;
      
      for (int z = 0; z < nzones; z++) {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z);
         
         // Energy values at quadrature point **********************************
         if (!use_external_e){
            L2FESpace.GetElementDofs(z, L2dofs);
            //dbg("\nL2dofs:\n");L2dofs.Print();
            energy.GetSubVector(L2dofs, e_loc);
            //dbg("\ne_loc (z=%d):\n",z); e_loc.Print();
            getL2Values(dim, nL2dof1D, nqp1D, e_loc.GetData(), e_vals.GetData());
            //dbg("\ne_vals (z=%d):\n",z); e_vals.Print();
         }
 
         // Jacobians at quadrature points *************************************
         if (!use_external_J)
         {
            H1FESpace.GetElementVDofs(z, H1dofs);
            //dbg("\nH1dofs:\n");H1dofs.Print();
            x.GetSubVector(H1dofs, vector_loc);
            //dbg("\nvector_loc (z=%d):\n",z);vector_loc.Print();
            getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, Jpr);
         }

         // Velocity gradient at quadrature points *****************************
         if (use_viscosity) {
            H1FESpace.GetElementVDofs(z, H1dofs);
            velocity.GetSubVector(H1dofs, vector_loc);
            getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, grad_v_ref);
         }
         
         // ********************************************************************
         for (int q = 0; q < nqp; q++) {
            const int idx = z * nqp + q;
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            const double weight = ip.weight;
            const double inv_weight = 1. / weight;

            const DenseMatrix &J = !use_external_J? Jpr(q):
               DenseMatrix(&geom->J[(z*nqp+q)*nzones],dim,dim);

            const double detJ = J.Det();
            min_detJ = fmin(min_detJ, detJ);   
            calcInverse2D(J.Height(), J.Data(), Jinv.Data());        
            
            // *****************************************************************
            const double rho = inv_weight * quad_data.rho0DetJ0w(idx) / detJ;
            const double e   = !use_external_e?
               fmax(0.0, e_vals(q)):
               fmax(0.0, e_quads.GetData()[z*nqp1D*nqp1D+q]);
            const double p  = (gamma - 1.0) * rho * e;
            const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
            // *****************************************************************
            stress = 0.0;
            for (int d = 0; d < dim; d++)  stress(d,d) = -p;
            // *****************************************************************
            double visc_coeff = 0.0;
            if (use_viscosity) {
               // Compression-based length scale at the point. The first
               // eigenvector of the symmetric velocity gradient gives the
               // direction of maximal compression. This is used to define the
               // relative change of the initial length scale.
               mult(sgrad_v.Height(),sgrad_v.Width(),grad_v_ref(q).Width(),
                    grad_v_ref(q).Data(), Jinv.Data(), sgrad_v.Data());
               symmetrize(sgrad_v.Height(),sgrad_v.Data());
               double eig_val_data[3], eig_vec_data[9];
               if (dim==1) {
                  eig_val_data[0] = sgrad_v(0, 0);
                  eig_vec_data[0] = 1.;
               }
               else {
                  calcEigenvalues(sgrad_v.Height(),sgrad_v.Data(),
                                  eig_val_data, eig_vec_data);
               }
               Vector compr_dir(eig_vec_data, dim);
               // Computes the initial->physical transformation Jacobian.
               mult(Jpi.Height(),Jpi.Width(),J.Width(),
                 J.Data(), quad_data.Jac0inv(idx).Data(), Jpi.Data());
               Vector ph_dir(dim);
               //Jpi.Mult(compr_dir, ph_dir);
               multV(Jpi.Height(), Jpi.Width(), Jpi.Data(),
                     compr_dir.GetData(), ph_dir.GetData());
               // Change of the initial mesh size in the compression direction.
               const double h = quad_data.h0 * ph_dir.Norml2() / compr_dir.Norml2();
               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
               add(stress.Height(), stress.Width(),visc_coeff, sgrad_v.Data(), stress.Data());
            }
            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min = calcSingularvalue(J.Height(), dim-1, J.Data()) / h1order;
            const double inv_h_min = 1. / h_min;
            const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
            const double inv_dt = sound_speed * inv_h_min + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
            if (min_detJ < 0.0) {
               // This will force repetition of the step with smaller dt.
               quad_data.dt_est = 0.0;
            } else {
               quad_data.dt_est = fmin(quad_data.dt_est, cfl * (1.0 / inv_dt) );
            }
            // Quadrature data for partial assembly of the force operator.
            multABt(stress.Height(), stress.Width(), Jinv.Height(),
                    stress.Data(), Jinv.Data(), stressJiT.Data());
            stressJiT *= weight * detJ;
            for (int vd = 0 ; vd < dim; vd++) {
               for (int gd = 0; gd < dim; gd++) {
                  quad_data.stressJinvT(vd)(z*nqp + q, gd) =
                     stressJiT(vd, gd);
               }
            }
         }
      }
      //assert(false);
      quad_data_is_current = true;
      timer.sw_qdata.Stop();
      timer.quad_tstep += nzones;
   }

} // namespace hydrodynamics

} // namespace mfem
