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

#include "laghos_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

   // **************************************************************************
   static void multABt(const size_t ah,
                       const size_t aw,
                       const size_t bh,
                       const double* __restrict__ A,
                       const double* __restrict__ B,
                       double* __restrict__ C){
      const size_t ah_x_bh = ah*bh;
      for(size_t i=0; i<ah_x_bh; i+=1)
         C[i] = 0.0;  
      for(size_t k=0; k<aw; k+=1) {
         double *c = C;
         for(size_t j=0; j<bh; j+=1){
            const double bjk = B[j];
            for(size_t i=0; i<ah; i+=1)
               c[i] += A[i] * bjk;            
            c += ah;
         }
         A += ah;
         B += bh;
      }
   }

   // **************************************************************************
   static void multAtB(const size_t ah,
                       const size_t aw,
                       const size_t bw,
                       const double* __restrict__ A,
                       const double* __restrict__ B,
                       double* __restrict__ C) {
      for(size_t j = 0; j < bw; j+=1) {
         const double *a = A;
         for(size_t i = 0; i < aw; i+=1) {
            double d = 0.0;
            for(size_t k = 0; k < ah; k+=1) 
               d += a[k] * B[k];
            *(C++) = d;
            a += ah;
         }
         B += ah;
      }
   }
   
   // **************************************************************************
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
               J(idx)(c, 0) = QQ[k1+nqp1D*k2];
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
               J(idx)(c, 1) = QQ[k1+nqp1D*k2];
            }
         }
      }
   }

   // **************************************************************************
   static inline double det2D(const double *d){
      return d[0] * d[3] - d[1] * d[2];
   }
   
   // **************************************************************************
   /*static inline double det3D(const double *d){
      return
         d[0] * (d[4] * d[8] - d[5] * d[7]) +
         d[3] * (d[2] * d[7] - d[1] * d[8]) +
         d[6] * (d[1] * d[5] - d[2] * d[4]);
         }*/

   // **************************************************************************
   static void calcInverse2D(const size_t n, const double *a, double *i){
      const double d = det2D(a);
      const double t = 1.0 / d;
      i[0*n+0] =  a[1*n+1] * t ;
      i[0*n+1] = -a[0*n+1] * t ;
      i[1*n+0] = -a[1*n+0] * t ;
      i[1*n+1] =  a[0*n+0] * t ;
   }

   // **************************************************************************
   static void mult(const size_t ah,
                    const size_t aw,
                    const size_t bw,
                    const double* __restrict__ B,
                    const double* __restrict__ C,
                    double* __restrict__ A){
      const size_t ah_x_aw = ah*aw;
      for (int i = 0; i < ah_x_aw; i++) A[i] = 0.0;
      for (int j = 0; j < aw; j++) {
         for (int k = 0; k < bw; k++) {
            for (int i = 0; i < ah; i++) {
               A[i+j*ah] += B[i+k*ah] * C[k+j*bw];
            }
         }
      }
   }

   // **************************************************************************
   static void symmetrize(const size_t n, double* __restrict__ d){
      for (size_t i = 0; i<n; i++){
         for (size_t j = 0; j<i; j++) {
            const double a = 0.5 * (d[i*n+j] + d[j*n+i]);
            d[j*n+i] = d[i*n+j] = a;
         }
      }
   }
   
   // **************************************************************************
   static inline double cpysign(const double x, const double y) {
      if ((x < 0 && y > 0) || (x > 0 && y < 0))
         return -x;
      return x;
   }

   // **************************************************************************
   static inline void eigensystem2S(const double &d12, double &d1, double &d2,
                                    double &c, double &s) {
      static const double sqrt_1_eps = sqrt(1./numeric_limits<double>::epsilon());
      if (d12 == 0.) {
         c = 1.;
         s = 0.;
      } else {
         // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
         double t, zeta = (d2 - d1)/(2*d12);
         if (fabs(zeta) < sqrt_1_eps) {
            t = cpysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
         } else {
            t = cpysign(0.5/fabs(zeta), zeta);
         }
         c = sqrt(1./(1. + t*t));
         s = c*t;
         t *= d12;
         d1 -= t;
         d2 += t;
      }
   }
   
   // **************************************************************************
   static void calcEigenvalues(const size_t n, const double *d,
                               double *lambda,
                               double *vec) {
      assert(n == 2);   
      double d0 = d[0];
      double d2 = d[2]; // use the upper triangular entry
      double d3 = d[3];
      double c, s;
      eigensystem2S(d2, d0, d3, c, s);
      if (d0 <= d3) {
         lambda[0] = d0;
         lambda[1] = d3;
         vec[0] =  c;
         vec[1] = -s;
         vec[2] =  s;
         vec[3] =  c;
      } else {
         lambda[0] = d3;
         lambda[1] = d0;
         vec[0] =  s;
         vec[1] =  c;
         vec[2] =  c;
         vec[3] = -s;
      }
   }

   // **************************************************************************
   static inline void getScalingFactor(const double &d_max, double &mult){
      int d_exp;
      if (d_max > 0.)
      {
         mult = frexp(d_max, &d_exp);
         if (d_exp == numeric_limits<double>::max_exponent)
         {
            mult *= numeric_limits<double>::radix;
         }
         mult = d_max/mult;
      }
      else
      {
         mult = 1.;
      }
      // mult = 2^d_exp is such that d_max/mult is in [0.5,1)
      // or in other words d_max is in the interval [0.5,1)*mult
   }

      // **************************************************************************
   static double calcSingularvalue(const int n, const int i, const double *d) {
      assert (n == 2);
      
      double d0, d1, d2, d3;
      d0 = d[0];
      d1 = d[1];
      d2 = d[2];
      d3 = d[3];
      double mult;
      
      {
         double d_max = fabs(d0);
         if (d_max < fabs(d1)) { d_max = fabs(d1); }
         if (d_max < fabs(d2)) { d_max = fabs(d2); }
         if (d_max < fabs(d3)) { d_max = fabs(d3); }

         getScalingFactor(d_max, mult);
      }
      
      d0 /= mult;
      d1 /= mult;
      d2 /= mult;
      d3 /= mult;
      
      double t = 0.5*((d0+d2)*(d0-d2)+(d1-d3)*(d1+d3));
      double s = d0*d2 + d1*d3;
      s = sqrt(0.5*(d0*d0 + d1*d1 + d2*d2 + d3*d3) + sqrt(t*t + s*s));
      
      if (s == 0.0)
      {
         return 0.0;
      }
      t = fabs(d0*d3 - d1*d2) / s;
      if (t > s)
      {
         if (i == 0)
         {
            return t*mult;
         }
         return s*mult;
      }
      if (i == 0)
      {
         return s*mult;
      }
      return t*mult;
   }
   
   // **************************************************************************
   static void multV(const size_t height,
                     const size_t width,
                     double *data,
                     const double *x, double *y) {
      if (width == 0) {
         for (int row = 0; row < height; row++) 
            y[row] = 0.0;         
         return;
      }
      double *d_col = data;
      double x_col = x[0];
      for (int row = 0; row < height; row++) {
         y[row] = x_col*d_col[row];
      }
      d_col += height;
      for (int col = 1; col < width; col++) {
         x_col = x[col];
         for (int row = 0; row < height; row++) {
            y[row] += x_col*d_col[row];
         }
         d_col += height;
      }
   }
   
   // **************************************************************************
   static void add(const size_t height, const size_t width,
                   const double c, const double *A,
                   double *D){
      for (int j = 0; j < width; j++)
         for (int i = 0; i < height; i++) {
            D[i*width+j] += c * A[i*width+j];
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
      assert(p_assembly);
      assert(material_pcf);
      
      ElementTransformation *T = H1FESpace.GetElementTransformation(0);
      const IntegrationPoint &ip = integ_rule.IntPoint(0);
      const double gamma = material_pcf->Eval(*T,ip);

      // ***********************************************************************
      if (quad_data_is_current) return;
      timer.sw_qdata.Start();

      const int nqp = integ_rule.GetNPoints();

      ParGridFunction x, velocity, energy;
      Vector* sptr = (Vector*) &S;
      x.MakeRef(&H1FESpace, *sptr, 0);
      velocity.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
      energy.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
      
      Vector e_loc(l2dofs_cnt), vector_loc(h1dofs_cnt * dim);
      DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);
      DenseMatrix vector_loc_mtx(vector_loc.GetData(), h1dofs_cnt, dim);
      DenseTensor grad_v_ref(dim, dim, nqp);
      Array<int> L2dofs, H1dofs;
      DenseTensor Jpr;
      Jpr.SetSize(dim, dim, nqp);

      const H1_QuadrilateralElement *fe =
         dynamic_cast<const H1_QuadrilateralElement *>(H1FESpace.GetFE(0));
      const Array<int> &h1_dof_map = fe->GetDofMap();
      
      const int nqp1D    = tensors1D->LQshape1D.Width();
      const int nL2dof1D = tensors1D->LQshape1D.Height();
      const int nH1dof1D = tensors1D->HQshape1D.Height();
      
      Vector e_vals(nqp1D * nqp1D);
      const double h1order = (double) H1FESpace.GetOrder(0);
      const double infinity = numeric_limits<double>::infinity();
      double min_detJ = infinity;
      
      for (int z = 0; z < nzones; z++) {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z);
         
         // Energy values at quadrature point **********************************
         L2FESpace.GetElementDofs(z, L2dofs);
         energy.GetSubVector(L2dofs, e_loc);
         getL2Values(dim, nL2dof1D, nqp1D, e_loc.GetData(), e_vals.GetData());
         
         // Jacobians at quadrature points *************************************
         H1FESpace.GetElementVDofs(z, H1dofs);
         x.GetSubVector(H1dofs, vector_loc);
         getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, Jpr);
         
         // Velocity gradient at quadrature points *****************************
         if (use_viscosity) {
            H1FESpace.GetElementVDofs(z, H1dofs);
            velocity.GetSubVector(H1dofs, vector_loc);
            getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, grad_v_ref);
         }
         
         // ********************************************************************
         for (int q = 0; q < nqp; q++) {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            const double weight = ip.weight;
            const double inv_weight = 1. / weight;

            const DenseMatrix &J = Jpr(q);
            const double detJ = J.Det();
            min_detJ = fmin(min_detJ, detJ);
            
            const int idx = z * nqp + q;
            
            const double rho = inv_weight * quad_data.rho0DetJ0w(idx) / detJ;
            const double e   = fmax(0.0, e_vals(q));
            // *****************************************************************
            const double p  = (gamma - 1.0) * rho * e;
            const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
            // *****************************************************************
            calcInverse2D(J.Height(), J.Data(), Jinv.Data());
            stress = 0.0;
            for (int d = 0; d < dim; d++)  stress(d, d) = -p;
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
      quad_data_is_current = true;
      timer.sw_qdata.Stop();
      timer.quad_tstep += nzones;
   }

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
