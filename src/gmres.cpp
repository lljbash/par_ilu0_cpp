#include "gmres.hpp"
#include <cmath>
#include <cblas.h>
#include "csr_matvec.h"
#include "scope_guard.hpp"

namespace lljbash {

std::pair<bool, int>
PreconditionedGmres::Solve(const CsrMatrix *mat, const double *rhs, double* sol) {
    auto tol = param_.tolerance;
    auto im = param_.krylov_subspace_dimension;
    auto maxits = param_.max_iterations;
    auto n = mat->size;
    constexpr int one = 1;
    constexpr double epsmac = 1e-16;
    int ptih = 0;
    double eps1 = 0;
    auto im1 = im + 1;
    intermediate.vv.resize(im1 * n);
    intermediate.z.resize(im * n);
    intermediate.hh.resize(im1 * (im + 3));
    auto vv = intermediate.vv.data();
    auto z = intermediate.z.data();
    auto hh = intermediate.hh.data();
    auto c = hh + im1 * im;
    auto s = c + im1;
    auto rs = s + im1;
/*-------------------- outer loop starts here */
    int its = 0;
/*-------------------- outer loop */
    while (its < maxits) {
/*-------------------- compute initial residual vector */
        CsrMatVec(mat, sol, vv);
        for (int j = 0; j < n; ++j) {
            vv[j] = rhs[j] - vv[j]; // vv[0] = initial residual
        }
        auto beta = cblas_dnrm2(n, vv, one);
        if (beta == 0.0) {
            break;
        }
        auto t = 1.0 / beta;
/*-------------------- normalize: vv = vv / beta */
        cblas_dscal(n, t, vv, one);
        if (its == 0) {
            eps1 = tol * beta;
        }
/*-------------------- initialize 1-st term of rhs hessenberg mtx */
        rs[0] = beta;
/*-------------------- Krylov loop */
        int i = -1;
        while ((i < im - 1) && (beta > eps1) && (its++ < maxits)) {
            ++i;
            int i1 = i + 1;
            int pti = i * n;
            int pti1 = i1 * n;
/*------------------------------------------------------------
|  (Right) Preconditioning Operation   z_{j} = M^{-1} v_{j}
+-----------------------------------------------------------*/
            if (precon_set_) {
                precon_(vv + pti, z + pti);
            }
            else {
                std::copy_n(vv + pti, n, z + pti);
            }
/*-------------------- matvec operation w = A z_{j} = A M^{-1} v_{j} */
            CsrMatVec(mat, &z[pti], &vv[pti1]);
/*-------------------- modified gram - schmidt...
|     h_{i,j} = (w,v_{i});
|     w  = w - h_{i,j} v_{i}
+-----------------------------------------------------------*/
            ptih = i * im1;
            for (int j = 0; j <= i; ++j) {
                t = cblas_ddot(n, &vv[j*n], one, &vv[pti1], one);
                hh[ptih+j] = t;
                auto negt = -t;
                cblas_daxpy(n, negt, &vv[j*n], one, &vv[pti1], one);
            }
/*-------------------- h_{j+1} = ||w||_{2} */
            t = cblas_dnrm2(n, &vv[pti1], one);
            hh[ptih+i1] = t;
            if (t == 0.0) {
                //return {false, its};
                break;
            }
            t = 1.0 / t;
/*-------------------- v_{j+1} = w / h_{j+1,j} */
            cblas_dscal(n, t, &vv[pti1], one);
/*-------- done with modified gram schmidt/arnoldi step
| now update factorization of hh
| perform previous transformations on i-th column of h
+------------------------------------------------------*/
            for (int k = 1; k <= i; ++k) {
                auto k1 = k - 1;
                t = hh[ptih+k1];
                hh[ptih+k1] = c[k1] * t + s[k1] * hh[ptih+k];
                hh[ptih+k] = -s[k1] * t + c[k1] * hh[ptih+k];
            }
            auto gam = std::sqrt(std::pow(hh[ptih+i], 2) + std::pow(hh[ptih+i1], 2));
/*-------------------- check if gamma is zero */
            if (gam == 0.0) {
                gam = epsmac;
            }
/*-------------------- get next plane rotation */
            c[i] = hh[ptih+i] / gam;
            s[i] = hh[ptih+i1] / gam;
            rs[i1] = -s[i] * rs[i];
            rs[i] = c[i] * rs[i];
/*-------------------- get residual norm + test convergence */
            hh[ptih+i] = c[i] * hh[ptih+i] + s[i] * hh[ptih+i1];
            beta = std::fabs(rs[i1]);
/*-------------------- end [inner] while loop [Arnoldi] */
        }
/*---------- now compute solution. 1st, solve upper trianglular system */
        rs[i] = rs[i] / hh[ptih+i];
        for (int ii = i - 1; ii >= 0; --ii) {
            t = rs[ii];
            for (int j = ii + 1; j <= i; ++j) {
                t -= hh[j*im1+ii] * rs[j];
            }
            rs[ii] = t / hh[ii*im1+ii];
        }
/*---------- linear combination of z_j's to get sol. */
        for (int j = 0; j <= i; ++j) {
            cblas_daxpy(n, rs[j], &z[j*n], one, sol, one);
        }
/*-------------------- restart outer loop if needed */
        if (beta < eps1) {
            break;
        }
        else if (its > maxits) {
            return {false, its};
        }
/*---------- end main [outer] while loop */
    }
/*-------------------- prepare to return */
    return {true, its};
}

} // namespace lljbash
