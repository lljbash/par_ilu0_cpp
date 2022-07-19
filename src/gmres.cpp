#include "gmres.hpp"
#include <cmath>
#include <mkl.h>
#include "scope_guard.hpp"
#include "tictoc.hpp"

namespace lljbash {

static_assert(std::is_same_v<MKL_INT, int>);

PreconditionedGmres::~PreconditionedGmres() {
#if 0
    mkl_free(mkl_.tmp);
    mkl_free(mkl_.b);
    mkl_free(mkl_.residual);
#else
    mkl_free(intermediate_.vv);
    mkl_free(intermediate_.z);
    mkl_free(intermediate_.hh);
#endif
    MKL_Free_Buffers();
}

#if 0
void PreconditionedGmres::SetupMkl(int n) {
    auto d = param_.krylov_subspace_dimension;
    auto tmp_size = (2 * d + 1) * n + d * (d + 9) / 2 + 1;
    mkl_.tmp = static_cast<double*>(mkl_malloc(sizeof(double[tmp_size]), 64));
    mkl_.b = static_cast<double*>(mkl_malloc(sizeof(double[n]), 64));
    mkl_.residual = static_cast<double*>(mkl_malloc(sizeof(double[n]), 64));
    mkl_.setup = true;
}

std::pair<bool, int>
PreconditionedGmres::Solve(const CsrMatrix *mat, double *rhs, double* sol) {
    Tic();
    MKL_INT n = mat->size;
    if (!mkl_.setup) {
        SetupMkl(n);
    }
#define LLJBASH_EXPORT_MKL_VAR(var) auto& var = mkl_.var
    LLJBASH_EXPORT_MKL_VAR(ipar);
    LLJBASH_EXPORT_MKL_VAR(dpar);
    LLJBASH_EXPORT_MKL_VAR(tmp);
    LLJBASH_EXPORT_MKL_VAR(b);
    LLJBASH_EXPORT_MKL_VAR(residual);
#undef LLJBASH_EXPORT_MKL_VAR
    const MKL_INT one = 1;
    const double d_neg_one = -1.0;
    double nrm;
    MKL_INT RCI_request;
    MKL_INT iter = -1;
    std::pair<bool, int> ret;
    uint64_t tic;

    MKL_INT* ia = mat->row_ptr;
    MKL_INT* ja = mat->col_idx;
    double* A = mat->value;
    matrix_descr descrA;
    sparse_matrix_t csrA = NULL;
    sparse_operation_t transA = SPARSE_OPERATION_NON_TRANSPOSE;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.mode = SPARSE_FILL_MODE_FULL;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, n, n, ia, ia+1, ja, A);
    mkl_sparse_set_mv_hint(csrA, transA, descrA, 50);
    mkl_sparse_optimize(csrA);
    dcopy(&n, rhs, &one, b, &one);

    dfgmres_init(&n, sol, rhs, &RCI_request, ipar, dpar, tmp);
    stat_.total_init_time += Toc();
    if (RCI_request != 0) {
        goto FAILED;
    }
    Tic();
    ipar[4] = param_.max_iterations;
    ipar[6] = 0;
    ipar[10] = 1;
    ipar[11] = 1;
    ipar[14] = param_.krylov_subspace_dimension;
    dpar[0] = param_.tolerance;
    dfgmres_check(&n, sol, rhs, &RCI_request, ipar, dpar, tmp);
    stat_.total_init_time += Toc();
    if (RCI_request != 0 && RCI_request != -1001) {
        goto FAILED;
    }

ONE:
    tic = Rdtsc();
    dfgmres(&n, sol, rhs, &RCI_request, ipar, dpar, tmp);
    stat_.total_fgmr_time += Toc(tic);
    switch(RCI_request) {
        case 0:
            goto COMPLETE;
        case 1:
            Tic();
            mkl_sparse_d_mv(transA, 1.0, csrA, descrA, &tmp[ipar[21]-1], 0.0, &tmp[ipar[22]-1]);
            stat_.total_mv_time += Toc();
            goto ONE;
        case 2:
            ipar[12] = 1;
            dfgmres_get(&n, sol, b, &RCI_request, ipar, dpar, tmp, &iter);
            Tic();
            mkl_sparse_d_mv( transA, 1.0, csrA, descrA, b, 0.0, residual);
            stat_.total_mv_time += Toc();
            daxpy (&n, &d_neg_one, rhs, &one, residual, &one);
            nrm = dnrm2(&n, residual, &one);
            if (nrm < param_.tolerance) {
                goto COMPLETE;
            }
            else {
                goto ONE;
            }
        case 3:
            Tic();
            precon_(&tmp[ipar[21]-1], &tmp[ipar[22]-1]);
            stat_.total_precon_time += Toc();
            goto ONE;
        default:
            goto FAILED;
    }

COMPLETE:
    ipar[12] = 0;
    dfgmres_get(&n, sol, rhs, &RCI_request, ipar, dpar, tmp, &iter);
    ret = {true, iter};
    goto END;

FAILED:
    ret = {false, iter};
    goto END;

END:
    mkl_sparse_destroy(csrA);
    return ret;
}
#else
std::pair<bool, int>
PreconditionedGmres::Solve(const CsrMatrix *mat, double *rhs, double* sol) {
    auto tol = param_.tolerance;
    auto im = param_.krylov_subspace_dimension;
    auto maxits = param_.max_iterations;
    auto n = mat->size;
    constexpr int one = 1;
    constexpr double epsmac = 1e-16;
    int ptih = 0;
    double eps1 = tol;
    auto im1 = im + 1;
    if (!intermediate_.setup) {
        intermediate_.vv = static_cast<double*>(mkl_malloc(sizeof(double[im1*n]), 64));
        intermediate_.z = static_cast<double*>(mkl_malloc(sizeof(double[im*n]), 64));
        intermediate_.hh = static_cast<double*>(mkl_malloc(sizeof(double[im1*(im+3)]), 64));
        intermediate_.setup = true;
    }
    auto vv = intermediate_.vv;
    auto z = intermediate_.z;
    auto hh = intermediate_.hh;
    auto c = hh + im1 * im;
    auto s = c + im1;
    auto rs = s + im1;

/*-------------------- setup mkl spmv */
    MKL_INT* ia = mat->row_ptr;
    MKL_INT* ja = mat->col_idx;
    double* A = mat->value;
    matrix_descr descrA;
    sparse_matrix_t csrA = NULL;
    sparse_operation_t transA = SPARSE_OPERATION_NON_TRANSPOSE;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.mode = SPARSE_FILL_MODE_FULL;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, n, n, ia, ia+1, ja, A);
    //mkl_sparse_set_mv_hint(csrA, transA, descrA, 50);
    //mkl_sparse_optimize(csrA);

/*-------------------- outer loop starts here */
    int its = 0;
/*-------------------- outer loop */
    while (its < maxits) {
/*-------------------- compute initial residual vector */
        cblas_dcopy(n, rhs, one, vv, one);
        Tic();
        mkl_sparse_d_mv(transA, -1.0, csrA, descrA, sol, 1.0, vv);
        stat_.total_mv_time += Toc();
        auto beta = cblas_dnrm2(n, vv, one);
        if (beta <= tol) {
            break;
        }
        auto t = 1.0 / beta;
/*-------------------- normalize: vv = vv / beta */
        cblas_dscal(n, t, vv, one);
        //if (its == 0) {
            //eps1 = tol * beta;
        //}
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
            Tic();
            precon_(vv + pti, z + pti);
            stat_.total_precon_time += Toc();
/*-------------------- matvec operation w = A z_{j} = A M^{-1} v_{j} */
            Tic();
            mkl_sparse_d_mv(transA, 1.0, csrA, descrA, &z[pti], 0.0, &vv[pti1]);
            stat_.total_mv_time += Toc();
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
            if (t <= epsmac) {
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
            //printf("%f\n", beta);
            if (std::isnan(beta)) {
                std::puts("nan in GMRES");
                return {false, its};
            }
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
        if (beta <= eps1) {
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
#endif

} // namespace lljbash
