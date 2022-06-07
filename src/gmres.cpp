#include "gmres.hpp"
#include <cmath>
#include <mkl.h>
#include "scope_guard.hpp"
#include "tictoc.hpp"

namespace lljbash {

static_assert(std::is_same_v<MKL_INT, int>);

PreconditionedGmres::~PreconditionedGmres() {
    mkl_free(mkl_.tmp);
    mkl_free(mkl_.b);
    mkl_free(mkl_.residual);
    MKL_Free_Buffers();
}

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

    tic = Rdtsc();
ONE:
    dfgmres(&n, sol, rhs, &RCI_request, ipar, dpar, tmp);
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
    stat_.total_fgmr_time += Toc(tic);
    ret = {true, iter};
    goto END;

FAILED:
    ret = {false, iter};
    goto END;

END:
    mkl_sparse_destroy(csrA);
    return ret;
}

} // namespace lljbash
