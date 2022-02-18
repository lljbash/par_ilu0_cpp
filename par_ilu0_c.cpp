#include "par_ilu0_c.h"
#include "ILUSolver.h"
#include <cstdlib>
#include <numeric>

int par_ilu0_c(csptr csmat, iluptr lu) {
    ILUSolver solver;
    int n = csmat->n;
    long nnz = std::accumulate(csmat->nzcount, csmat->nzcount + n, 0l);

    // csmat -> amat
    {
        CCSMatrix* amat = solver.GetMatrix();
        amat->Setup(csmat->n, nnz);
        long* col_ptr = amat->GetColumnPointer();
        int* row_idx = amat->GetRowIndex();
        double* value = amat->GetValue();
        std::fill_n(col_ptr, n + 1, 0);
        for (int i = 0; i < n; ++i) {
            for (int ii = 0; ii < csmat->nzcount[i]; ++ii) {
                int j = csmat->ja[i][ii];
                ++col_ptr[j+1];
            }
        }
        for (int j = 0; j < n; ++j) {
            col_ptr[j+1] += col_ptr[j];
        }
        for (int i = 0; i < n; ++i) {
            for (int ii = 0; ii < csmat->nzcount[i]; ++ii) {
                int j = csmat->ja[i][ii];
                long ji = col_ptr[j]++;
                row_idx[ji] = i;
                value[ji] = csmat->ma[i][ii];
            }
        }
        for (int j = n; j > 0; --j) {
            col_ptr[j] = col_ptr[j-1];
        }
        col_ptr[0] = 0;
    }

    // factorize
    try {
        solver.SetThreads(8);
        solver.SetupMatrix();
        solver.Factorize();
        solver.CollectLUMatrix();
    }
    catch (const std::runtime_error& e) {
        lu->D = nullptr;
        lu->L = nullptr;
        lu->U = nullptr;
        lu->work = nullptr;
        if (std::string(e.what()) == "missing diag") {
            return -2;
        }
        else {
            return -1;
        }
    }

    // ilumat -> lu
    {
        const CCSMatrix& ilumat = solver.GetILUMatrix();
        const long* col_ptr = ilumat.GetColumnPointer();
        const int* row_idx = ilumat.GetRowIndex();
        const double* value = ilumat.GetValue();

        lu->n = n;
        lu->D = (double*) std::calloc(n, sizeof(double));
        lu->L = (csptr) std::malloc(sizeof(SparMat));
        lu->L->n = n;
        lu->L->nzcount = (int*) std::calloc(n, sizeof(int));
        lu->L->ja = (int**) std::calloc(n, sizeof(int*));
        lu->L->ma = (double**) std::calloc(n, sizeof(double*));
        lu->U = (csptr) std::malloc(sizeof(SparMat));
        lu->U->n = n;
        lu->U->nzcount = (int*) std::calloc(n, sizeof(int));
        lu->U->ja = (int**) std::calloc(n, sizeof(int*));
        lu->U->ma = (double**) std::calloc(n, sizeof(double*));
        lu->work = (int*) std::malloc(sizeof(int[n]));
        std::fill_n(lu->work, n, -1);

        for (int j = 0; j < n; ++j) {
            for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
                int i = row_idx[ji];
                if (i < j) {
                    ++lu->U->nzcount[i];
                }
                else if (i > j) {
                    ++lu->L->nzcount[i];
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            lu->L->ja[i] = (int*) std::calloc(lu->L->nzcount[i], sizeof(int));
            lu->L->ma[i] = (double*) std::calloc(lu->L->nzcount[i], sizeof(double));
            lu->L->nzcount[i] = 0;
            lu->U->ja[i] = (int*) std::calloc(lu->U->nzcount[i], sizeof(int));
            lu->U->ma[i] = (double*) std::calloc(lu->U->nzcount[i], sizeof(double));
            lu->U->nzcount[i] = 0;
        }
        for (int j = 0; j < n; ++j) {
            for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
                int i = row_idx[ji];
                double v = value[ji];
                if (i < j) {
                    int ii = lu->U->nzcount[i]++;
                    lu->U->ja[i][ii] = j;
                    lu->U->ma[i][ii] = v;
                }
                else if (i == j) {
                    lu->D[i] = 1.0 / v;
                }
                else if (i > j) {
                    int ii = lu->L->nzcount[i]++;
                    lu->L->ja[i][ii] = j;
                    lu->L->ma[i][ii] = v;
                }
            }
        }
    }

    return 0;
}