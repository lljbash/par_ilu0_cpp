#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "csr_matvec.h"
#include "gmres_c.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s filename\n", argv[0]);
        return -1;
    }
    int ret = 0;
    IluSolverHdl ilu = IluSolverCreate(0);
    CsrMatrix a = CSR_MATRIX_DEFAULT;
    ReadCsrMatrixMM1(&a, argv[1]);
    CsrMatrix* ailu = IluSolverGetMatrix(ilu);
    CopyCsrMatrix(ailu, &a);
    puts("Reading...");
    double* rhs = (double*) calloc(a.size, sizeof(double));
    double* sol = (double*) calloc(a.size, sizeof(double));
    double* x = (double*) calloc(a.size, sizeof(double));
    for (int i = 0; i < a.size; ++i) {
        sol[i] = 1.0;
    }
    CsrMatVec(&a, sol, rhs);
    puts("Factorizing...");
    IluSolverFactorize(ilu, true);
    GmresHdl gmres = GmresCreate();
    GmresParameters* param = GmresGetParameters(gmres);
    GmresSetPreconditioner(gmres, ilu);
    puts("Solving...");
    int iter = 0;
    bool succ = GmresSolve(gmres, &a, rhs, x, &iter);
    if (!succ) {
        if (iter < param->max_iterations) {
            puts("gmres failed");
            ret = -2;
        }
        else {
            puts("not converged");
            ret = -3;
        }
        goto EXIT;
    }
    printf("iter: %d\n", iter);
    double max_err = 0;
    for (int i = 0; i < a.size; ++i) {
        double err = fabs(x[i] - sol[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    printf("max_err: %g\n", max_err);
EXIT:
    GmresDestroy(gmres);
    free(x);
    free(sol);
    free(rhs);
    DestroyCsrMatrix(&a);
    IluSolverDestroy(ilu);
    return ret;
}
