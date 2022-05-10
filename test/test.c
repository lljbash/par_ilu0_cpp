#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "csc_matvec.h"
#include "gmres_c.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s filename\n", argv[0]);
        return -1;
    }
    int ret = 0;
    IluSolverHdl ilu = IluSolverCreate(8);
    CscMatrix* mat = IluSolverGetMatrix(ilu);
    puts("Reading...");
    ReadCscMatrixMM1(mat, argv[1]);
    double* rhs = (double*) calloc(mat->size, sizeof(double));
    double* sol = (double*) calloc(mat->size, sizeof(double));
    double* x = (double*) calloc(mat->size, sizeof(double));
    for (int i = 0; i < mat->size; ++i) {
        sol[i] = 1.0;
    }
    CscMatVec(mat, sol, rhs);
    puts("Factorizing...");
    IluSolverFactorize(ilu, true);
    GmresHdl gmres = GmresCreate();
    GmresParameters* param = GmresGetParameters(gmres);
    GmresSetPreconditioner(gmres, ilu);
    puts("Solving...");
    int iter = 0;
    bool succ = GmresSolve(gmres, mat, rhs, x, &iter);
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
    for (int i = 0; i < mat->size; ++i) {
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
    IluSolverDestroy(ilu);
    return ret;
}
