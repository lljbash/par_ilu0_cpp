#include "gmres_c.h"
#include "gmres.hpp"

using namespace lljbash;

GmresHdl GmresCreate() {
    return reinterpret_cast<GmresHdl>(new PreconditionedGmres);
}

void GmresDestroy(GmresHdl hdl) {
    delete reinterpret_cast<PreconditionedGmres*>(hdl);
}

GmresParameters* GmresGetParameters(GmresHdl hdl) {
    return &reinterpret_cast<PreconditionedGmres*>(hdl)->param();
}

void GmresSetPreconditioner(GmresHdl hdl, IluSolverHdl ilu) {
    reinterpret_cast<PreconditionedGmres*>(hdl)
        ->SetIluPreconditioner(reinterpret_cast<IluSolver*>(ilu));
}

bool GmresSolve(GmresHdl hdl, const CsrMatrix* mat, const double* rhs, double* sol, int* iter) {
    auto ret = reinterpret_cast<PreconditionedGmres*>(hdl)->Solve(mat, rhs, sol);
    if (iter) {
        *iter = ret.second;
    }
    return ret.first;
}
