#include "gmres_c.h"
#include "gmres.hpp"

using namespace lljbash;

GmresHdl GmresCreate() {
    return static_cast<GmresHdl>(new PreconditionedGmres);
}

void GmresDestroy(GmresHdl hdl) {
    delete static_cast<PreconditionedGmres*>(hdl);
}

GmresParameters* GmresGetParameters(GmresHdl hdl) {
    return &static_cast<PreconditionedGmres*>(hdl)->param();
}

const GmresStat* GmresGetStat(GmresHdl hdl) {
    return &static_cast<PreconditionedGmres*>(hdl)->stat();
}

void GmresSetPreconditioner(GmresHdl hdl, IluSolverHdl ilu) {
    static_cast<PreconditionedGmres*>(hdl)
        ->SetIluPreconditioner(static_cast<IluSolver*>(ilu));
}

bool GmresSolve(GmresHdl hdl, const CsrMatrix* mat, double* rhs, double* sol, int* iter) {
    auto ret = static_cast<PreconditionedGmres*>(hdl)->Solve(mat, rhs, sol);
    if (iter) {
        *iter = ret.second;
    }
    return ret.first;
}
