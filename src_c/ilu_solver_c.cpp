#include "ilu_solver_c.h"
#include "ilu_solver.hpp"

using namespace lljbash;

IluSolverHdl IluSolverCreate(int nthreads) {
    auto hdl = new IluSolver;
    hdl->SetThreads(nthreads);
    return hdl;
}

void IluSolverDestroy(IluSolverHdl hdl) {
    delete static_cast<IluSolver*>(hdl);
}

CsrMatrix* IluSolverGetMatrix(IluSolverHdl hdl) {
    return static_cast<IluSolver*>(hdl)->GetMatrix();
}

int IluSolverSetup(IluSolverHdl hdl) {
    IluSolver& solver = *static_cast<IluSolver*>(hdl);
    try {
        solver.SetupMatrix();
    }
    catch (const std::exception& e) {
        if (std::string(e.what()) == "missing diag") {
            return -2;
        }
        else {
            return -1;
        }
    }
    return 0;
}

int IluSolverFactorize(IluSolverHdl hdl) {
    IluSolver& solver = *static_cast<IluSolver*>(hdl);
    try {
        solver.Factorize();
    }
    catch (const std::exception& e) {
        if (std::string(e.what()) == "missing diag") {
            return -2;
        }
        else {
            return -1;
        }
    }
    return 0;
}
