#include "ilu_solver_c.h"
#include "ilu_solver.hpp"

using namespace lljbash;

IluSolverHdl IluSolverCreate(int nthreads) {
    auto hdl = new IluSolver;
    hdl->SetThreads(nthreads);
    return hdl;
}

void IluSolverDestroy(IluSolverHdl hdl) {
    delete reinterpret_cast<IluSolver*>(hdl);
}

CscMatrix* IluSolverGetMatrix(IluSolverHdl hdl) {
    return reinterpret_cast<IluSolver*>(hdl)->GetMatrix();
}

int IluSolverFactorize(IluSolverHdl hdl, bool different_structure) {
    IluSolver& solver = *reinterpret_cast<IluSolver*>(hdl);
    try {
        if (different_structure) {
            solver.SetupMatrix();
        }
        solver.Factorize();
        solver.CollectLUMatrix();
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
