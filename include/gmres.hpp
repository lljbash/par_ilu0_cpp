#pragma once

#include <functional>
#include "gmres_param.h"
#include "ilu_solver.hpp"

namespace lljbash {

class PreconditionedGmres {
public:
    ~PreconditionedGmres();

    GmresParameters& param() { return param_; }
    GmresStat& stat() { return stat_; }

    void SetIluPreconditioner(IluSolver* ilu_solver) {
        precon_ = [ilu_solver](const double* rhs, double* sol) {
            ilu_solver->Substitute(rhs, sol);
        };
    }

    std::pair<bool, int> Solve(const CsrMatrix* mat, double* rhs, double* sol);

private:
    void SetupMkl(int n);

    GmresParameters param_;
    std::function<void(const double*, double*)> precon_;
    GmresStat stat_;
    struct {
        bool setup = false;
        int ipar[128];
        double dpar[128];
        double* tmp;
        double* b;
        double* residual;
    } mkl_;
};

} // namespace lljbash
