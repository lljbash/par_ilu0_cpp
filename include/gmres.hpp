#pragma once

#include <functional>
#include "gmres_param.h"
#include "ilu_solver.hpp"

namespace lljbash {

class PreconditionedGmres {
public:
    GmresParameters& param() {
        return param_;
    }

    void SetIluPreconditioner(IluSolver* ilu_solver) {
        precon_ = [ilu_solver](const double* rhs, double* sol) {
            ilu_solver->Substitute(rhs, sol);
        };
        precon_set_ = true;
    }

    std::pair<bool, int> Solve(const CscMatrix* mat, const double* rhs, double* sol);

private:
    GmresParameters param_;
    bool precon_set_ = false;
    std::function<void(const double*, double*)> precon_;
};

} // namespace lljbash
