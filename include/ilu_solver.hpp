#pragma once

#include <string>
#include "csr_matrix.h"

namespace lljbash {

class IluSolver {
public:
    IluSolver() = default;

    ~IluSolver();
    const CsrMatrix& GetILUMatrix() const { return aMatrix_; }
    CsrMatrix* GetMatrix() { return &aMatrix_; }

    void SetThreads(int threads) { threads_ = threads; }

    void SetupMatrix();
    bool Factorize();
    void Substitute(const double* b, double* x);
private:
    int              threads_ = 0;
    CsrMatrix        aMatrix_;

    //HERE, please add your L/U data stuctures
    struct Ext;
    Ext* ext_ = nullptr;
    struct ThreadLocalExt;
    ThreadLocalExt* extt_ = nullptr;
};

} // namespace lljbash
