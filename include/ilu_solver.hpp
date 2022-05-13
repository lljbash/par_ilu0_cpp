#pragma once

#include <string>
#include "csc_matrix.h"

namespace lljbash {

class IluSolver {
public:
    IluSolver() = default;

    ~IluSolver();
    const CscMatrix& GetILUMatrix() const { return iluMatrix_; }
    CscMatrix* GetMatrix() { return &aMatrix_; }

    void SetThreads(int threads) { threads_ = threads; }

    void SetupMatrix();
    bool Factorize();
    void Substitute(const double* b, double* x);
    void CollectLUMatrix();
private:
    int              threads_ = 1;
    CscMatrix        aMatrix_;
    CscMatrix        iluMatrix_;

    //HERE, please add your L/U data stuctures
    struct Ext;
    Ext* ext_ = nullptr;
    struct ThreadLocalExt;
    ThreadLocalExt* extt_ = nullptr;
};

} // namespace lljbash
