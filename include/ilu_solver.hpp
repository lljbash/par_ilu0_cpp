#pragma once

#include <string>
#include "csc_matrix.h"

namespace lljbash {

class IluSolver {
public:
    IluSolver() : threads_(1), x_(nullptr), b_(nullptr) {}
    IluSolver(int dimension)
      : dimension_(dimension)
      , threads_(1)
      , x_(nullptr)
      , b_(nullptr)
    {}

    ~IluSolver();
    const double*    GetSolution()  const { return x_; }
    const CscMatrix& GetILUMatrix() const { return iluMatrix_; }
    CscMatrix* GetMatrix() { return &aMatrix_; }
    int GetDimension() const { return dimension_; }
    void SetupSubstitution(double* b, double* x) { b_ = b; x_ = x; }

    void SetThreads(int threads) { threads_ = threads; }

    void SetupMatrix();
    bool Factorize();
    void Substitute();
    void CollectLUMatrix();
private:
    int              dimension_;
    int              threads_;
    double*          x_;
    double*          b_;
    CscMatrix        aMatrix_;
    CscMatrix        iluMatrix_;

    //HERE, please add your L/U data stuctures
    struct Ext;
    Ext* ext_ = nullptr;
    struct ThreadLocalExt;
    ThreadLocalExt* extt_ = nullptr;
};

} // namespace lljbash
