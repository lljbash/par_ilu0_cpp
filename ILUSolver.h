
#ifndef _ILUSOLVER_H_ 
#define _ILUSOLVER_H_ 

#include "CommonDef.h"

using namespace ILUSOLVER;

class ILUSolver {
public:
    ILUSolver() : threads_(1), x_(nullptr), b_(nullptr) {}
    ILUSolver(int dimension) 
      : dimension_(dimension)
      , threads_(1)
      , x_(nullptr)
      , b_(nullptr)
    {}

    ~ILUSolver();
    const double*    GetSolution()  const { return x_; }
    const CCSMatrix& GetILUMatrix() const { return iluMatrix_; }
    CCSMatrix* GetMatrix() { return &aMatrix_; }
    int GetDimension() const { return dimension_; }

    void SetThreads(int thread) { threads_ = thread; }
    bool GenerateRhs(double v, bool random = false); 
    bool ReadRhs(const std::string& fname, bool sparse = false);
    bool ReadAMatrix(const std::string& fname);

    void SetupMatrix();         
    bool Factorize();          
    void Substitute();           
    void CollectLUMatrix();     
private:
    int              dimension_;
    int              threads_;  
    double*          x_;  
    double*          b_;  
    CCSMatrix        aMatrix_;  
    CCSMatrix        iluMatrix_;

    //HERE, please add your L/U data stuctures
    struct Ext;
    Ext* ext_ = nullptr;
    struct ThreadLocalExt;
    ThreadLocalExt* extt_ = nullptr;
};

#endif
