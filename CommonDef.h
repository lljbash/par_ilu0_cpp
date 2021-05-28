#ifndef _COMMONDEF_H_
#define _COMMONDEF_H_

#include "memory.h"
#include "math.h"
#include "float.h"
#include <vector>
#include <map>
namespace ILUSOLVER {

class Float{
public:
    static bool ToBool(double d)                            { return fabs(d) > DBL_MIN; }
    static bool IsZero(double d)                            { return fabs(d) < DBL_MIN; }
    static bool Equal(double d1, double d2)                 { return fabs(d1 - d2) < DBL_MIN; }
    static bool GreaterEqual(double d1, double d2)          { return (d1 - d2) >= -DBL_MIN;}
    static bool Greater(double d1, double d2)               { return (d1 - d2) > DBL_MIN;       }
    static bool LessEqual(double d1, double d2)             { return (d1 - d2) <= DBL_MIN;}
    static bool Less(double d1, double d2)                  { return (d1 - d2) < -DBL_MIN;       }
    static bool NotEqual(double d1, double d2)              { return !Equal(d1, d2); }
};

class ProfData {
    typedef unsigned long long CPUCYCLE;
public:
    int setupTime_;
    int factorTime_;
    int solveTime_;
    CPUCYCLE readPC_;
    CPUCYCLE setupPC_;
    CPUCYCLE factorPC_;
    //CPUCYCLE lSolvePC_;
    //CPUCYCLE uSolvePC_;
    CPUCYCLE solvePC_;
    CPUCYCLE collectPC_;
    CPUCYCLE evalPC_;
    void Clear() { memset(this, 0, sizeof(ProfData)); }
    void Dump(int dim, double ct);
    ProfData() { Clear(); }
};

class CCSMatrix {
    typedef std::map<int, double>      MatrixColData;
    typedef std::vector<MatrixColData> MatrixData;
public:
    const long int*     GetColumnPointer() const    { return columnPointers_;   }
    const int*          GetRowIndex() const         { return rowIndex_;         }
    const double*       GetValue() const            { return value_;            }
    long int*           GetColumnPointer()          { return columnPointers_;   }
    int*                GetRowIndex()               { return rowIndex_;         }
    double*             GetValue()                  { return value_;            }
    int                 GetSize() const             { return size_;             }
    long int            GetNonZeros() const         { return nonZeros_;         }

    CCSMatrix() 
      : columnPointers_(nullptr)
      , rowIndex_(nullptr)
      , value_(nullptr)
    {}
    
    CCSMatrix(int size, long int nnz) 
      : size_(size)
      , nonZeros_(nnz) 
      , columnPointers_(new long int[size_ + 1])
      , rowIndex_(new int[nonZeros_])
      , value_(new double[nonZeros_])
    {}

    //please add your own copy constructor and operator=
    CCSMatrix& operator=(const CCSMatrix& A){
        size_ = A.GetSize();
        nonZeros_ = A.GetNonZeros();

        if (columnPointers_) delete [] columnPointers_;
        columnPointers_ = new long int[size_ + 1];
        memcpy(columnPointers_, A.GetColumnPointer(), sizeof(long int) * (size_ + 1));

        if (rowIndex_) delete [] rowIndex_;
        rowIndex_ = new int[nonZeros_];
        memcpy(rowIndex_, A.GetRowIndex(), sizeof(int) * nonZeros_);

        if (value_) delete [] value_;
        value_ = new double[nonZeros_];
        memcpy(value_, A.GetValue(), sizeof(double) * nonZeros_);

        return *this;
    }

    void Setup(int size, long int nnz) {
        size_ = size;
        nonZeros_ = nnz;
        columnPointers_ = new long int[size_ + 1];
        rowIndex_ = new int[nonZeros_];
        value_ = new double[nonZeros_];
    }

    ~CCSMatrix() {
        if (columnPointers_) { delete [] columnPointers_; columnPointers_ = nullptr; }
        if (rowIndex_)       { delete [] rowIndex_; rowIndex_ = nullptr; }
        if (value_)          { delete [] value_; value_ = nullptr; }
    }


    void PrintOnMatrix(std::ostream& os) const;
    void PrintOnCCSMatrix(std::ostream& os) const;
    bool LoadFromFile(const std::string& fname);
    bool LoadDenseFromFile(int dim, const std::string& fname);
private:
    int                     size_;
    long int                nonZeros_;
    long int*               columnPointers_;
    int*                    rowIndex_;
    double*                 value_;
};
}
#endif // _COMMONDEF_H_
