#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <complex>
#include <iomanip>
#include <vector>
#include <set>
#include "ILUSolver.h"

using CPUCYCLE = unsigned long long;

static double s_cycletime = 0;

static CPUCYCLE GetCycleCount()
{
#if defined(__GNUC__) && defined(__i386__)
  /** This works much better if compiled with "-O3". */
    CPUCYCLE result;
  __asm__ __volatile__ ("rdtsc" : "=A" (result));
  return result;
#elif defined(__GNUC__) && defined(__x86_64__)
  CPUCYCLE result;
  __asm__ __volatile__ ("rdtsc\n\t" \
                        "shlq $32,%%rdx\n\t" \
                        "orq %%rdx,%%rax"
                        : "=a" (result) :: "%edx");
  return result;
#else
  return 0;
#endif
}

static void UpdateCycleTime(double precision)
{

    CPUCYCLE total = (unsigned long long)(3.0e2 / precision);
    CPUCYCLE start = GetCycleCount();
    usleep(total);   ///sleep 0.06s
    CPUCYCLE stop = GetCycleCount();
    s_cycletime = total * 1.0e-6 / (stop - start);
}

double ComputeNorm(const double* vec, int size) {
    double normSum = 0.0;
    for (int i = 0; i < size; ++i) {
        normSum += std::norm(vec[i]);
    }
    return sqrt(normSum / size);
}

bool
CheckPatterns(const CCSMatrix& origA, const CCSMatrix& iluMatrix) {
    const long int* origAColPtr = origA.GetColumnPointer();
    const int*      origARowIdx = origA.GetRowIndex();
    const long int* iluColPtr = iluMatrix.GetColumnPointer();
    const int*      iluRowIdx = iluMatrix.GetRowIndex();
    if (origAColPtr == iluColPtr && origARowIdx == iluRowIdx) return true;
    const int       size = origA.GetSize(); 
    const long int  nnz  = origAColPtr[size];
    return !memcmp(origAColPtr, iluColPtr, (size + 1) * sizeof(long int)) && 
        !memcmp(origARowIdx, iluRowIdx, nnz * sizeof(int));
}

int
CheckMatrixValue(double& maxdiff, const CCSMatrix& iluMatrix, const CCSMatrix& refMatrix) {
    int diffCnt = 0;
    const long int* refColPtr = refMatrix.GetColumnPointer();
    const long int* iluColPtr = iluMatrix.GetColumnPointer();
    const int       size = iluMatrix.GetSize(); 
    const long int  nnz = iluColPtr[size];
    assert(nnz == refColPtr[size]);
    const double*   iluVal    = iluMatrix.GetValue();
    const double*   refVal    = refMatrix.GetValue();
    
    for (long int pos = 0; pos < nnz; ++pos) {
        const double iluV = iluVal[pos]; 
        const double refV = refVal[pos]; 
        double diff = fabs(iluV - refV);
        maxdiff = std::max(diff, maxdiff);
        if (Float::LessEqual(diff, 1e-6)) continue;
        if (Float::LessEqual(diff/std::min(fabs(iluV),fabs(refV)), 1e-3)) continue;
        ++diffCnt;
    }
    return diffCnt;
}

bool
EvaluateLUResult(const CCSMatrix& iluMatrix, const CCSMatrix& refMatrix) {
    bool success = CheckPatterns(refMatrix, iluMatrix);
    if (!success) {
        std::cout << "The result of ilu0 factorization should have the same patterns as original A matrix!" << std::endl;
        return false; 
    }

    double maxDiff = 0;
    int diffPoint = CheckMatrixValue(maxDiff, iluMatrix, refMatrix);
    if (diffPoint) {
        std::cout << "The accuracy check of ilu0 factorization fails!" << std::endl
            << "\tDifferent Points: " << diffPoint << std::endl
            << "\tMax Difference: " << std::setiosflags(std::ios::scientific) << std::setprecision(15) << maxDiff << std::resetiosflags(std::ios::scientific) << std::endl;
        return false; 
    }
    
    return true;
}

int
EvaluateSolution(double& maxdiff, const double* sol, const double* ref, int dim) {
    int diffCnt = 0;
    for (int i = 0; i < dim; ++i) {
        double diff = fabs(sol[i] - ref[i]);
        maxdiff = std::max(diff, maxdiff);
        if (Float::Greater(diff, 1e-6)) {
            ++diffCnt;
        }
    }
    return diffCnt;
}

typedef std::vector<std::string> StrVec;
typedef std::set<std::string> StrSet;

bool
CollectFiles(StrVec& fileList, const std::string& path, const std::string& list) {
    std::string filePath = path + "/" + list;  
    std::ifstream ifs(filePath);
    if(ifs.is_open()) {
        StrSet fileSet;
        while (1) {
            std::string f; ifs >> f;
            if (f.empty()) break;
            if (f.find("#")==0 || f.find("*")==0 || f.find("//")==0) continue;
            if (fileSet.insert(f).second) {
                if (f[0] != '/') {
                    char buf[PATH_MAX];
                    f = path + "/" + f; 
                    fileList.emplace_back(f);
                }
            }
        }
        ifs.close();
    } else {
        ifs.open(path);
        if (ifs.is_open()) {
            fileList.emplace_back(path); 
            ifs.close();
        } else {
            std::cout << "Can't open matrix in " << path << std::endl;
            return false;
        }
    }
    return true;
}

void 
RunSolver(ILUSolver* solver, int loop, bool needSetup = true) {
    ProfData profData;
    CPUCYCLE PC = GetCycleCount(), startPC = PC;
    if (needSetup) {
        solver->SetupMatrix();
        profData.setupPC_ += GetCycleCount() - PC; 
        ++profData.setupTime_; 
        PC = GetCycleCount();
    }

    while (loop--) {
        solver->Factorize();
        profData.factorPC_ += GetCycleCount() - PC; 
        ++profData.factorTime_; 
        PC = GetCycleCount();

        solver->Substitute();
        profData.solvePC_ += GetCycleCount() - PC; 
        ++profData.solveTime_; 
    }
    PC = GetCycleCount();
    solver->CollectLUMatrix();
    profData.collectPC_ += GetCycleCount() - PC; 
    profData.Dump(solver->GetDimension(), s_cycletime);
}

bool CollectReferenceResult(int dim, CCSMatrix& refMatrix, double* refSolution, const std::string& matFilePath) {
    refMatrix.LoadDenseFromFile(dim, matFilePath + "_refMat");
    assert(dim == refMatrix.GetSize());
    const std::string refSolFile = matFilePath + "_refSol";
    
    std::ifstream ifs(refSolFile);
    if (!ifs.is_open()) {
        std::cout << "Reference Solution file " << refSolFile << " does not exist " << std::endl;
        return false;
    }

    for (int i = 0; i < dim; ++i)
        ifs >> refSolution[i];
    return true;
}

bool EvaluateResult(const ILUSolver* solver, const CCSMatrix& refMatrix, const double* refSolution) {
    bool success = EvaluateLUResult(solver->GetILUMatrix(), refMatrix);
    double maxdiff = 0;
    int diffPoint = EvaluateSolution(maxdiff, solver->GetSolution(), refSolution, refMatrix.GetSize());
    if (diffPoint) {
        std::cout << "The accuracy check of substitution fails!" << std::endl
            << "\tDifferent Points: " << diffPoint << std::endl
            << "\tMax Difference: " << std::setiosflags(std::ios::scientific) << std::setprecision(15) << maxdiff << std::resetiosflags(std::ios::scientific) << std::endl;
    }
    return success && !diffPoint;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Input ERROR!\nUsage: ./demo mode inputDir" << std::endl;
        std::cout << "mode: \
                            thread# \
                            loop# "
                  << std::endl;

        return -1;
    }
    
    UpdateCycleTime(5e-3);

    const std::string mode(argv[1]);
    std::string inputDir(argv[2]);
    assert(!inputDir.empty());
    if (inputDir[0] != '/') {
        char buf[PATH_MAX];
        inputDir = getcwd(buf, sizeof(buf)) + std::string("/") + inputDir;
    }

    // mode
    const int loop = 
        (std::string::npos != mode.find("loop")) ? atoi(mode.substr(mode.find("loop")+4).c_str()) : 10;
    const int threads = 
        (std::string::npos != mode.find("thread")) ? atoi(mode.substr(mode.find("thread")+6).c_str()) : 16;
   
    CPUCYCLE PC = GetCycleCount();
    // read files
    StrVec matFileList;
    CollectFiles(matFileList, inputDir, "amat_list");
    
    std::unique_ptr<ILUSolver> solver(new ILUSolver);
    std::unique_ptr<double[]> refSol = nullptr;
    solver->SetThreads(threads);
    for (int i = 0; i < matFileList.size(); ++i) {
        const std::string& matFile = matFileList[i];
        bool needSetup = (i == 0);
        std::cout << "******************************************************" << std::endl;
        std::cout << "Test Case No. " << i + 1 << " : " << matFile << std::endl;
        solver->ReadAMatrix(matFile);  //read matrix before reading rhs
        const std::string& rhsFile = matFile + "_rhs"; 
        std::ifstream frhs(rhsFile);
        if (frhs.is_open()) {
            solver->ReadRhs(rhsFile, false/*sparse*/);
            frhs.close();
        } else {
            solver->GenerateRhs(1);
        }

        RunSolver(solver.get(), loop, needSetup);

        CCSMatrix refMatrix;
        if (needSetup) {
            refSol.reset(new double[solver->GetDimension()]);
            refSol.get_deleter() = std::default_delete<double[]>();
        }
        CollectReferenceResult(solver->GetDimension(), refMatrix, refSol.get(), matFile);
        bool success = EvaluateResult(solver.get(), refMatrix, refSol.get());
        std::cout << "Result " << (success ? "Pass!" : "FAIL!!") << std::endl;
        std::cout << "******************************************************" << std::endl;
    }
    return 0;
}


