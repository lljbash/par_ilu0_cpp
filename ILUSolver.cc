#include "ILUSolver.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <omp.h>
#include "scope_guard.h"

bool             
ILUSolver::ReadRhs(const std::string& fname, bool sparse) {
    if (b_) { delete [] b_; }
    b_ = new double[dimension_];
    memset(b_, 0, sizeof(double) * dimension_);

    std::ifstream ifile(fname);
    if (!ifile.is_open()) {
        std::cout << "Vector file " << fname << " not exist " << std::endl;
        return false;
    }

    if (!sparse) {
        for (int i = 0; i < dimension_; ++i)
            ifile >> b_[i];
        return true;
    }

    int size = 0;
    long int nnz = 0;
    ifile >> size >> nnz;
    if (size != dimension_) {
        std::cout << "Wrong right-hand-side size!" << std::endl;
        return false;
    }
    int row = 0; double v = 0;
    for (int i = 0; i < nnz; ++i) {
        ifile >> row >> v;
        b_[row - 1] = v;
    }
    return true;
}

bool
ILUSolver::GenerateRhs(double v, bool random) {
    if (b_) { delete [] b_; }
    b_ = new double[dimension_];
    if (!random) {
        for (int i = 0; i < dimension_; ++i) {
            b_[i] = v; 
        }
        return true;
    }
    
    // as rhs will be given by file, do not implement random generator here
    // you can add your own here
    return true; 
}

bool
ILUSolver::ReadAMatrix(const std::string& fname) {
    aMatrix_.LoadFromFile(fname);
    dimension_ = aMatrix_.GetSize();
    return true;
}

struct ILUSolver::Ext {
    long* diag_ptr = nullptr;

    struct CSR {
        long* row_ptr = nullptr;
        int* col_idx = nullptr;
        long* diag_ptr = nullptr;
        int* nnz_cnt = nullptr;
    } csr;

    double* dense_col = nullptr;
    long* col_rowj_start = nullptr;
};

struct ILUSolver::ThreadLocalExt {
};

void             
ILUSolver::SetupMatrix() {
    // HERE, you could setup the reasonable stuctures of L and U as you want
    omp_set_dynamic(0);
    omp_set_num_threads(threads_);

    int n = aMatrix_.GetSize();
    long* col_ptr = aMatrix_.GetColumnPointer();
    int* row_idx = aMatrix_.GetRowIndex();
    long nnz = aMatrix_.GetNonZeros();
    ext_ = new Ext;
    ext_->diag_ptr = new long[n];
    ext_->csr.row_ptr = new long[n+1](); // initialized to zero
    ext_->csr.col_idx = new int[nnz];
    ext_->csr.diag_ptr = new long[n];
    ext_->csr.nnz_cnt = new int[n](); // initialized to zero

    // get diag_ptr
#pragma omp parallel
    for (int j = 0; j < n; ++j) {
        for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            if (row_idx[ji] >= j) {
                ext_->diag_ptr[j] = ji;
                break;
            }
        }
    }

    // CSC -> CSR
#pragma omp parallel for
    for (long ji = 0; ji < nnz; ++ji) {
#pragma omp atomic
        ++ext_->csr.row_ptr[row_idx[ji] + 1];
    }
    for (int i = 0; i < n; ++i) {
        ext_->csr.row_ptr[i+1] += ext_->csr.row_ptr[i];
    }
    for (int j = 0; j < n; ++j) {
#pragma omp parallel for
        for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            int i = row_idx[ji];
            long ii = ext_->csr.row_ptr[i] + ext_->csr.nnz_cnt[i];
            ext_->csr.col_idx[ii] = j;
            ++ext_->csr.nnz_cnt[i];
            if (i == j) {
                ext_->csr.diag_ptr[i] = ii;
            }
        }
    }

    ext_->dense_col = new double[n](); // should always be zero before use
    ext_->col_rowj_start = new long[n];

    iluMatrix_ = aMatrix_;
}

static void right_looking_col(int k, const long* col_ptr, const int* row_idx, double* a,
                              long* col_rowj_start, const double* dense_col) {
    double ajk = a[col_rowj_start[k]++];
    for (long ii = col_rowj_start[k]; ii < col_ptr[k+1]; ++ii) {
        a[ii] -= ajk * dense_col[row_idx[ii]];
    }
}

bool             
ILUSolver::Factorize() {
    // HERE, do triangle decomposition 
    // to calculate the values in L and U
    //iluMatrix_ = aMatrix_;
    int n = iluMatrix_.GetSize();
    long* col_ptr = iluMatrix_.GetColumnPointer();
    int* row_idx = iluMatrix_.GetRowIndex();
    double* a = iluMatrix_.GetValue();
    std::memcpy(a, aMatrix_.GetValue(), sizeof(double[aMatrix_.GetNonZeros()]));

    std::memcpy(ext_->col_rowj_start, col_ptr, sizeof(long[n]));
    for (int i = 0; i < n; ++i) {
        // aji /= aii
        double aii = a[ext_->diag_ptr[i]];
        for (long ji = ext_->diag_ptr[i] + 1; ji < col_ptr[i+1]; ++ji) {
            ext_->dense_col[row_idx[ji]] = (a[ji] /= aii);
        }

        // ajk -= aji * aik
#pragma omp parallel for schedule(dynamic, 4)
        for (long ki = ext_->csr.diag_ptr[i] + 1; ki < ext_->csr.row_ptr[i+1]; ++ki) {
            right_looking_col(ext_->csr.col_idx[ki], col_ptr, row_idx, a,
                              ext_->col_rowj_start, ext_->dense_col);
        }

        // recover dense_col
        for (long ji = ext_->diag_ptr[i] + 1; ji < col_ptr[i+1]; ++ji) {
            ext_->dense_col[row_idx[ji]] = 0;
        }
    }

    return true;
}

void
ILUSolver::Substitute() {
    // HERE, use the L and U calculated by ILUSolver::Factorize to solve the triangle systems 
    // to calculate the x
    int n = iluMatrix_.GetSize();
    long* col_ptr = iluMatrix_.GetColumnPointer();
    int* row_idx = iluMatrix_.GetRowIndex();
    double* a = iluMatrix_.GetValue();
    x_ = new double[n];
    memcpy(x_, b_, sizeof(double[n]));
    for (int j = 0; j < n; ++j) {
        for (long ji = ext_->diag_ptr[j] + 1; ji < col_ptr[j+1]; ++ji) {
            int i = row_idx[ji];
            x_[i] -= x_[j] * a[ji];
        }
    }
    for (int j = n - 1; j >= 0; --j) {
        x_[j] /= a[ext_->diag_ptr[j]];
        for (long ji = col_ptr[j]; ji < ext_->diag_ptr[j]; ++ji) {
            int i = row_idx[ji];
            x_[i] -= x_[j] * a[ji];
        }
    }
}

void    
ILUSolver::CollectLUMatrix() {
    // put L and U together into iluMatrix_ 
    // as the diag of L is 1, set the diag of iluMatrix_ with u 
    // iluMatrix_ should have the same size and patterns with aMatrix_
}    

