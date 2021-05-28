
#include <iostream>
#include <fstream>
#include <memory.h>
#include "ILUSolver.h"

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
};

struct ILUSolver::ThreadLocalExt {
    int* colcol_k = nullptr;
    long* colcol_kptr = nullptr;
    double* colcol_akj = nullptr;
};

void             
ILUSolver::SetupMatrix() {
    // HERE, you could setup the reasonable stuctures of L and U as you want
    int n = aMatrix_.GetSize();
    ext_ = new Ext;
    ext_->diag_ptr = new long[n];
    extt_ = new ThreadLocalExt;
    extt_->colcol_k = new int[n];
    extt_->colcol_kptr = new long[n];
    extt_->colcol_akj = new double[n];
    iluMatrix_ = aMatrix_;
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
    memcpy(a, aMatrix_.GetValue(), sizeof(double[aMatrix_.GetNonZeros()]));
    for (int j = 0; j < n; ++j) {
        int cnt;
        long ji;
        for (ji = col_ptr[j], cnt = 0; ji < col_ptr[j+1]; ++ji, ++cnt) {
            int k = row_idx[ji];
            for (int kcnt = 0; kcnt < cnt; ++kcnt) {
                long& kptr = extt_->colcol_kptr[kcnt];
                while (kptr < col_ptr[extt_->colcol_k[kcnt]+1] && row_idx[kptr] < k) {
                    ++kptr;
                }
                if (kptr < col_ptr[extt_->colcol_k[kcnt]+1] && row_idx[kptr] == k) {
                    a[ji] -= a[kptr] * extt_->colcol_akj[kcnt];
                    ++kptr;
                }
            }
            if (k >= j) {
                break;
            }
            extt_->colcol_k[cnt] = k;
            extt_->colcol_kptr[cnt] = ext_->diag_ptr[k] + 1;
            extt_->colcol_akj[cnt] = a[ji];
        }
        double ajj = a[ji];
        ext_->diag_ptr[j] = ji;
        for (++ji; ji < col_ptr[j+1]; ++ji) {
            int k = row_idx[ji];
            for (int kcnt = 0; kcnt < cnt; ++kcnt) {
                long& kptr = extt_->colcol_kptr[kcnt];
                while (kptr < col_ptr[extt_->colcol_k[kcnt]+1] && row_idx[kptr] < k) {
                    ++kptr;
                }
                if (kptr < col_ptr[extt_->colcol_k[kcnt]+1] && row_idx[kptr] == k) {
                    a[ji] -= a[kptr] * extt_->colcol_akj[kcnt];
                    ++kptr;
                }
            }
            a[ji] /= ajj;
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

