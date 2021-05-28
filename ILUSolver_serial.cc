
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
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
    int* hash_key;
    long* hash_value;
};

namespace {

class HashTable {
public:
    HashTable(unsigned size, int* key, long* value) : mask_(size), key_(key), value_(value) {
        mask_ |= mask_ >> 16;
        mask_ |= mask_ >> 8;
        mask_ |= mask_ >> 4;
        mask_ |= mask_ >> 2;
        mask_ |= mask_ >> 1;
        memset(key_, 0xff, sizeof(int[mask_ + 1]));
    }
    void set(int key, long value) {
        unsigned code = hash_code(key);
        while (key_[code] >= 0) {
            code = (code + 1) & mask_;
        }
        key_[code] = key;
        value_[code] = value;
    }
    long get(int key) const {
        for (unsigned code = hash_code(key); key_[code] >= 0; code = (code + 1) & mask_) {
            if (key_[code] == key) {
                return value_[code];
            }
        }
        return -1;
    }
private:
    int hash_code(int x) const {
        static constexpr unsigned prime = 23333;
        return (static_cast<unsigned>(x) * prime) & mask_;
    };

    unsigned mask_;
    int* key_;
    long* value_;
};

constexpr int hash_size_multiplier = 3;

}

void             
ILUSolver::SetupMatrix() {
    // HERE, you could setup the reasonable stuctures of L and U as you want
    int n = aMatrix_.GetSize();
    ext_ = new Ext;
    ext_->diag_ptr = new long[n];
    extt_ = new ThreadLocalExt;
    long max_nnz = 0;
    long* col_ptr = aMatrix_.GetColumnPointer();
    for (int j = 0; j < n; ++j) {
        max_nnz = std::max(max_nnz, col_ptr[j+1] - col_ptr[j]);
    }
    extt_->hash_key = new int[max_nnz * hash_size_multiplier * 2];
    extt_->hash_value = new long[max_nnz * hash_size_multiplier * 2];
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
    std::memcpy(a, aMatrix_.GetValue(), sizeof(double[aMatrix_.GetNonZeros()]));
    for (int j = 0; j < n; ++j) {
        int col_nnz = static_cast<int>(col_ptr[j+1] - col_ptr[j]);
        HashTable colj_row_pos(col_nnz * hash_size_multiplier, extt_->hash_key, extt_->hash_value);
        long ji;
        for (ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            colj_row_pos.set(row_idx[ji], ji);
        }
        for (ji = col_ptr[j]; row_idx[ji] < j; ++ji) {
            int k = row_idx[ji];
            for (long ki = ext_->diag_ptr[k] + 1; ki < col_ptr[k+1]; ++ki) {
                long aij_pos = colj_row_pos.get(row_idx[ki]);
                if (aij_pos > 0) {
                    a[aij_pos] -= a[ki] * a[ji];
                }
            }
        }
        double ajj = a[ji];
        ext_->diag_ptr[j] = ji;
        for (++ji; ji < col_ptr[j+1]; ++ji) {
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

