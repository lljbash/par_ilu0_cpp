
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <omp.h>
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

struct ILUSolver::Ext {
    long* diag_ptr = nullptr;
    std::atomic_int* dep_cnt = nullptr;
    std::atomic_int* task_queue = nullptr;
    std::atomic_int task_head;
    std::atomic_int task_tail;

    struct CSR {
        long* row_ptr = nullptr;
        int* col_idx = nullptr;
        int* a = nullptr;
        long* diag_ptr = nullptr;
        int* nnz_cnt = nullptr;
    } csr;
};

struct ILUSolver::ThreadLocalExt {
    int* hash_key = nullptr;
    long* hash_value = nullptr;
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
    long max_nnz = 0;
    ext_ = new Ext;
    ext_->diag_ptr = new long[n];
    ext_->dep_cnt = new std::atomic_int[n];
    ext_->task_queue = new std::atomic_int[n];
    ext_->csr.row_ptr = new long[n+1](); // initialized to zero
    ext_->csr.col_idx = new int[nnz];
    //ext_->csr.a = new int[nnz];
    ext_->csr.diag_ptr = new long[n];
    ext_->csr.nnz_cnt = new int[n](); // initialized to zero

#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            if (row_idx[ji] >= j) {
                ext_->diag_ptr[j] = ji;
                break;
            }
        }
        max_nnz = std::max(max_nnz, col_ptr[j+1] - col_ptr[j]);
    }

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

    extt_ = new ThreadLocalExt[threads_];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        extt_[tid].hash_key = new int[max_nnz * hash_size_multiplier * 2];
        extt_[tid].hash_value = new long[max_nnz * hash_size_multiplier * 2];
    }

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
    ext_->task_head = 0;
    ext_->task_tail = 0;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        ext_->task_queue[i].store(-1, std::memory_order_relaxed);
    }
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        int deps = static_cast<int>(ext_->diag_ptr[j] - col_ptr[j]);
        ext_->dep_cnt[j].store(deps, std::memory_order_relaxed);
        if (deps == 0) {
            int new_task = ext_->task_tail.fetch_add(1, std::memory_order_relaxed);
            ext_->task_queue[new_task] = j;
        }
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        while (true) {
            // get task
            int task_id = ext_->task_head.fetch_add(1, std::memory_order_relaxed);
            if (task_id >= n) {
                break;
            }
            //std::printf("%d: wait %d\n", tid, task_id);
            while (ext_->task_queue[task_id].load(std::memory_order_relaxed) < 0);
            //std::printf("%d: exec %d\n", tid, task_id);
            int j = ext_->task_queue[task_id].load(std::memory_order_relaxed);

            // execute task
            int col_nnz = static_cast<int>(col_ptr[j+1] - col_ptr[j]);
            HashTable colj_row_pos(col_nnz * hash_size_multiplier, extt_[tid].hash_key, extt_[tid].hash_value);
            for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
                colj_row_pos.set(row_idx[ji], ji);
            }
            for (long ji = col_ptr[j]; ji < ext_->diag_ptr[j]; ++ji) {
                int k = row_idx[ji];
                for (long ki = ext_->diag_ptr[k] + 1; ki < col_ptr[k+1]; ++ki) {
                    long aij_pos = colj_row_pos.get(row_idx[ki]);
                    if (aij_pos > 0) {
                        a[aij_pos] -= a[ki] * a[ji];
                    }
                }
            }
            for (long ji = ext_->diag_ptr[j] + 1; ji < col_ptr[j+1]; ++ji) {
                a[ji] /= a[ext_->diag_ptr[j]];
            }

            // generate task
            for (long ii = ext_->csr.diag_ptr[j] + 1; ii < ext_->csr.row_ptr[j+1]; ++ii) {
                int k = ext_->csr.col_idx[ii];
                int dep_rem = ext_->dep_cnt[k].fetch_sub(1, std::memory_order_relaxed);
                if (dep_rem == 1) {
                    int new_task = ext_->task_tail.fetch_add(1, std::memory_order_relaxed);
                    ext_->task_queue[new_task].store(k, std::memory_order_relaxed);
                }
            }
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

