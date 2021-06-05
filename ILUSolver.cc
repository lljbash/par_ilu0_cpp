#include "ILUSolver.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <omp.h>
#include "scope_guard.h"
#include "subtree.h"

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
    int* part_ptr = nullptr;
    int* partitions = nullptr;
    std::atomic_bool* task_done;

    struct CSR {
        long* row_ptr = nullptr;
        int* col_idx = nullptr;
        int* a = nullptr;
        long* diag_ptr = nullptr;
        int* nnz_cnt = nullptr;
    } csr;
};

struct ILUSolver::ThreadLocalExt {
    double* col_modification = nullptr;
};

class naive_load_balancer {
public:
    using weight_t = int;
    weight_t get_weight(int vertex) const {
        return 1;
    }
    bool is_balanced(int nproc, int nsubtree, weight_t max_weight, weight_t sum_weight) const {
        return max_weight * nproc < sum_weight;
    }
    bool fallback(int nproc, int nsubtree) const { // 子树数量实在太少时回退
        return nsubtree < nproc;
    }
    weight_t estimate_cost(weight_t subtree_part, weight_t queue_part) const {
        return subtree_part + queue_part + queue_part / 16;
    }
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
    ext_->partitions = new int[n];
    ext_->part_ptr = new int[threads_ + 1];
    ext_->task_done = new std::atomic_bool[n];
    ext_->csr.row_ptr = new long[n+1](); // initialized to zero
    ext_->csr.col_idx = new int[nnz];
    //ext_->csr.a = new int[nnz];
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

    /* 0, n, 1 or n-1, -1, -1 */
    int cost = partition_subtree(threads_, 0, n, 1, col_ptr, ext_->diag_ptr, row_idx, ext_->part_ptr, ext_->partitions, naive_load_balancer());

    extt_ = new ThreadLocalExt[threads_];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        extt_[tid].col_modification = new double[n];
    }

    iluMatrix_ = aMatrix_;
}

struct Skip {
    static void busy_waiting(std::atomic_bool* cond) {}
};
struct Wait {
    static void busy_waiting(std::atomic_bool* cond) {
        while (!cond->load(std::memory_order_acquire)); // busy waiting
    }
};

struct ScaleL {
    static double scale(double a, double diag) {
        return a / diag;
    }
};
struct ScaleU {
    static double scale(double a, double diag) {
        return a;
    }
};

template<typename L, typename U>
struct DiagnalScale {
    static double scale_L(double a, double diag) {
        return L::scale(a, diag);
    }
    static double scale_U(double a, double diag) {
        return U::scale(a, diag);
    }
};

using NonTranspose = DiagnalScale<ScaleL, ScaleU>;
using Transpose = DiagnalScale<ScaleU, ScaleL>;  // For CSR format

template <typename W, typename T>
static void left_looking_col(int j, long* col_ptr, int* row_idx, double* a, long* diag_ptr,
                             double* col_modification, std::atomic_bool* task_done) {
    for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
        col_modification[row_idx[ji]] = 0;
    }
    for (long ji = col_ptr[j]; ji < diag_ptr[j]; ++ji) {
        int k = row_idx[ji];
        W::busy_waiting(&task_done[k]);
        a[ji] = T::scale_U(a[ji] + col_modification[k], a[diag_ptr[k]]);
        for (long ki = diag_ptr[k] + 1; ki < col_ptr[k+1]; ++ki) {
            col_modification[row_idx[ki]] -= a[ki] * a[ji];
        }
    }
    a[diag_ptr[j]] += col_modification[row_idx[diag_ptr[j]]];
    double ajj = a[diag_ptr[j]];
    for (long ji = diag_ptr[j] + 1; ji < col_ptr[j+1]; ++ji) {
        a[ji] = T::scale_L(a[ji] + col_modification[row_idx[ji]], ajj);
    }

    task_done[j].store(true, std::memory_order_release);
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
    std::atomic_int task_head{ext_->part_ptr[threads_]};
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        ext_->task_done[j].store(false, std::memory_order_relaxed);
    }

//#pragma omp parallel for schedule(dynamic, 1)
    //for(int j = 0; j < n; ++j) {
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        /* independent part */
        for(int k = ext_->part_ptr[tid]; k < ext_->part_ptr[tid + 1]; ++k) {
            int j = ext_->partitions[k];
            
            // execute task
            left_looking_col<Skip, NonTranspose>(j, col_ptr, row_idx, a, ext_->diag_ptr,
                                    extt_[tid].col_modification, ext_->task_done);
        }
        /* queue part */
        while (true) {
            // get task
            int task_id = task_head.fetch_add(1, std::memory_order_relaxed);
            if (task_id >= n) {
                break;
            }
            int j = ext_->partitions[task_id];

            // execute task
            left_looking_col<Wait, NonTranspose>(j, col_ptr, row_idx, a, ext_->diag_ptr,
                                   extt_[tid].col_modification, ext_->task_done);
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

