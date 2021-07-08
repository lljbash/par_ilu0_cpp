#include "ILUSolver.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <omp.h>
#include "scope_guard.h"
#include "sub_tree.h"

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
    int* task_queue = nullptr;
    //int* part_ptr = nullptr;
    //int* partitions = nullptr;
    int* task_queue_L = nullptr;
    int* level_end_L = nullptr;
    int* task_queue_U = nullptr;
    int* level_end_U = nullptr;
    std::atomic_bool* task_done;

    struct CSR {
        long* row_ptr = nullptr;
        int* col_idx = nullptr;
        long* a_map = nullptr;
        double* a = nullptr;
        long* diag_ptr = nullptr;
    } csr;
};

struct ILUSolver::ThreadLocalExt {
    double* col_modification = nullptr;
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
    ext_->task_queue = new int[n];
    //ext_->partitions = new int[n];
    //ext_->part_ptr = new int[threads_ + 2];
    ext_->task_queue_L = new int[n];
    ext_->level_end_L = new int[n];
    ext_->task_queue_U = new int[n];
    ext_->level_end_U = new int[n];
    ext_->task_done = new std::atomic_bool[n];
    ext_->csr.row_ptr = new long[n+1](); // initialized to zero
    ext_->csr.col_idx = new int[nnz];
    ext_->csr.a_map = new long[nnz];
    ext_->csr.a = new double[nnz];
    ext_->csr.diag_ptr = new long[n];
    int* nnz_cnt = new int[n](); // initialized to zero
    ON_SCOPE_EXIT { delete[] nnz_cnt; };
    int* dep_cnt = new int[n];
    ON_SCOPE_EXIT { delete[] dep_cnt; };

    // get diag_ptr
#pragma omp parallel
    for (int j = 0; j < n; ++j) {
        for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            if (row_idx[ji] >= j) {
                ext_->diag_ptr[j] = ji;
                break;
            }
        }
        dep_cnt[j] = static_cast<int>(ext_->diag_ptr[j] - col_ptr[j]);
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
            long ii = ext_->csr.row_ptr[i] + nnz_cnt[i];
            ext_->csr.col_idx[ii] = j;
            ext_->csr.a_map[ii] = ji;
            ++nnz_cnt[i];
            if (i == j) {
                ext_->csr.diag_ptr[i] = ii;
            }
        }
    }

    // get topo task queue
    std::atomic_int task_tail{0};
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        if (dep_cnt[j] == 0) {
            int new_task = task_tail.fetch_add(1, std::memory_order_relaxed);
            ext_->task_queue[new_task] = j;
        }
    }
    for (int t = 0; t < n; ++t) {
        int j = ext_->task_queue[t];
        for (long ii = ext_->csr.diag_ptr[j] + 1; ii < ext_->csr.row_ptr[j+1]; ++ii) {
            int k = ext_->csr.col_idx[ii];
            int dep_rem = dep_cnt[k]--;
            if (dep_rem == 1) {
                int new_task = task_tail.fetch_add(1, std::memory_order_relaxed);
                ext_->task_queue[new_task] = k;
            }
        }
    }
    //partition_subtree(n, threads_, col_ptr, ext_->diag_ptr, row_idx, ext_->part_ptr, ext_->partitions);
    task_tail = 0;
    for (int i = 0; i < n; ++i) {
        dep_cnt[i] = static_cast<int>(ext_->csr.diag_ptr[i] - ext_->csr.row_ptr[i]);
        if (dep_cnt[i] == 0) {
            ext_->task_queue_L[task_tail] = i;
            ++task_tail;
        }
    }
    ext_->level_end_L[0] = task_tail;
    for (int l = 1, t = 0; task_tail < n; ++l) {
        int te = task_tail;
        for (; t < te; ++t) {
            int i = ext_->task_queue_L[t];
            for (long ji = ext_->diag_ptr[i] + 1; ji < col_ptr[i+1]; ++ji) {
                int k = row_idx[ji];
                if (--dep_cnt[k] == 0) {
                    ext_->task_queue_L[task_tail] = k;
                    ++task_tail;
                }
            }
        }
        ext_->level_end_L[l] = task_tail;
    }
    task_tail = 0;
    for (int i = 0; i < n; ++i) {
        dep_cnt[i] = static_cast<int>(ext_->csr.row_ptr[i+1] - ext_->csr.diag_ptr[i] - 1);
        if (dep_cnt[i] == 0) {
            ext_->task_queue_U[task_tail] = i;
            ++task_tail;
        }
    }
    ext_->level_end_U[0] = task_tail;
    for (int l = 1, t = 0; task_tail < n; ++l) {
        int te = task_tail;
        for (; t < te; ++t) {
            int i = ext_->task_queue_U[t];
            for (long ji = col_ptr[i]; ji < ext_->diag_ptr[i]; ++ji) {
                int k = row_idx[ji];
                if (--dep_cnt[k] == 0) {
                    ext_->task_queue_U[task_tail] = k;
                    ++task_tail;
                }
            }
        }
        ext_->level_end_U[l] = task_tail;
    }

    extt_ = new ThreadLocalExt[threads_];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        extt_[tid].col_modification = new double[n];
    }

    iluMatrix_ = aMatrix_;
}

template <bool B>
static typename std::enable_if<B, void>::type
busy_waiting_if(std::atomic_bool* cond) {
    while (!cond->load(std::memory_order_acquire)); // busy waiting
};

template <bool B>
static typename std::enable_if<!B, void>::type
busy_waiting_if(std::atomic_bool*) {};

template <bool B>
static void left_looking_col(int j, long* col_ptr, int* row_idx, double* a, long* diag_ptr,
                             double* col_modification, std::atomic_bool* task_done) {
    for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
        col_modification[row_idx[ji]] = 0;
    }
    for (long ji = col_ptr[j]; ji < diag_ptr[j]; ++ji) {
        int k = row_idx[ji];
        busy_waiting_if<B>(&task_done[k]);
        a[ji] += col_modification[k];
        for (long ki = diag_ptr[k] + 1; ki < col_ptr[k+1]; ++ki) {
            col_modification[row_idx[ki]] -= a[ki] * a[ji];
        }
    }
    a[diag_ptr[j]] += col_modification[row_idx[diag_ptr[j]]];
    double ajj = a[diag_ptr[j]];
    for (long ji = diag_ptr[j] + 1; ji < col_ptr[j+1]; ++ji) {
        a[ji] = (a[ji] + col_modification[row_idx[ji]]) / ajj;
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
    std::atomic_int task_head{0};
    //std::atomic_int task_head{ext_->part_ptr[threads_]};
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
        //for(int k = ext_->part_ptr[tid]; k < ext_->part_ptr[tid + 1]; ++k) {
            //int j = ext_->partitions[k];

            //// execute task
            //left_looking_col<false>(j, col_ptr, row_idx, a, ext_->diag_ptr,
                                    //extt_[tid].col_modification, ext_->task_done);
        //}
        /* queue part */
        while (true) {
            // get task
            int task_id = task_head.fetch_add(1, std::memory_order_relaxed);
            if (task_id >= n) {
                break;
            }
            int j = ext_->task_queue[task_id];
            //int j = ext_->partitions[task_id];

            // execute task
            left_looking_col<true>(j, col_ptr, row_idx, a, ext_->diag_ptr,
                                   extt_[tid].col_modification, ext_->task_done);
        }
    }

    // CSC -> CSR
    long nnz = iluMatrix_.GetNonZeros();
#pragma omp parallel for
    for (int ii = 0; ii < nnz; ++ii) {
        ext_->csr.a[ii] = a[ext_->csr.a_map[ii]];
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

    // L
    for (int l = 0, tb = 0; ; ++l) {
        int te = ext_->level_end_L[l];
#pragma omp parallel for schedule(static, 1)
        for (int t = tb; t < te; ++t) {
            int i = ext_->task_queue_L[t];
            //printf("%d ", i);
            double xi = x_[i];
            for (long ii = ext_->csr.row_ptr[i]; ii < ext_->csr.diag_ptr[i]; ++ii) {
                int j = ext_->csr.col_idx[ii];
                xi -= x_[j] * ext_->csr.a[ii];
            }
            x_[i] = xi;
        }
        //printf("\n----------\n");
        if (te >= n) {
            break;
        }
        tb = te;
    }
    //for (int j = 0; j < n; ++j) {
        //for (long ji = ext_->diag_ptr[j] + 1; ji < col_ptr[j+1]; ++ji) {
            //int i = row_idx[ji];
            //x_[i] -= x_[j] * a[ji];
        //}
    //}

    // U
    for (int l = 0, tb = 0; ; ++l) {
        int te = ext_->level_end_U[l];
#pragma omp parallel for schedule(static, 1)
        for (int t = tb; t < te; ++t) {
            int i = ext_->task_queue_U[t];
            //printf("%d ", i);
            double xi = x_[i];
            for (long ii = ext_->csr.diag_ptr[i] + 1; ii < ext_->csr.row_ptr[i+1]; ++ii) {
                int j = ext_->csr.col_idx[ii];
                xi -= x_[j] * ext_->csr.a[ii];
            }
            xi /= ext_->csr.a[ext_->csr.diag_ptr[i]];
            x_[i] = xi;
        }
        //printf("\n==========\n");
        if (te >= n) {
            break;
        }
        tb = te;
    }
    //for (int j = n - 1; j >= 0; --j) {
        //x_[j] /= a[ext_->diag_ptr[j]];
        //for (long ji = col_ptr[j]; ji < ext_->diag_ptr[j]; ++ji) {
            //int i = row_idx[ji];
            //x_[i] -= x_[j] * a[ji];
        //}
    //}
}

void    
ILUSolver::CollectLUMatrix() {
    // put L and U together into iluMatrix_ 
    // as the diag of L is 1, set the diag of iluMatrix_ with u 
    // iluMatrix_ should have the same size and patterns with aMatrix_
}    

