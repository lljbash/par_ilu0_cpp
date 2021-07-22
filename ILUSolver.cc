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

template <class T>
static void destroy_object(T* ptr) {
    if (ptr != nullptr) {
        delete ptr;
        ptr = nullptr;
    }
}

template <class T>
static void destroy_array(T* arr) {
    if (arr != nullptr) {
        delete[] arr;
        arr = nullptr;
    }
}

struct CSRMatrix {
    long* row_ptr = nullptr;
    int* col_idx = nullptr;
    long* a_map = nullptr;
    double* a = nullptr;
    long* diag_ptr = nullptr;

    void create(int n, long nnz) {
        row_ptr = new long[n+1](); // initialized to zero
        col_idx = new int[nnz];
        a_map = new long[nnz];
        a = new double[nnz];
        diag_ptr = new long[n];
    }
    void destroy() {
        destroy_array(row_ptr);
        destroy_array(col_idx);
        destroy_array(a_map);
        destroy_array(a);
        destroy_array(diag_ptr);
    }
};

struct SubtreePartition {
    int* part_ptr = nullptr;
    int* partitions = nullptr;

    void create(int nthread, int n) {
        part_ptr = new int[nthread];
        partitions = new int[n];
    }
    void destroy() {
        destroy_array(part_ptr);
        destroy_array(partitions);
    }
};

struct ILUSolver::Ext {
    long* csc_diag_ptr = nullptr;
    std::atomic_bool* task_done = nullptr;
    bool transpose_fact;
    bool paralleled_subs;
    static constexpr long transpose_min_nnz = 1000000;
    //static constexpr long transpose_min_nnz = 1;

    CSRMatrix csr;

    SubtreePartition subpart_ucol;
    SubtreePartition subpart_urow;
    SubtreePartition subpart_lrow;

    ~Ext() {
        destroy_array(csc_diag_ptr);
        destroy_array(task_done);
        csr.destroy();
        subpart_ucol.destroy();
        subpart_urow.destroy();
        subpart_lrow.destroy();
    }
};

struct ILUSolver::ThreadLocalExt {
    double* col_modification = nullptr;

    ~ThreadLocalExt() {
        destroy_array(col_modification);
    }
};

class naive_load_balancer {
public:
    using weight_t = int;
    weight_t sequential_cost(int vertex) const {
        return 1;
    }
    weight_t pipeline_latency(int vertex) const {  // 假设j依赖i，则parallel_latency(j)指的是在任务队列中，从i的完成到j的完成之间的时间差
        return 1;
    }
    /*
    bool is_balanced(int nproc, int nsubtree, weight_t max_weight, weight_t sum_weight) const {
        return max_weight * nproc < sum_weight;
    }
    bool fallback(int nproc, int nsubtree) const { // 子树数量实在太少时回退
        return nsubtree < nproc;
    }
    weight_t estimate_total_cost(weight_t subtree_part, weight_t queue_part) const {
        return subtree_part + queue_part + queue_part / 16;
    }*/
};

static int64_t estimate_cost(int n, const long* vbegin, const long* vend, const int* vtx) {
    int64_t est = 0;
    for (int i = 0; i < n; ++i) {
        est += vend[i] - vbegin[i];
    }
    return est;
}

void             
ILUSolver::SetupMatrix() {
    // HERE, you could setup the reasonable stuctures of L and U as you want
    omp_set_dynamic(0);
    omp_set_num_threads(threads_);

    int n = aMatrix_.GetSize();
    long* col_ptr = aMatrix_.GetColumnPointer();
    int* row_idx = aMatrix_.GetRowIndex();
    long nnz = aMatrix_.GetNonZeros();
    destroy_object(ext_);
    ext_ = new Ext;
    ext_->csc_diag_ptr = new long[n];
    ext_->task_done = new std::atomic_bool[n];

    // get diag_ptr
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            if (row_idx[ji] >= j) {
                ext_->csc_diag_ptr[j] = ji;
                break;
            }
        }
    }

    // CSC -> CSR
    auto csc2csr = [&]() {
        ext_->csr.create(n, nnz);
        for (long ji = 0; ji < nnz; ++ji) {
            ++ext_->csr.row_ptr[row_idx[ji] + 1];
        }
        for (int i = 0; i < n; ++i) {
            ext_->csr.row_ptr[i+1] += ext_->csr.row_ptr[i];
        }
        int* nnz_cnt = new int[n](); ON_SCOPE_EXIT { delete[] nnz_cnt; };
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
    };

    if (threads_ == 1 || nnz < ext_->transpose_min_nnz) {
        ext_->transpose_fact = false;
        ext_->paralleled_subs = false;
    }
    else {
        csc2csr();
        auto rl_est = estimate_cost(n, col_ptr, ext_->csc_diag_ptr, row_idx);
        auto ul_est = estimate_cost(n, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx);
        printf("U: %ld\nL: %ld\n", rl_est, ul_est);
        ext_->transpose_fact = rl_est > ul_est;
        //ext_->transpose_fact = false;
        ext_->paralleled_subs = true;
    }
    puts(ext_->transpose_fact ? "up-looking" : "right-looking");
    puts(ext_->paralleled_subs ? "par-subs" : "seq-subs");

    /* 0, n, 1 or n-1, -1, -1 */
    if (!ext_->transpose_fact) {
        ext_->subpart_ucol.create(threads_, n);
        partition_subtree(threads_, 0, n, 1, col_ptr, ext_->csc_diag_ptr, row_idx, ext_->subpart_ucol.part_ptr, ext_->subpart_ucol.partitions, naive_load_balancer());
        if (ext_->paralleled_subs) {
            ext_->subpart_lrow.create(threads_, n);
            partition_subtree(threads_, 0, n, 1, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx, ext_->subpart_lrow.part_ptr, ext_->subpart_lrow.partitions, naive_load_balancer());
            ext_->subpart_urow.create(threads_, n);
            partition_subtree(threads_, n-1, -1, -1, {ext_->csr.diag_ptr, 1}, ext_->csr.row_ptr + 1, ext_->csr.col_idx, ext_->subpart_urow.part_ptr, ext_->subpart_urow.partitions, naive_load_balancer());
        }
    }
    else {
        ext_->subpart_lrow.create(threads_, n);
        partition_subtree(threads_, 0, n, 1, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx, ext_->subpart_lrow.part_ptr, ext_->subpart_lrow.partitions, naive_load_balancer());
        if (ext_->paralleled_subs) {
            ext_->subpart_urow.create(threads_, n);
            partition_subtree(threads_, n-1, -1, -1, {ext_->csr.diag_ptr, 1}, ext_->csr.row_ptr + 1, ext_->csr.col_idx, ext_->subpart_urow.part_ptr, ext_->subpart_urow.partitions, naive_load_balancer());
        }
    }

    destroy_array(extt_);
    extt_ = new ThreadLocalExt[threads_];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        extt_[tid].col_modification = new double[n];
    }

    if (!ext_->transpose_fact) {
        iluMatrix_ = aMatrix_;
    }
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
    int n = aMatrix_.GetSize();
    long nnz = aMatrix_.GetNonZeros();
    double* orig = aMatrix_.GetValue();
    long* col_ptr;
    long* diag_ptr;
    int* row_idx;
    double* a;
    SubtreePartition* subpart;
    decltype(left_looking_col<Skip, NonTranspose>)* llcs;
    decltype(left_looking_col<Wait, NonTranspose>)* llcw;
    if (!ext_->transpose_fact) {
        col_ptr = iluMatrix_.GetColumnPointer();
        diag_ptr = ext_->csc_diag_ptr;
        row_idx = iluMatrix_.GetRowIndex();
        a = iluMatrix_.GetValue();
#pragma omp parallel for schedule(static, 2048)
        for (int ii = 0; ii < nnz; ++ii) {
            a[ii] = orig[ii];
        }
        subpart = &ext_->subpart_ucol;
        llcs = left_looking_col<Skip, NonTranspose>;
        llcw = left_looking_col<Wait, NonTranspose>;
    }
    else {
        col_ptr = ext_->csr.row_ptr;
        diag_ptr = ext_->csr.diag_ptr;
        row_idx = ext_->csr.col_idx;
        a = ext_->csr.a;
#pragma omp parallel for schedule(static, 2048)
        for (int ii = 0; ii < nnz; ++ii) {
            a[ii] = orig[ext_->csr.a_map[ii]];
        }
        subpart = &ext_->subpart_lrow;
        llcs = left_looking_col<Skip, Transpose>;
        llcw = left_looking_col<Wait, Transpose>;
    }
    std::atomic_int task_head{subpart->part_ptr[threads_]};
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
        for(int k = subpart->part_ptr[tid]; k < subpart->part_ptr[tid + 1]; ++k) {
            int j = subpart->partitions[k];
            llcs(j, col_ptr, row_idx, a, diag_ptr,
                extt_[tid].col_modification, ext_->task_done);
        }
        /* queue part */
        while (true) {
            // get task
            int task_id = task_head.fetch_add(1, std::memory_order_relaxed);
            if (task_id >= n) {
                break;
            }
            int j = subpart->partitions[task_id];

            // execute task
            llcw(j, col_ptr, row_idx, a, diag_ptr,
                extt_[tid].col_modification, ext_->task_done);
        }
    }

    if (ext_->paralleled_subs && !ext_->transpose_fact) {
#pragma omp parallel for schedule(static, 2048)
        for (int ii = 0; ii < nnz; ++ii) {
            ext_->csr.a[ii] = a[ext_->csr.a_map[ii]];
        }
    }

    return true;
}

template <typename W>
double substitute_row_L(int i, const long* row_ptr, const long* diag_ptr, const int* col_idx,
                        const double* lvalue, const double* x,
                        std::atomic_bool* task_done) {
    double xi = x[i];
    for (long ii = row_ptr[i]; ii < diag_ptr[i]; ++ii) {
        int j = col_idx[ii];
        W::busy_waiting(&task_done[j]);
        xi -= x[j] * lvalue[ii];
    }
    return xi;
}

template <typename W>
double substitute_row_U(int i, const long* row_ptr, const long* diag_ptr, const int* col_idx,
                        const double* uvalue, const double* x,
                        std::atomic_bool* task_done) {
    double xi = x[i];
    for (long ii = row_ptr[i+1] - 1; ii > diag_ptr[i]; --ii) {
        int j = col_idx[ii];
        W::busy_waiting(&task_done[j]);
        xi -= x[j] * uvalue[ii];
    }
    xi /= uvalue[diag_ptr[i]];
    return xi;
}

void
ILUSolver::Substitute() {
    // HERE, use the L and U calculated by ILUSolver::Factorize to solve the triangle systems 
    // to calculate the x
    int n = aMatrix_.GetSize();
    x_ = new double[n];
    std::memcpy(x_, b_, sizeof(double[n]));
    if (!ext_->paralleled_subs) {
        long* col_ptr = iluMatrix_.GetColumnPointer();
        int* row_idx = iluMatrix_.GetRowIndex();
        double* a = iluMatrix_.GetValue();
        for (int j = 0; j < n; ++j) {
            for (long ji = ext_->csc_diag_ptr[j] + 1; ji < col_ptr[j+1]; ++ji) {
                int i = row_idx[ji];
                x_[i] -= x_[j] * a[ji];
            }
        }
        for (int j = n - 1; j >= 0; --j) {
            x_[j] /= a[ext_->csc_diag_ptr[j]];
            for (long ji = col_ptr[j]; ji < ext_->csc_diag_ptr[j]; ++ji) {
                int i = row_idx[ji];
                x_[i] -= x_[j] * a[ji];
            }
        }
    }
    else {
#define SUBS_ROW(D, W, I, X) substitute_row_##D<W>(I, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx, ext_->csr.a, X, ext_->task_done)
        // L
        std::atomic_int task_head{ext_->subpart_lrow.part_ptr[threads_]};
#pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            ext_->task_done[j].store(false, std::memory_order_relaxed);
        }
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            /* independent part */
            //printf("%d: %d %d\n", tid, ext_->l.part_ptr[tid], ext_->l.part_ptr[tid+1]);
            for(int k = ext_->subpart_lrow.part_ptr[tid]; k < ext_->subpart_lrow.part_ptr[tid + 1]; ++k) {
                int i = ext_->subpart_lrow.partitions[k];
                x_[i] = SUBS_ROW(L, Skip, i, x_);
                ext_->task_done[i].store(true, std::memory_order_release);
            }
            /* queue part */
            while (true) {
                int task_id = task_head.fetch_add(1, std::memory_order_relaxed);
                if (task_id >= n) {
                    break;
                }
                int i = ext_->subpart_lrow.partitions[task_id];
                x_[i] = SUBS_ROW(L, Wait, i, x_);
                ext_->task_done[i].store(true, std::memory_order_release);
            }
        }
        // U
        task_head = ext_->subpart_urow.part_ptr[threads_];
#pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            ext_->task_done[j].store(false, std::memory_order_relaxed);
        }
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            //printf("%d: %d %d\n", tid, ext_->u.part_ptr[tid], ext_->u.part_ptr[tid+1]);
            /* independent part */
            for(int k = ext_->subpart_urow.part_ptr[tid]; k < ext_->subpart_urow.part_ptr[tid + 1]; ++k) {
                int i = ext_->subpart_urow.partitions[k];
                x_[i] = SUBS_ROW(U, Skip, i, x_);
                ext_->task_done[i].store(true, std::memory_order_release);
            }
            /* queue part */
            while (true) {
                int task_id = task_head.fetch_add(1, std::memory_order_relaxed);
                if (task_id >= n) {
                    break;
                }
                int i = ext_->subpart_urow.partitions[task_id];
                x_[i] = SUBS_ROW(U, Wait, i, x_);
                ext_->task_done[i].store(true, std::memory_order_release);
            }
        }
#undef SUBS_ROW
    }
}

void    
ILUSolver::CollectLUMatrix() {
    // put L and U together into iluMatrix_ 
    // as the diag of L is 1, set the diag of iluMatrix_ with u 
    // iluMatrix_ should have the same size and patterns with aMatrix_
    if (ext_->transpose_fact) {
        iluMatrix_ = aMatrix_;
        int n = iluMatrix_.GetSize();
        long* col_ptr = iluMatrix_.GetColumnPointer();
        int* row_idx = iluMatrix_.GetRowIndex();
        double* a = iluMatrix_.GetValue();
        int *nnz_cnt = new int[n](); // initialized to zero
        ON_SCOPE_EXIT { delete[] nnz_cnt; };
        for (int j = 0; j < n; ++j) {
#pragma omp parallel for
            for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
                int i = row_idx[ji];
                long ii = ext_->csr.row_ptr[i] + nnz_cnt[i];
                a[ji] = ext_->csr.a[ii];
                ++nnz_cnt[i];
            }
        }
    }
}    

