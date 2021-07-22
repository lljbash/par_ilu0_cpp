#include "ILUSolver.h"
#include <type_traits>
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

namespace param {

constexpr long transpose_min_nnz = 1000000;
constexpr int granu_min_n = 10000;
constexpr double fact_fixed_subtree_weight = 1;
constexpr double fact_fixed_queue_weight = 2;
constexpr int fact_granu = 4;
constexpr double subs_fixed_subtree_weight = 1;
constexpr double subs_fixed_queue_weight = 2;
constexpr int subs_granu = 6;

}

class FixedLoadBalancer {
public:
    using weight_t = double;
    weight_t weight_in_subtree(int) const { return weight_in_subtree_; }
    weight_t weight_in_queue(int) const { return weight_in_queue_; }
    int queue_granularity() const { return queue_granularity_; }

    weight_t weight_in_subtree_;
    weight_t weight_in_queue_;
    int queue_granularity_;
};

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
    int* task_queue = nullptr;
    int ntasks;

    void create(int nthread, int n) {
        part_ptr = new int[nthread];
        partitions = new int[n];
    }
    void destroy() {
        destroy_array(part_ptr);
        destroy_array(partitions);
        destroy_array(task_queue);
    }
};

struct ILUSolver::Ext {
    long* csc_diag_ptr = nullptr;
    std::atomic_bool* task_done = nullptr;
    bool transpose_fact;
    bool paralleled_subs;
    bool packed;

    CSRMatrix csr;

    SubtreePartition subpart_fact;
    SubtreePartition subpart_subs_l;
    SubtreePartition subpart_subs_u;

    ~Ext() {
        destroy_array(csc_diag_ptr);
        destroy_array(task_done);
        csr.destroy();
        subpart_fact.destroy();
        subpart_subs_l.destroy();
        subpart_subs_u.destroy();
    }
};

struct ILUSolver::ThreadLocalExt {
    double* col_modification = nullptr;
    long* interupt = nullptr;

    ~ThreadLocalExt() {
        destroy_array(col_modification);
        destroy_array(interupt);
    }
};

ILUSolver::~ILUSolver() {
    destroy_array(x_);
    destroy_array(b_);
    destroy_object(ext_);
    destroy_array(extt_);
}

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

    if (threads_ == 1 || nnz < param::transpose_min_nnz) {
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
    ext_->packed = threads_ > 1 && n >= param::granu_min_n;
    puts(ext_->transpose_fact ? "up-looking" : "right-looking");
    puts(ext_->paralleled_subs ? "par-subs" : "seq-subs");
    puts(ext_->packed ? "packed tasks" : "single task");

    FixedLoadBalancer lb_fact{param::fact_fixed_subtree_weight, param::fact_fixed_queue_weight, 1};
    FixedLoadBalancer lb_subs{param::subs_fixed_subtree_weight, param::subs_fixed_queue_weight, 1};
    if (ext_->packed) {
        lb_fact.queue_granularity_ = param::fact_granu;
        lb_subs.queue_granularity_ = param::subs_granu;
    }

    auto get_subpart = [this, n](int p2, int p3, int p4, ConstBiasArray<long> p5, const long* p6, const int* p7, SubtreePartition& out, const FixedLoadBalancer& lb) {
        out.create(threads_, n);
        out.ntasks = tree_schedule(threads_, p2, p3, p4, p5, p6, p7, out.part_ptr, out.partitions, out.task_queue, lb);
    };
    if (!ext_->transpose_fact) {
        get_subpart(0, n, 1, col_ptr, ext_->csc_diag_ptr, row_idx, ext_->subpart_fact, lb_fact);
    }
    else {
        get_subpart(0, n, 1, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx, ext_->subpart_fact, lb_fact);
    }
    if (ext_->paralleled_subs) {
        get_subpart(0, n, 1, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx, ext_->subpart_subs_l, lb_subs);
        get_subpart(n-1, -1, -1, {ext_->csr.diag_ptr, 1}, ext_->csr.row_ptr + 1, ext_->csr.col_idx, ext_->subpart_subs_u, lb_subs);
    }

    destroy_array(extt_);
    extt_ = new ThreadLocalExt[threads_];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        extt_[tid].col_modification = new double[n * lb_fact.queue_granularity()];
        extt_[tid].interupt = new long[std::max(lb_fact.queue_granularity(), lb_subs.queue_granularity())];
    }

    if (!ext_->transpose_fact) {
        iluMatrix_ = aMatrix_;
    }
}

template <class JobW, class JobI,
          typename std::enable_if<std::is_same<JobI, std::nullptr_t>::value, int>::type = 0>
static void subpart_parallel_run_queue_exec(int task_id, const SubtreePartition& subpart,
                                            const JobW& fw, const JobI&, int tid) {
    fw(subpart.partitions[task_id], tid, 0);
}

template <class JobW, class JobI,
          typename std::enable_if<!std::is_same<JobI, std::nullptr_t>::value, int>::type = 0>
static void subpart_parallel_run_queue_exec(int task_id, const SubtreePartition& subpart,
                                            const JobW& fw, const JobI& fi, int tid) {
    for (int k = subpart.task_queue[task_id]; k < subpart.task_queue[task_id+1]; ++k) {
        fi(subpart.partitions[k], tid, k - subpart.task_queue[task_id]);
    }
    for (int k = subpart.task_queue[task_id]; k < subpart.task_queue[task_id+1]; ++k) {
        fw(subpart.partitions[k], tid, k - subpart.task_queue[task_id]);
    }
}

// tasks are packed only if fi != nullptr
template <class JobS, class JobW, class JobI>
static void subpart_parallel_run(int threads, int n, const SubtreePartition& subpart,
                                 const JobS& fs, const JobW& fw, const JobI& fi,
                                 std::atomic_bool* task_done) {
    std::atomic_int task_head;
    int task_tail;
    if /* constexpr */ (std::is_same<JobI, std::nullptr_t>::value) {
        task_head.store(subpart.part_ptr[threads], std::memory_order_relaxed);
        task_tail = n;
    }
    else {
        task_head.store(0, std::memory_order_relaxed);
        task_tail = subpart.ntasks;
    }
#pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            task_done[j].store(false, std::memory_order_relaxed);
        }
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        /* independent part */
        for(int k = subpart.part_ptr[tid]; k < subpart.part_ptr[tid + 1]; ++k) {
            fs(subpart.partitions[k], tid);
        }
        /* queue part */
        while (true) {
            // get task
            int task_id = task_head.fetch_add(1, std::memory_order_relaxed);
            if (task_id >= task_tail) {
                break;
            }

            // execute task
            subpart_parallel_run_queue_exec(task_id, subpart, fw, fi, tid);
        }
    }
}

struct Skip {
    static bool busy_waiting(std::atomic_bool*) { return true; }
};
struct Wait {
    static bool busy_waiting(std::atomic_bool* cond) {
        while (!cond->load(std::memory_order_acquire)); // busy waiting
        return true;
    }
};
struct Interupt {
    static int busy_waiting(std::atomic_bool* cond) {
        return cond->load(std::memory_order_acquire);
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



template <typename W, typename T, bool R>
static long left_looking_col(int j, const long* col_ptr, const int* row_idx, double* a, const long* diag_ptr,
                             double* col_modification, std::atomic_bool* task_done,
                             long interupt = 0) { // -1: new; -2: finished; >= 0: interupted
    long begin;
    if /*constexpr*/(R) {
        if (interupt == -2) {
            return -2;
        }
        begin = interupt;
    } else {
        for (long ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            col_modification[row_idx[ji]] = 0;
        }
        begin = col_ptr[j];
    }
    long diag = diag_ptr[j];
    for (long ji = begin; ji < diag; ++ji) {
        int k = row_idx[ji];
        if (!W::busy_waiting(&task_done[k])) {
            return ji;
        }
        a[ji] = T::scale_U(a[ji] + col_modification[k], a[diag_ptr[k]]);
        for (long ki = diag_ptr[k] + 1; ki < col_ptr[k+1]; ++ki) {
            col_modification[row_idx[ki]] -= a[ki] * a[ji];
        }
    }
    a[diag] += col_modification[row_idx[diag]];
    double ajj = a[diag];
    for (long ji = diag + 1; ji < col_ptr[j+1]; ++ji) {
        a[ji] = T::scale_L(a[ji] + col_modification[row_idx[ji]], ajj);
    }

    task_done[j].store(true, std::memory_order_release);
    return -2;
}

template<typename Trans, typename Ext>
static void fact_parallel_run(int threads, int n, const long* col_ptr, const long* diag_ptr, const int* row_idx, double* a, Ext* ext, std::atomic_bool* task_done, SubtreePartition& subpart, bool packed) {
    if (packed) {
        auto fs = [col_ptr, row_idx, a, diag_ptr, ext, task_done](int j, int tid) {
            left_looking_col<Skip, Trans, false>(j, col_ptr, row_idx, a, diag_ptr, ext[tid].col_modification, task_done);
        };
        auto fw = [col_ptr, row_idx, a, diag_ptr, ext, task_done, n](int j, int tid, int k) {
            left_looking_col<Wait, Trans, true>(j, col_ptr, row_idx, a, diag_ptr, ext[tid].col_modification + k * n, task_done, ext[tid].interupt[k]);
        };
        auto fi = [col_ptr, row_idx, a, diag_ptr, ext, task_done, n](int j, int tid, int k) {
            ext[tid].interupt[k] = left_looking_col<Interupt, Trans, false>(j, col_ptr, row_idx, a, diag_ptr, ext[tid].col_modification + k * n, task_done);
        };
        subpart_parallel_run(threads, n, subpart, fs, fw, fi, task_done);
    }
    else {
        auto fs = [col_ptr, row_idx, a, diag_ptr, ext, task_done](int j, int tid) {
            left_looking_col<Skip, Trans, false>(j, col_ptr, row_idx, a, diag_ptr, ext[tid].col_modification, task_done);
        };
        auto fw = [col_ptr, row_idx, a, diag_ptr, ext, task_done](int j, int tid, int) {
            left_looking_col<Wait, Trans, false>(j, col_ptr, row_idx, a, diag_ptr, ext[tid].col_modification, task_done);
        };
        subpart_parallel_run(threads, n, subpart, fs, fw, nullptr, task_done);
    }
}

bool             
ILUSolver::Factorize() {
    // HERE, do triangle decomposition 
    // to calculate the values in L and U
    int n = aMatrix_.GetSize();
    long nnz = aMatrix_.GetNonZeros();
    double* orig = aMatrix_.GetValue();
    const long* col_ptr;
    const long* diag_ptr;
    const int* row_idx;
    double* a;
    if (!ext_->transpose_fact) {
        col_ptr = iluMatrix_.GetColumnPointer();
        diag_ptr = ext_->csc_diag_ptr;
        row_idx = iluMatrix_.GetRowIndex();
        a = iluMatrix_.GetValue();
#pragma omp parallel for schedule(static, 2048)
        for (int ii = 0; ii < nnz; ++ii) {
            a[ii] = orig[ii];
        }
        fact_parallel_run<NonTranspose, ThreadLocalExt>(threads_, n, col_ptr, diag_ptr, row_idx, a, extt_, ext_->task_done, ext_->subpart_fact, ext_->packed);
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
        fact_parallel_run<Transpose, ThreadLocalExt>(threads_, n, col_ptr, diag_ptr, row_idx, a, extt_, ext_->task_done, ext_->subpart_fact, ext_->packed);
    }

    if (ext_->paralleled_subs && !ext_->transpose_fact) {
#pragma omp parallel for schedule(static, 2048)
        for (int ii = 0; ii < nnz; ++ii) {
            ext_->csr.a[ii] = a[ext_->csr.a_map[ii]];
        }
    }

    return true;
}

template <typename W, bool R>
long substitute_row_L(int i, const long* row_ptr, const long* diag_ptr, const int* col_idx,
                        const double* lvalue, double* x, std::atomic_bool* task_done,
                        long interupt = 0) {
    long begin;
    if /*constexpr*/(R) {
        if (interupt == -2) {
            return -2;
        }
        begin = interupt;
    } else {
        begin = row_ptr[i];
    }
    double xi = x[i];
    for (long ii = begin, diag = diag_ptr[i]; ii < diag; ++ii) {
        int j = col_idx[ii];
        if (!W::busy_waiting(&task_done[j])) {
            x[i] = xi;
            return ii;
        }
        xi -= x[j] * lvalue[ii];
    }
    x[i] = xi;
    task_done[i].store(true, std::memory_order_release);
    return -2;
}

template <typename W, bool R>
long substitute_row_U(int i, const long* row_ptr, const long* diag_ptr, const int* col_idx,
                        const double* uvalue, double* x, std::atomic_bool* task_done,
                        long interupt = 0) {
    long begin;
    if /*constexpr*/(R) {
        if (interupt == -2) {
            return -2;
        }
        begin = interupt;
    } else {
        begin = row_ptr[i + 1] - 1;
    }
    double xi = x[i];
    long diag = diag_ptr[i];
    for (long ii = begin; ii > diag; --ii) { /* first complete first update */
        int j = col_idx[ii];
        if (!W::busy_waiting(&task_done[j])) {
            x[i] = xi;
            return ii;
        }
        xi -= x[j] * uvalue[ii];
    }
    x[i] = xi / uvalue[diag];
    task_done[i].store(true, std::memory_order_release);
    return -2;
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
    else if (ext_->packed) {
#define SUBS_ROW(D, W) substitute_row_##D<W,false>(i, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx, ext_->csr.a, x_, ext_->task_done)
#define SUBS_ROWR(D, W, I) substitute_row_##D<W,true>(i, ext_->csr.row_ptr, ext_->csr.diag_ptr, ext_->csr.col_idx, ext_->csr.a, x_, ext_->task_done, I)
        auto lfs = [this](int i, int) { SUBS_ROW(L, Skip); };
        auto lfw = [this](int i, int tid, int k) { SUBS_ROWR(L, Wait, extt_[tid].interupt[k]); };
        auto lfi = [this](int i, int tid, int k) { extt_[tid].interupt[k] = SUBS_ROW(L, Interupt); };
        auto ufs = [this](int i, int) { SUBS_ROW(U, Skip); };
        auto ufw = [this](int i, int tid, int k) { SUBS_ROWR(U, Wait, extt_[tid].interupt[k]); };
        auto ufi = [this](int i, int tid, int k) { extt_[tid].interupt[k] = SUBS_ROW(U, Interupt); };
        subpart_parallel_run(threads_, n, ext_->subpart_subs_l, lfs, lfw, lfi, ext_->task_done);
        subpart_parallel_run(threads_, n, ext_->subpart_subs_u, ufs, ufw, ufi, ext_->task_done);
    }
    else {
        auto lfs = [this](int i, int) { SUBS_ROW(L, Skip); };
        auto lfw = [this](int i, int, int) { SUBS_ROW(L, Wait); };
        auto ufs = [this](int i, int) { SUBS_ROW(U, Skip); };
        auto ufw = [this](int i, int, int) { SUBS_ROW(U, Wait); };
        subpart_parallel_run(threads_, n, ext_->subpart_subs_l, lfs, lfw, nullptr, ext_->task_done);
        subpart_parallel_run(threads_, n, ext_->subpart_subs_u, ufs, ufw, nullptr, ext_->task_done);
    }
#undef SUBS_ROW
#undef SUBS_ROWR
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

