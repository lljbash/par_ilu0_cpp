#include "ilu_solver.hpp"
#include <type_traits>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <omp.h>
#include <mkl.h>
#include "scope_guard.hpp"
#include "subtree.hpp"
#include "tictoc.hpp"

namespace lljbash {

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

constexpr int par_subs_min_nnz_per_line = 20;
constexpr int granu_min_n = 10000;
constexpr double fact_fixed_subtree_weight = 1;
constexpr double fact_fixed_queue_weight = 2;
constexpr int fact_granu = 4;
constexpr double subs_fixed_subtree_weight = 1;
constexpr double subs_fixed_queue_weight = 2;
constexpr int subs_granu = 6;
constexpr int subs_max_threads = 8;

}


class NaiveLoadBalancer {
public:
    using weight_t = int;
    weight_t weight_in_subtree(int) const { return 1; }
    weight_t weight_in_queue(int) const { return 1; }
    int queue_granularity() const { return queue_granularity_; }
    template<typename ... Args>
    NaiveLoadBalancer(int granu, Args ... /*ignored*/) : queue_granularity_(granu) {}

    int queue_granularity_;
};

class NNZLoadBalancer {
public:
    using weight_t = int;
    weight_t weight_in_subtree(int i) const { return end_[i] - begin_[i] + base_; }
    weight_t weight_in_queue(int i) const { return end_[i] - begin_[i] + base_; }
    int queue_granularity() const { return queue_granularity_; }
    NNZLoadBalancer(int granu, const weight_t *begin, const weight_t *end, weight_t base)
    : queue_granularity_(granu), begin_(begin), end_(end), base_(base) {}

    int queue_granularity_;
    const weight_t *begin_;
    const weight_t *end_;
    weight_t base_;
};

using LoadBalancer = NNZLoadBalancer;

struct SubtreePartition {
    int* part_ptr = nullptr;
    int* partitions = nullptr;
    int* task_queue = nullptr;
    int ntasks;

    void create(int nthread, int n) {
        part_ptr = new int[nthread + 1];
        partitions = new int[n];
    }
    void destroy() {
        destroy_array(part_ptr);
        destroy_array(partitions);
        destroy_array(task_queue);
    }
};

struct IluSolver::Ext {
    int* csr_diag_ptr = nullptr;
    std::atomic_bool* task_done = nullptr;
    bool paralleled_subs_queue;
    bool packed;
    int subs_threads = 1;

    SubtreePartition subpart_fact;
    SubtreePartition subpart_subs_l;
    SubtreePartition subpart_subs_u;

#if USE_MKL_ILU
#pragma message("use mkl ilu")
    MKL_INT* ia = nullptr;
    MKL_INT* ja = nullptr;
    double* bilu = nullptr;
    MKL_INT ipar[128];
    double dpar[128];
    struct Trash {
        std::vector<double> sol;
        std::vector<double> rhs;
        std::vector<double> tmp;
        void Setup(int n) {
            sol.resize(n);
            rhs.resize(n);
            auto d = std::min(n, 150);
            auto tmp_size = (2 * d + 1) * n + d * (d + 9) / 2 + 1;
            tmp.resize(tmp_size);
        }
    } trash;
#endif

#if USE_MKL_SV
#pragma message("use mkl sv")
    sparse_matrix_t csrL = nullptr;
    sparse_matrix_t csrU = nullptr;
    double* trvec = nullptr;
#endif

    ~Ext() {
        destroy_array(csr_diag_ptr);
        destroy_array(task_done);
        subpart_fact.destroy();
        subpart_subs_l.destroy();
        subpart_subs_u.destroy();
#if USE_MKL_ILU
        if (ia) { mkl_free(ia); }
        if (ja) { mkl_free(ja); }
        if (bilu) { mkl_free(bilu); }
#endif
#if USE_MKL_SV
        if (csrL) { mkl_sparse_destroy(csrL); }
        if (csrU) { mkl_sparse_destroy(csrU); }
        if (trvec) { mkl_free(trvec); }
#endif
    }
};

struct IluSolver::ThreadLocalExt {
    double* col_modification = nullptr;
    int* interupt = nullptr;

    ~ThreadLocalExt() {
        destroy_array(col_modification);
        destroy_array(interupt);
    }
};

double ttt[4] = {0};
int ttti = 0;

IluSolver::~IluSolver() {
    destroy_array(extt_);
    destroy_object(ext_);
    DestroyCsrMatrix(&aMatrix_);
#if USE_MKL_ILU || USE_MKL_SV
    MKL_Free_Buffers();
#endif
    printf("%f %f\n", ttt[0], ttt[1]);
    printf("%f %f\n", ttt[2], ttt[3]);
}

#if 0
static int64_t estimate_cost(int n, const int* vbegin, const int* vend, const int* /*vtx*/) {
    int64_t est = 0;
    for (int i = 0; i < n; ++i) {
        est += vend[i] - vbegin[i];
    }
    return est;
}
#endif

template<typename EXT>
inline void print_algorithm(const EXT* ext_) {
    printf("subs threads: %d\n", ext_->subs_threads);
    puts(ext_->paralleled_subs_queue ? "par-subs-queue" : "seq-subs-queue");
    puts(ext_->packed ? "packed tasks" : "single task");
}

inline void print_subtree(const char* name, int threads, int n, const SubtreePartition& subtree) {
    printf("subtree %s: ", name);
    for (int i = 0; i < threads; ++i) {
        printf("%d ", subtree.part_ptr[i+1] - subtree.part_ptr[i]);
    }
    printf("[%d]\n", n - subtree.part_ptr[threads]);
    for (int i = 0; i < threads; ++i) {
        printf("(%d, %d) ", subtree.partitions[subtree.part_ptr[i]],
                            subtree.partitions[subtree.part_ptr[i+1]-1]);
    }
    printf("[(%d, %d)]\n", subtree.partitions[subtree.part_ptr[threads]],
                           subtree.partitions[n-1]);
}

#ifdef SHOW_ALGORITHM
#define PRINT_LU_EST(rl_est, ul_est) printf("U: %ld\nL: %ld\n", rl_est, ul_est);
//std::cout << "U: " << rl_est << std::endl << "L: " << ul_est << std::endl;
#define PRINT_ALGORITHM(e) print_algorithm(e)
#define PRINT_SUBTREE(...) print_subtree(__VA_ARGS__)
#else
#define PRINT_LU_EST(r, u)
#define PRINT_ALGORITHM(e)
#define PRINT_SUBTREE(...)
#endif

void
IluSolver::SetupMatrix() {
    // HERE, you could setup the reasonable stuctures of L and U as you want
    if (threads_) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads_);
    }
    else
#pragma omp parallel
#pragma omp single
    {
        threads_ = omp_get_num_threads();
    }
    printf("threads: %d\n", threads_);

    int n = aMatrix_.size;
    int* row_ptr = aMatrix_.row_ptr;
    int* col_idx = aMatrix_.col_idx;
    int nnz = GetCsrNonzeros(&aMatrix_);
    destroy_object(ext_);
    ext_ = new Ext;
    ext_->csr_diag_ptr = new int[n];
    ext_->task_done = new std::atomic_bool[n];
#if USE_MKL_ILU
    ext_->ia = static_cast<int*>(mkl_malloc(sizeof(int[n+1]), 64));
    ext_->ja = static_cast<int*>(mkl_malloc(sizeof(int[nnz]), 64));
    ext_->bilu = static_cast<double*>(mkl_malloc(sizeof(double[nnz]), 64));
    ext_->trash.Setup(n);
    int RCI_request;
    dfgmres_init(&n, ext_->trash.sol.data(), ext_->trash.rhs.data(), &RCI_request, ext_->ipar, ext_->dpar, ext_->trash.tmp.data());
    ext_->ipar[30] = 0;
    //ext_->ipar[30] = 1;
    //ext_->dpar[30] = 1e-20;
    //ext_->dpar[31] = 1e-16;
#pragma omp parallel for
    for (int i = 0; i <= n; ++i) {
        ext_->ia[i] = row_ptr[i] + 1;
    }
#pragma omp parallel for
    for (int i = 0; i < nnz; ++i) {
        ext_->ja[i] = col_idx[i] + 1;
    }
#endif
#if USE_MKL_SV
    ext_->trvec = static_cast<double*>(mkl_malloc(sizeof(double[n]), 64));
#endif

    // get diag_ptr
#ifdef CHECK_DIAG
    bool missing_diag = false;
#endif
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int ji = row_ptr[i]; ji < row_ptr[i+1]; ++ji) {
            if (col_idx[ji] >= i) {
#ifdef CHECK_DIAG
                if (col_idx[ji] != j) {
                    missing_diag = true;
                }
#endif
                ext_->csr_diag_ptr[i] = ji;
                break;
            }
        }
    }
#ifdef CHECK_DIAG
    if (missing_diag) {
        throw std::runtime_error("missing diag");
    }
#endif

    ext_->subs_threads = std::min(threads_, param::subs_max_threads);
    if (ext_->subs_threads == 1 || nnz / n < param::par_subs_min_nnz_per_line) {
        ext_->paralleled_subs_queue = false;
    }
    else {
        ext_->paralleled_subs_queue = true;
    }
    ext_->packed = threads_ > 1 && n >= param::granu_min_n;
    PRINT_ALGORITHM(ext_);

    int fact_granularity = 1, subs_granularity = 1;
    if (ext_->packed) {
        fact_granularity = param::fact_granu;
        if (ext_->paralleled_subs_queue) {
            subs_granularity = param::subs_granu;
        }
    }

    auto get_subpart = [n](int p1, int p2, int p3, int p4, ConstBiasArray<int> p5, const int* p6, const int* p7, SubtreePartition& out, const LoadBalancer& lb) {
        out.create(p1, n);
        out.ntasks = tree_schedule(p1, p2, p3, p4, p5, p6, p7, out.part_ptr, out.partitions, out.task_queue, lb);
    };
    LoadBalancer lb_fact {fact_granularity, row_ptr, ext_->csr_diag_ptr, 1};
    get_subpart(threads_, 0, n, 1, row_ptr, ext_->csr_diag_ptr, col_idx, ext_->subpart_fact, lb_fact);
    PRINT_SUBTREE("Fact", threads_, n, ext_->subpart_fact);
    if (ext_->subs_threads > 1) {
        LoadBalancer lb_subs1 {subs_granularity, row_ptr,  ext_->csr_diag_ptr, 1};
        LoadBalancer lb_subs2 {subs_granularity, ext_->csr_diag_ptr, row_ptr + 1, 0};
        get_subpart(ext_->subs_threads, 0, n, 1, row_ptr, ext_->csr_diag_ptr, col_idx, ext_->subpart_subs_l, lb_subs1);
        get_subpart(ext_->subs_threads, n-1, -1, -1, {ext_->csr_diag_ptr, 1}, row_ptr + 1, col_idx, ext_->subpart_subs_u, lb_subs2);
        if (!ext_->paralleled_subs_queue) {
            auto sort_queue = [n, t = ext_->subs_threads](SubtreePartition& p, const auto& cmp) {
                std::sort(&p.partitions[p.part_ptr[t]], &p.partitions[n], cmp);
            };
            sort_queue(ext_->subpart_subs_l, std::less<int>());
            sort_queue(ext_->subpart_subs_u, std::greater<int>());
        }
        PRINT_SUBTREE("SubsL", ext_->subs_threads, n, ext_->subpart_subs_l);
        PRINT_SUBTREE("SubsU", ext_->subs_threads, n, ext_->subpart_subs_u);
    }

    destroy_array(extt_);
    extt_ = new ThreadLocalExt[threads_];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        extt_[tid].col_modification = new double[n * fact_granularity];
        extt_[tid].interupt = new int[std::max(fact_granularity, subs_granularity)];
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
    if constexpr (std::is_same<JobI, std::nullptr_t>::value) {
        task_head.store(subpart.part_ptr[threads], std::memory_order_relaxed);
        task_tail = n;
    }
    else {
        task_head.store(0, std::memory_order_relaxed);
        task_tail = subpart.ntasks;
    }
#pragma omp parallel for schedule(static, 16)
        for (int j = 0; j < n; ++j) {
            task_done[j].store(false, std::memory_order_relaxed);
        }
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid < threads) {
            /* independent part */
            for (int k = subpart.part_ptr[tid]; k < subpart.part_ptr[tid + 1]; ++k) {
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
}

template <class JobS>
static void subpart_parallel_run_seq_queue(int threads, int n, const SubtreePartition& subpart,
                                           const JobS& fs) {
    auto tic = Rdtsc();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid < threads) {
            /* independent part */
            for (int k = subpart.part_ptr[tid]; k < subpart.part_ptr[tid + 1]; ++k) {
                fs(subpart.partitions[k]);
            }
        }
    }
    ttt[ttti] += Toc(tic);
    tic = Rdtsc();
            /* queue part */
    for (int k = subpart.part_ptr[threads]; k < n; ++k) {
        fs(subpart.partitions[k]);
    }
    ttt[ttti+1] += Toc(tic);
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
    static double scale(double a, double /*diag*/) {
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
static int left_looking_col(int j, const int* col_ptr, const int* row_idx, double* a, const int* diag_ptr,
                             double* col_modification, std::atomic_bool* task_done,
                             int interupt = 0) { // -1: new; -2: finished; >= 0: interupted
    int begin;
    if constexpr (R) {
        if (interupt == -2) {
            return -2;
        }
        begin = interupt;
    } else {
        for (int ji = col_ptr[j]; ji < col_ptr[j+1]; ++ji) {
            col_modification[row_idx[ji]] = 0;
        }
        begin = col_ptr[j];
    }
    int diag = diag_ptr[j];
    for (int ji = begin; ji < diag; ++ji) {
        int k = row_idx[ji];
        if (!W::busy_waiting(&task_done[k])) {
            return ji;
        }
        a[ji] = T::scale_U(a[ji] + col_modification[k], a[diag_ptr[k]]);
        for (int ki = diag_ptr[k] + 1; ki < col_ptr[k+1]; ++ki) {
            col_modification[row_idx[ki]] -= a[ki] * a[ji];
        }
    }
    a[diag] += col_modification[row_idx[diag]];
    double ajj = a[diag];
    for (int ji = diag + 1; ji < col_ptr[j+1]; ++ji) {
        a[ji] = T::scale_L(a[ji] + col_modification[row_idx[ji]], ajj);
    }

    task_done[j].store(true, std::memory_order_release);
    return -2;
}

template<typename Trans, typename Ext>
static void fact_parallel_run(int threads, int n, const int* col_ptr, const int* diag_ptr, const int* row_idx, double* a, Ext* ext, std::atomic_bool* task_done, SubtreePartition& subpart, bool packed) {
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
IluSolver::Factorize() {
    // HERE, do triangle decomposition
    // to calculate the values in L and U
    int n = aMatrix_.size;
#if USE_MKL_ILU
    int err;
    dcsrilu0(&n, aMatrix_.value, ext_->ia, ext_->ja, ext_->bilu, ext_->ipar, ext_->dpar, &err);
    //std::swap(ext_->bilu, aMatrix_.value);
    int one = 1;
    int nnz = GetCsrNonzeros(&aMatrix_);
    dcopy(&nnz, ext_->bilu, &one, aMatrix_.value, &one);
    return err == 0;
#else
    const int* col_ptr = aMatrix_.row_ptr;
    const int* diag_ptr = ext_->csr_diag_ptr;
    const int* row_idx = aMatrix_.col_idx;
    double* a = aMatrix_.value;
//#pragma omp parallel for
    //for (int i = 0; i < n; ++i) {
        //if (a[diag_ptr[i]] < 1e-16) {
            //a[diag_ptr[i]] = 1e-10;
        //}
    //}
    fact_parallel_run<Transpose, ThreadLocalExt>(threads_, n, col_ptr, diag_ptr, row_idx, a, extt_, ext_->task_done, ext_->subpart_fact, ext_->packed);
#endif
#if USE_MKL_SV
    if (ext_->csrL) {
        mkl_sparse_destroy(ext_->csrL);
        ext_->csrL = nullptr;
    }
    if (ext_->csrU) {
        mkl_sparse_destroy(ext_->csrU);
        ext_->csrU = nullptr;
    }
#endif
    return true;
}

template <typename W, bool R, bool S = false>
int substitute_row_L(int i, const int* row_ptr, const int* diag_ptr, const int* col_idx,
                        const double* lvalue, double* x, std::atomic_bool* task_done,
                        int interupt = 0) {
    int begin;
    if constexpr (R) {
        if (interupt == -2) {
            return -2;
        }
        begin = interupt;
    } else {
        begin = row_ptr[i];
    }
    double xi = x[i];
    for (int ii = begin, diag = diag_ptr[i]; ii < diag; ++ii) {
        int j = col_idx[ii];
        if constexpr (!S) {
            if (!W::busy_waiting(&task_done[j])) {
                x[i] = xi;
                return ii;
            }
        }
        xi -= x[j] * lvalue[ii];
    }
    x[i] = xi;
    if constexpr (!S) {
        task_done[i].store(true, std::memory_order_release);
    }
    return -2;
}

template <typename W, bool R, bool S = false>
int substitute_row_U(int i, const int* row_ptr, const int* diag_ptr, const int* col_idx,
                        const double* uvalue, double* x, std::atomic_bool* task_done,
                        int interupt = 0) {
    int begin;
    if constexpr (R) {
        if (interupt == -2) {
            return -2;
        }
        begin = interupt;
    } else {
        begin = row_ptr[i + 1] - 1;
    }
    double xi = x[i];
    int diag = diag_ptr[i];
    for (int ii = begin; ii > diag; --ii) { /* first complete first update */
        int j = col_idx[ii];
        if constexpr (!S) {
            if (!W::busy_waiting(&task_done[j])) {
                x[i] = xi;
                return ii;
            }
        }
        xi -= x[j] * uvalue[ii];
    }
    x[i] = xi / uvalue[diag];
    if constexpr (!S) {
        task_done[i].store(true, std::memory_order_release);
    }
    return -2;
}

void
IluSolver::Substitute(const double* b, double* x) {
    // HERE, use the L and U calculated by ILUSolver::Factorize to solve the triangle systems
    // to calculate the x
    int n = aMatrix_.size;
#if USE_MKL_SV
    auto& csrL = ext_->csrL;
    auto& csrU = ext_->csrU;
    matrix_descr descrL;
    descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrL.mode = SPARSE_FILL_MODE_LOWER;
    descrL.diag = SPARSE_DIAG_UNIT;
    matrix_descr descrU;
    descrU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrU.mode = SPARSE_FILL_MODE_UPPER;
    descrU.diag = SPARSE_DIAG_NON_UNIT;
    if (!csrL) {
        mkl_sparse_d_create_csr(&csrL, SPARSE_INDEX_BASE_ZERO, n, n, aMatrix_.row_ptr, aMatrix_.row_ptr+1, aMatrix_.col_idx, aMatrix_.value);
        mkl_sparse_set_sv_hint(csrL, SPARSE_OPERATION_NON_TRANSPOSE, descrL, 50);
        mkl_sparse_optimize(csrL);
        mkl_sparse_d_create_csr(&csrU, SPARSE_INDEX_BASE_ZERO, n, n, aMatrix_.row_ptr, aMatrix_.row_ptr+1, aMatrix_.col_idx, aMatrix_.value);
        mkl_sparse_set_sv_hint(csrU, SPARSE_OPERATION_NON_TRANSPOSE, descrU, 50);
        mkl_sparse_optimize(csrU);
    }
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrL, descrL, b, ext_->trvec);
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrU, descrU, ext_->trvec, x);
#else
    if (b != x) {
        cblas_dcopy(n, b, 1, x, 1);
    }
    int threads = ext_->subs_threads;
    if (threads == 1) {
        int* row_ptr = aMatrix_.row_ptr;
        int* col_idx = aMatrix_.col_idx;
        double* a = aMatrix_.value;
        for (int i = 0; i < n; ++i) {
            for (int ji = row_ptr[i]; ji < ext_->csr_diag_ptr[i]; ++ji) {
                int j = col_idx[ji];
                x[i] -= x[j] * a[ji];
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int ji = ext_->csr_diag_ptr[i] + 1; ji < row_ptr[i+1]; ++ji) {
                int j = col_idx[ji];
                x[i] -= x[j] * a[ji];
            }
            x[i] /= a[ext_->csr_diag_ptr[i]];
        }
    }
    else if (!ext_->paralleled_subs_queue) {
#define SUBS_ROW(D, W) substitute_row_##D<W,false,true>(i, aMatrix_.row_ptr, ext_->csr_diag_ptr, aMatrix_.col_idx, aMatrix_.value, x, nullptr)
        auto lfs = [&](int i) { SUBS_ROW(L, Skip); };
        auto ufs = [&](int i) { SUBS_ROW(U, Skip); };
        ttti = 0;
        subpart_parallel_run_seq_queue(threads, n, ext_->subpart_subs_l, lfs);
        ttti = 2;
        subpart_parallel_run_seq_queue(threads, n, ext_->subpart_subs_u, ufs);
#undef SUBS_ROW
    }
    else if (ext_->packed) {
#define SUBS_ROW(D, W) substitute_row_##D<W,false>(i, aMatrix_.row_ptr, ext_->csr_diag_ptr, aMatrix_.col_idx, aMatrix_.value, x, ext_->task_done)
#define SUBS_ROWR(D, W, I) substitute_row_##D<W,true>(i, aMatrix_.row_ptr, ext_->csr_diag_ptr, aMatrix_.col_idx, aMatrix_.value, x, ext_->task_done, I)
        auto lfs = [&](int i, int) { SUBS_ROW(L, Skip); };
        auto lfw = [&](int i, int tid, int k) { SUBS_ROWR(L, Wait, extt_[tid].interupt[k]); };
        auto lfi = [&](int i, int tid, int k) { extt_[tid].interupt[k] = SUBS_ROW(L, Interupt); };
        auto ufs = [&](int i, int) { SUBS_ROW(U, Skip); };
        auto ufw = [&](int i, int tid, int k) { SUBS_ROWR(U, Wait, extt_[tid].interupt[k]); };
        auto ufi = [&](int i, int tid, int k) { extt_[tid].interupt[k] = SUBS_ROW(U, Interupt); };
        subpart_parallel_run(threads, n, ext_->subpart_subs_l, lfs, lfw, lfi, ext_->task_done);
        subpart_parallel_run(threads, n, ext_->subpart_subs_u, ufs, ufw, ufi, ext_->task_done);
    }
    else {
        int threads = ext_->subs_threads;
        auto lfs = [&](int i, int) { SUBS_ROW(L, Skip); };
        auto lfw = [&](int i, int, int) { SUBS_ROW(L, Wait); };
        auto ufs = [&](int i, int) { SUBS_ROW(U, Skip); };
        auto ufw = [&](int i, int, int) { SUBS_ROW(U, Wait); };
        subpart_parallel_run(threads, n, ext_->subpart_subs_l, lfs, lfw, nullptr, ext_->task_done);
        subpart_parallel_run(threads, n, ext_->subpart_subs_u, ufs, ufw, nullptr, ext_->task_done);
    }
#undef SUBS_ROW
#undef SUBS_ROWR
#endif
}

} // namespace lljbash
