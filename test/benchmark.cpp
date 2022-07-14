#include <cstdio>
#include <map>
#include <mkl.h>
#include "cxxopts.hpp"
#include "gmres.hpp"
#include "tictoc.hpp"
#include "scope_guard.hpp"

#define MKL_MALLOC(T, n) \
    static_cast<T*>(mkl_malloc(sizeof(T[n]), 64))

using namespace lljbash;

int main(int argc, char* argv[]) {
    cxxopts::Options options(argv[0], "benckmark for parallel ILU(0) GMRES");
    options.add_options()
        ("m,mat", "Coefficient matrix file (MM1)", cxxopts::value<std::string>())
        ("k,lof", "ILU(k) level of fill", cxxopts::value<int>()->default_value("0"))
        ("amd", "AMD reorder")
        ("nd", "ND reorder")
        //("r,rhs", "Right hand side file", cxxopts::value<std::string>()->default_value(""))
        ("e,eps", "GMRES tolerance", cxxopts::value<double>()->default_value("1e-8"))
        ("i,maxit", "GMRES max iterations", cxxopts::value<int>()->default_value("1000"))
        ("r,restart", "GMRES restart iterations", cxxopts::value<int>()->default_value("60"))
        ("t,threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("n,rep", "Number of repeats", cxxopts::value<int>()->default_value("1"))
        ("w,warmup", "Include warmups", cxxopts::value<int>()->implicit_value("1"))
        ("onlyfact", "only test factorization")
        ("onlysubs", "only test substitution")
        ("h,help", "Print usage")
    ;
    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        printf("%s\n", options.help().c_str());
        exit(0);
    }

    CsrMatrix csr;
    ReadCsrMatrixMM1(&csr, args["mat"].as<std::string>().c_str());
    int n = csr.size;
    int nnz = GetCsrNonzeros(&csr);

    bool amd = args["amd"].as<bool>();
    bool nd = args["nd"].as<bool>();
    if (amd || nd) {
        auto amd_p = MKL_MALLOC(int, n);
        auto amd_ip = MKL_MALLOC(int, n);
        ON_SCOPE_EXIT {
            mkl_free(amd_p);
            mkl_free(amd_ip);
        };
        if (amd) {
            CsrAmdOrder(&csr, amd_p, amd_ip);
        }
        else {
            CsrNdOrder(&csr, amd_p, amd_ip);
        }
        std::vector<std::tuple<int, int, double>> elements;
        elements.reserve(nnz);
        for (int i = 0; i < n; ++i) {
            for (int ji = csr.row_ptr[i]; ji < csr.row_ptr[i+1]; ++ji) {
                int j = csr.col_idx[ji];
                double v = csr.value[ji];
                elements.emplace_back(amd_ip[i], amd_ip[j], v);
            }
        }
        sort(elements.begin(), elements.end());
        int last_i = -1;
        int innz = 0;
        for (auto [i, j, v] : elements) {
            while (last_i < i) {
                ++last_i;
                csr.row_ptr[last_i] = innz;
            }
            csr.col_idx[innz] = j;
            csr.value[innz] = v;
            ++innz;
        }
    }

    auto rhs = MKL_MALLOC(double, n);
    auto sol = MKL_MALLOC(double, n);
    auto x = MKL_MALLOC(double, n);

    std::fill_n(sol, n, 1.0);
    CsrMatVec(&csr, sol, rhs);

    IluSolver ilu;
    CsrMatrix* csr_ilu = ilu.GetMatrix();
    PreconditionedGmres gmres;
    gmres.SetIluPreconditioner(&ilu);
    gmres.param().tolerance = args["eps"].as<double>();
    gmres.param().max_iterations = args["maxit"].as<int>();
    gmres.param().krylov_subspace_dimension = args["restart"].as<int>();
    int threads = args["threads"].as<int>();
    ilu.SetThreads(threads);

    int lof = args["lof"].as<int>();
    int knnz = 0;
    std::vector<std::map<int, int>> row_elements(n);
    for (int i = 0; i < n; ++i) {
        for (int ji = csr.row_ptr[i]; ji < csr.row_ptr[i+1]; ++ji) {
            row_elements[i][csr.col_idx[ji]] = 0;
        }
        for (auto kit = row_elements[i].begin(); kit != row_elements[i].end(); ++kit) {
            int k = kit->first;
            if (k >= i) {
                break;
            }
            if (kit->second >= lof) {
                continue;
            }
            for (auto jit = row_elements[k].find(k); jit != row_elements[k].end(); ++jit) {
                int level = kit->second + jit->second + 1;
                if (level <= lof) {
                    row_elements[i].try_emplace(jit->first, level);
                }
            }
        }
        knnz += static_cast<int>(row_elements[i].size());
    }
    std::vector<int> nzmap(nnz);
    SetupCsrMatrix(csr_ilu, n, knnz);
    int innz = 0;
    int iknnz = 0;
    for (int i = 0; i < n; ++i) {
        csr_ilu->row_ptr[i] = iknnz;
        for (auto [j, level] : row_elements[i]) {
            if (level == 0) {
                nzmap[innz] = iknnz;
                ++innz;
            }
            csr_ilu->col_idx[iknnz] = j;
            ++iknnz;
        }
    }
    csr_ilu->row_ptr[n] = knnz;
    if (innz != nnz) {
        std::puts("error");
        std::exit(-1);
    }
    printf("knnz: %d\n", knnz);

    double setup_time = 0.0;
    double fact_time = 0.0;
    double gmres_time = 0.0;

    auto call = [](const auto& f, double& time) {
        auto tic = Rdtsc();
        ON_SCOPE_EXIT { time += Toc(tic); };
        return f();
    };

    auto runGmres = [&](bool setup) {
        //CopyCsrMatrix(csr_ilu, &csr);
        std::memset(csr_ilu->value, 0, sizeof(double) * knnz);
        for (int i = 0; i < nnz; ++i) {
            csr_ilu->value[nzmap[i]] = csr.value[i];
        }
        if (setup) {
            call([&]() { ilu.SetupMatrix(); }, setup_time);
        }
        call([&]() { ilu.Factorize(); }, fact_time);
        std::fill_n(x, n, 0.0);
        auto ret = call([&]() { return gmres.Solve(&csr, rhs, x); }, gmres_time);
        if (!ret.first) {
            std::puts("GMRES error");
            std::exit(0);
        }
        printf("iter: %d\n", ret.second);
    };

    auto runFact = [&](bool setup) {
        std::memset(csr_ilu->value, 0, sizeof(double) * knnz);
        for (int i = 0; i < nnz; ++i) {
            csr_ilu->value[nzmap[i]] = csr.value[i];
        }
        if (setup) {
            call([&]() { ilu.SetupMatrix(); }, setup_time);
        }
        call([&]() { ilu.Factorize(); }, fact_time);
    };

    auto runSubs = [&](bool setup) {
        if (setup) {
            std::memset(csr_ilu->value, 0, sizeof(double) * knnz);
            for (int i = 0; i < nnz; ++i) {
                csr_ilu->value[nzmap[i]] = csr.value[i];
            }
            call([&]() { ilu.SetupMatrix(); }, setup_time);
            ilu.Factorize();
        }
        call([&]() { ilu.Substitute(rhs, x); }, gmres.stat().total_precon_time);
    };

    std::function<void(bool)> run;
    if (args.count("onlyfact")) {
        run = runFact;
    }
    else if (args.count("onlysubs")) {
        run = runSubs;
    }
    else {
        run = runGmres;
    }

    if (args.count("warmup")) {
        int warmup = args["warmup"].as<int>();
        for (int i = 0; i < warmup; ++i) {
            run(i == 0);
        }
    }

    setup_time = 0.0;
    fact_time = 0.0;
    gmres_time = 0.0;

    int rep = args["rep"].as<int>();
    for (int i = 0; i < rep; ++i) {
        run(i == 0);
    }

    // check
    cblas_daxpy(n, -1.0, sol, 1, x, 1);
    double max_err = std::abs(x[cblas_idamax(n, x, 1) - 1]);
    std::printf("max_err: %g\n", max_err);

    std::puts("\nSummary:");
    std::printf("n:                          %d\n", n);
    std::printf("nnz:                        %d\n", nnz);
    std::printf("k:                          %d\n", lof);
    std::printf("nnz(F)/nnz(A):              %f\n", (double) knnz / nnz);
    std::printf("gmres tolerance:            %g\n", gmres.param().tolerance);
    std::printf("gmres max iterations:       %d\n", gmres.param().max_iterations);
    std::printf("gmres restart iterations:   %d\n", gmres.param().krylov_subspace_dimension);
    std::printf("threads:                    %d\n", threads);
    std::printf("reps:                       %d\n", rep);
    std::printf("setup time (s):             %g\n", setup_time);
    std::printf("total factorize time (s):   %g\n", fact_time);
    std::printf("total gmres time (s):       %g\n", gmres_time);
    std::printf("total mv time (s):          %g\n", gmres.stat().total_mv_time);
    std::printf("total precon time (s):      %g\n", gmres.stat().total_precon_time);
    std::printf("average factorize time (s): %g\n", fact_time / rep);
    std::printf("average precon time (s):    %g\n", gmres.stat().total_precon_time / rep);
    std::printf("average gmres time (s):     %g\n", gmres_time / rep);

    mkl_free(rhs);
    mkl_free(sol);
    mkl_free(x);
    MKL_Free_Buffers();
    return 0;
}
