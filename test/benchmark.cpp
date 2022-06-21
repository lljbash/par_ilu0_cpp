#include <cstdio>
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
        //("r,rhs", "Right hand side file", cxxopts::value<std::string>()->default_value(""))
        ("e,eps", "GMRES tolerance", cxxopts::value<double>()->default_value("1e-8"))
        ("i,maxit", "GMRES max iterations", cxxopts::value<int>()->default_value("1000"))
        ("r,restart", "GMRES restart iterations", cxxopts::value<int>()->default_value("60"))
        ("t,threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("n,rep", "Number of repeats", cxxopts::value<int>()->default_value("1"))
        ("w,warmup", "Include warmups", cxxopts::value<int>()->implicit_value("1"))
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

    double setup_time = 0.0;
    double fact_time = 0.0;
    double gmres_time = 0.0;

    auto call = [](const auto& f, double& time) {
        auto tic = Rdtsc();
        ON_SCOPE_EXIT { time += Toc(tic); };
        return f();
    };

    auto run = [&](bool setup) {
        CopyCsrMatrix(csr_ilu, &csr);
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
    std::printf("nnz:                        %d\n", GetCsrNonzeros(&csr));
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
    std::printf("average gmres time (s):     %g\n", gmres_time / rep);

    mkl_free(rhs);
    mkl_free(sol);
    mkl_free(x);
    MKL_Free_Buffers();
    return 0;
}
