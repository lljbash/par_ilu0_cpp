#include "csr_matrix.h"
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#include <mkl.h>
#include <suitesparse/amd.h>
#include "scope_guard.hpp"

template <typename T>
static void realloc_array(T*& arr, std::size_t size) {
    if (arr) {
        mkl_free(arr);
    }
    if (size) {
        arr = static_cast<T*>(mkl_malloc(sizeof(T[size]), 64));
    }
    else {
        arr = nullptr;
    }
}

const CsrMatrix CSR_MATRIX_DEFAULT;

void SetupCsrMatrix(CsrMatrix* mat, int size, int max_nnz) {
    mat->size = size;
    mat->max_nnz = max_nnz;
    realloc_array(mat->row_ptr, size + 1);
    realloc_array(mat->col_idx, max_nnz);
    realloc_array(mat->value, max_nnz);
}

void CopyCsrMatrix(CsrMatrix* dst, const CsrMatrix* src) {
    const int n = src->size;
    const int nnz = GetCsrNonzeros(src);
    SetupCsrMatrix(dst, n, nnz);
    std::copy_n(src->row_ptr, n + 1, dst->row_ptr);
    std::copy_n(src->col_idx, nnz, dst->col_idx);
    cblas_dcopy(nnz, src->value, 1, dst->value, 1);
}

void CopyCsrMatrixValues(CsrMatrix* dst, const CsrMatrix* src) {
    const int nnz = GetCsrNonzeros(src);
    cblas_dcopy(nnz, src->value, 1, dst->value, 1);
}

void DestroyCsrMatrix(CsrMatrix* mat) {
    SetupCsrMatrix(mat, -1, 0);
}

#define MIN_DIAG 1e-12

static bool match(int i, const std::vector<std::vector<std::pair<int, double>>>& x, int* p, int* vis) {
    for (auto [j, a] : x[i]) {
        if (std::abs(a) > MIN_DIAG && !vis[j]) {
            vis[j] = 1;
            if (p[j] == -1 || match(p[j], x, p, vis)) {
                p[j] = i;
                return true;
            }
        }
    }
    return false;
}

static void max_match(const std::vector<std::vector<std::pair<int, double>>>& x, int* p) {
    int n = static_cast<int>(x.size());
    std::vector<int> vis(n);
    std::vector<int> ip(p, p + n);
    for (int i = 0; i < n; ++i) {
        std::fill(vis.begin(), vis.end(), 0);
        if (ip[i] == -1 && !match(i, x, p, vis.data())) {
            std::puts("error while matching");
            std::exit(-1);
        }
    }
}

void ReadCsrMatrixMM1(CsrMatrix* mat, const char* filename) {
    std::FILE* fin = std::fopen(filename, "r");
    if (!fin) {
        std::fprintf(stderr, "cannot open file %s\n", filename);
        exit(-1);
    }
    ON_SCOPE_EXIT { std::fclose(fin); };
    constexpr int kBufSize = 256;
    char buf[kBufSize];
    bool is_first_line = true;
    int n = -1;
    int nnz = 0;
    std::vector<std::vector<std::pair<int, double>>> elements;
    std::vector<int> p;
    int innz = 0;
    while ((is_first_line || innz < nnz) && std::fgets(buf, kBufSize, fin)) {
        if (buf[0] != '%') {
            if (is_first_line) {
                int m = -1;
                int ret = std::sscanf(buf, "%d%d%d", &n, &m, &nnz);
                if (ret != 3) {
                    continue;
                }
                if (m > n) {
                    n = m;
                }
                SetupCsrMatrix(mat, n, nnz + n);
                elements.resize(n);
                p.resize(n, -1);
                is_first_line = false;
            }
            else {
                int i = 0;
                int j = 0;
                double a = 0;
                int ret = std::sscanf(buf, "%d%d%lf", &i, &j, &a);
                if (ret != 3) {
                    continue;
                }
                elements[i-1].emplace_back(j - 1, a);
                if (i == j && std::abs(a) > MIN_DIAG) {
                    p[i-1] = i - 1;
                }
                ++innz;
            }
        }
    }
    if (innz != nnz) {
        std::puts("nnz number error");
        std::exit(-1);
    }
    for (int i = 0; i < n; ++i) {
        std::sort(elements[i].begin(), elements[i].end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
    }
    max_match(elements, p.data());
    innz = 0;
    for (int i = 0; i < n; ++i) {
        int k = p[i];
        std::sort(elements[k].begin(), elements[k].end());
        mat->row_ptr[i] = innz;
        bool diag = false;
        for (auto [j, a] : elements[k]) {
            mat->col_idx[innz] = j;
            mat->value[innz] = a;
            ++innz;
            if (j == i && std::abs(a) > MIN_DIAG) {
                diag = true;
            }
        }
        if (!diag) {
            puts("error: missing diag");
            std::exit(-1);
        }
    }
    mat->row_ptr[n] = innz;
}

#undef MIN_DIAG

void CsrMatVec(const CsrMatrix* mat, const double* vec, double* sol) {
    std::fill_n(sol, mat->size, 0);
    for (int row = 0; row < mat->size; ++row) {
        for (int i = mat->row_ptr[row]; i < mat->row_ptr[row+1]; ++i) {
            int col = mat->col_idx[i];
            sol[row] += mat->value[i] * vec[col];
        }
    }
}

int CsrAmdOrder(const CsrMatrix* mat, int* p, int* ip) {
    int n = mat->size;
    //int nnz = GetCsrNonzeros(mat);
    //std::vector<int> ap(n + 2, 0);
    //std::vector<int> ai(nnz);
    //for (int i = 0; i < nnz; ++i) {
        //ap[mat->col_idx[i]+2]++;
    //}
    //std::partial_sum(ap.begin(), ap.end(), ap.begin());
    //for (int i = 0; i < n; ++i) {
        //for (int j = mat->row_ptr[i]; j < mat->row_ptr[i+1]; ++j) {
            //ai[ap[mat->col_idx[j]+1]++] = i;
        //}
    //}
    //auto ret = amd_order(mat->size, ap.data(), ai.data(), p, nullptr, nullptr);
    auto ret = amd_order(mat->size, mat->row_ptr, mat->col_idx, p, nullptr, nullptr);
    if (ip) {
        for (int i = 0; i < n; ++i) {
            ip[p[i]] = i;
        }
    }
    return -(ret == AMD_INVALID || ret == AMD_OUT_OF_MEMORY);
}
