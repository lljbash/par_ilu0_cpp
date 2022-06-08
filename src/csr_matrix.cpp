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
    const int n_plus_one = n + 1;
    const int nnz = GetCsrNonzeros(src);
    const int one = 1;
    SetupCsrMatrix(dst, n, nnz);
    std::copy_n(src->row_ptr, n_plus_one, dst->row_ptr);
    std::copy_n(src->col_idx, nnz, dst->col_idx);
    dcopy(&nnz, src->value, &one, dst->value, &one);
}

void CopyCsrMatrixValues(CsrMatrix* dst, const CsrMatrix* src) {
    const int nnz = GetCsrNonzeros(src);
    const int one = 1;
    dcopy(&nnz, src->value, &one, dst->value, &one);
}

void DestroyCsrMatrix(CsrMatrix* mat) {
    SetupCsrMatrix(mat, -1, 0);
}

void ReadCsrMatrixMM1(CsrMatrix* mat, const char* filename) {
    std::FILE* fin = std::fopen(filename, "r");
    ON_SCOPE_EXIT { std::fclose(fin); };
    constexpr int kBufSize = 256;
    char buf[kBufSize];
    bool is_first_line = true;
    int n = -1;
    int nnz = 0;
    std::vector<std::tuple<int, int, double>> elements;
    while ((is_first_line || static_cast<int>(elements.size()) < nnz)
            && std::fgets(buf, kBufSize, fin)) {
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
                SetupCsrMatrix(mat, n, nnz);
                elements.reserve(nnz);
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
                elements.emplace_back(i - 1, j - 1, a);
            }
        }
    }
    std::sort(elements.begin(), elements.end());
    nnz = 0;
    int last_i = -1;
    for (auto [i, j, a] : elements) {
        if (last_i < i) {
            for (int ii = last_i + 1; ii <= i; ++ii) {
                mat->row_ptr[ii] = nnz;
            }
            last_i = i;
        }
        mat->col_idx[nnz] = j;
        mat->value[nnz] = a;
        ++nnz;
    }
    mat->row_ptr[n] = nnz;
}

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
