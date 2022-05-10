#include "csc_matrix.h"
#include <algorithm>
#include <tuple>
#include <vector>
#include "scope_guard.hpp"

template <typename T>
static void realloc_array(T*& arr, std::size_t size) {
    if (arr) {
        delete [] arr;
    }
    if (size) {
        arr = new T[size]{};
    }
    else {
        arr = nullptr;
    }
}

const CscMatrix CSC_MATRIX_DEFAULT;

void SetupCscMatrix(CscMatrix* mat, int size, long max_nnz) {
    mat->size = size;
    mat->max_nnz = max_nnz;
    realloc_array(mat->col_ptr, size + 1);
    realloc_array(mat->row_idx, max_nnz);
    realloc_array(mat->value, max_nnz);
}

void CopyCscMatrix(CscMatrix* dst, const CscMatrix* src) {
    SetupCscMatrix(dst, src->size, src->max_nnz);
    std::copy_n(src->col_ptr, src->size + 1, dst->col_ptr);
    std::copy_n(src->row_idx, src->max_nnz, dst->row_idx);
    std::copy_n(src->value, src->max_nnz, dst->value);
}

void DestroyCscMatrix(CscMatrix* mat) {
    SetupCscMatrix(mat, -1, 0);
}

void ReadCscMatrixMM1(CscMatrix* mat, const char* filename) {
    std::FILE* fin = std::fopen(filename, "r");
    ON_SCOPE_EXIT { std::fclose(fin); };
    constexpr int kBufSize = 256;
    char buf[kBufSize];
    bool is_first_line = true;
    int n = -1;
    long nnz = 0;
    std::vector<std::tuple<int, int, double>> elements;
    while ((is_first_line || static_cast<long>(elements.size()) < nnz)
            && std::fgets(buf, kBufSize, fin)) {
        if (buf[0] != '%') {
            if (is_first_line) {
                int m = -1;
                int ret = std::sscanf(buf, "%d%d%ld", &n, &m, &nnz);
                if (ret != 3) {
                    continue;
                }
                if (m > n) {
                    n = m;
                }
                SetupCscMatrix(mat, n, nnz);
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
                elements.emplace_back(j - 1, i - 1, a);
            }
        }
    }
    std::sort(elements.begin(), elements.end());
    nnz = 0;
    int last_j = -1;
    for (auto [j, i, a] : elements) {
        if (last_j < j) {
            for (int jj = last_j + 1; jj <= j; ++jj) {
                mat->col_ptr[jj] = nnz;
            }
            last_j = j;
        }
        mat->row_idx[nnz] = i;
        mat->value[nnz] = a;
        ++nnz;
    }
    mat->col_ptr[n] = nnz;
}
