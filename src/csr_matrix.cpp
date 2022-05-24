#include "csr_matrix.h"
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

const CsrMatrix CSR_MATRIX_DEFAULT;

void SetupCsrMatrix(CsrMatrix* mat, int size, int max_nnz) {
    mat->size = size;
    mat->max_nnz = max_nnz;
    realloc_array(mat->row_ptr, size + 1);
    realloc_array(mat->col_idx, max_nnz);
    realloc_array(mat->value, max_nnz);
}

void CopyCsrMatrix(CsrMatrix* dst, const CsrMatrix* src) {
    SetupCsrMatrix(dst, src->size, src->max_nnz);
    std::copy_n(src->row_ptr, src->size + 1, dst->row_ptr);
    std::copy_n(src->col_idx, src->max_nnz, dst->col_idx);
    std::copy_n(src->value, src->max_nnz, dst->value);
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
