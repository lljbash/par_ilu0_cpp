#include "csr_matvec.h"
#include <algorithm>

void CsrMatVec(const CsrMatrix* mat, const double* vec, double* sol) {
    std::fill_n(sol, mat->size, 0);
    for (int row = 0; row < mat->size; ++row) {
        for (long i = mat->row_ptr[row]; i < mat->row_ptr[row+1]; ++i) {
            int col = mat->col_idx[i];
            sol[row] += mat->value[i] * vec[col];
        }
    }
}
