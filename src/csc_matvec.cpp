#include "csc_matvec.h"
#include <cstring>

void CscMatVec(const CscMatrix* mat, const double* vec, double* sol) {
    std::memset(sol, 0, sizeof(double[mat->size]));
    for (int col = 0; col < mat->size; ++col) {
        for (long i = mat->col_ptr[col]; i < mat->col_ptr[col+1]; ++i) {
            int row = mat->row_idx[i];
            sol[row] += mat->value[i] * vec[col];
        }
    }
}
