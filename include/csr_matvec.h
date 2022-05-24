#pragma once

#include "csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CsrMatVec LLJBASH_DECL(CsrMatVec)
void CsrMatVec(const CsrMatrix* mat, const double* vec, double* sol);

#ifdef __cplusplus
}
#endif
