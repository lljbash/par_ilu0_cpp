#pragma once

#include "csc_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CscMatVec LLJBASH_DECL(CscMatVec)
void CscMatVec(const CscMatrix* mat, const double* vec, double* sol);

#ifdef __cplusplus
}
#endif
