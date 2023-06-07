#pragma once

#include "csr_matrix.h"
#include "lljbash_decl.h"

#ifdef __cplusplus
extern "C" {
#endif

// add structral nonzeros
// lof: level of fill
// nzmap[nnz(A)]: (output, optional) f->value[nzmap[i]] == a->value[i]
// return nnz(F)
#define IlukSymbolic LLJBASH_DECL(IlukSymbolic)
int IlukSymbolic(const CsrMatrix* a, int lof, CsrMatrix* f, int* nzmap);

#ifdef __cplusplus
}
#endif
