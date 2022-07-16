#pragma once

#include "csr_matrix.h"

namespace lljbash {

// add structral nonzeros
// lof: level of fill
// nzmap[nnz(A)]: (output, optional) f->value[nzmap[i]] == a->value[i]
// return nnz(F)
int IlukSymbolic(const CsrMatrix* a, int lof, CsrMatrix* f, int* nzmap);

} // namespace lljbash
