#pragma once

#include "lljbash_decl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CsrMatrix LLJBASH_DECL(CsrMatrix)
struct CsrMatrix {
    int size LLJBASH_DEFAULT_VALUE(-1);
    long max_nnz LLJBASH_DEFAULT_VALUE(0);
    long* row_ptr LLJBASH_DEFAULT_VALUE(nullptr);
    int* col_idx LLJBASH_DEFAULT_VALUE(nullptr);
    double* value LLJBASH_DEFAULT_VALUE(nullptr);
};
LLJBASH_STRUCT_TYPEDEF(CsrMatrix);

#define CSR_MATRIX_DEFAULT LLJBASH_DECL(CSR_MATRIX_DEFAULT)
extern const CsrMatrix CSR_MATRIX_DEFAULT;

#define SetupCsrMatrix LLJBASH_DECL(SetupCsrMatrix)
void SetupCsrMatrix(CsrMatrix* mat, int size, long max_nnz);

#define CopyCsrMatrix LLJBASH_DECL(CopyCsrMatrix)
void CopyCsrMatrix(CsrMatrix* dst, const CsrMatrix* src);

#define DestroyCsrMatrix LLJBASH_DECL(DestroyCsrMatrix)
void DestroyCsrMatrix(CsrMatrix* mat);

#define GetCsrNonzeros LLJBASH_DECL(GetCsrNonzeros)
inline long GetCsrNonzeros(const CsrMatrix* mat) {
    return mat->row_ptr[mat->size];
}

#define ReadCsrMatrixMM1 LLJBASH_DECL(ReadCsrMatrixMM1)
void ReadCsrMatrixMM1(CsrMatrix* mat, const char* filename);

#ifdef __cplusplus
}
#endif
