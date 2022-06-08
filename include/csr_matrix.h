#pragma once

#include "lljbash_decl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CsrMatrix LLJBASH_DECL(CsrMatrix)
struct CsrMatrix {
    int size LLJBASH_DEFAULT_VALUE(-1);
    int max_nnz LLJBASH_DEFAULT_VALUE(0);
    int* row_ptr LLJBASH_DEFAULT_VALUE(nullptr);
    int* col_idx LLJBASH_DEFAULT_VALUE(nullptr);
    double* value LLJBASH_DEFAULT_VALUE(nullptr);
};
LLJBASH_STRUCT_TYPEDEF(CsrMatrix);

#define CSR_MATRIX_DEFAULT LLJBASH_DECL(CSR_MATRIX_DEFAULT)
extern const CsrMatrix CSR_MATRIX_DEFAULT;

#define SetupCsrMatrix LLJBASH_DECL(SetupCsrMatrix)
void SetupCsrMatrix(CsrMatrix* mat, int size, int max_nnz);

#define CopyCsrMatrix LLJBASH_DECL(CopyCsrMatrix)
void CopyCsrMatrix(CsrMatrix* dst, const CsrMatrix* src);

#define CopyCsrMatrixValues LLJBASH_DECL(CopyCsrMatrixValues)
void CopyCsrMatrixValues(CsrMatrix* dst, const CsrMatrix* src);

#define DestroyCsrMatrix LLJBASH_DECL(DestroyCsrMatrix)
void DestroyCsrMatrix(CsrMatrix* mat);

#define GetCsrNonzeros LLJBASH_DECL(GetCsrNonzeros)
inline int GetCsrNonzeros(const CsrMatrix* mat) {
    return mat->row_ptr[mat->size];
}

#define ReadCsrMatrixMM1 LLJBASH_DECL(ReadCsrMatrixMM1)
void ReadCsrMatrixMM1(CsrMatrix* mat, const char* filename);

#define CsrMatVec LLJBASH_DECL(CsrMatVec)
void CsrMatVec(const CsrMatrix* mat, const double* vec, double* sol);

#define CsrAmdOrder LLJBASH_DECL(CsrAmdOrder)
int CsrAmdOrder(const CsrMatrix* mat, int* p, int* ip);

#ifdef __cplusplus
}
#endif
