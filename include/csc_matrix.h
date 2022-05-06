#pragma once

#include "lljbash_decl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CscMatrix LLJBASH_DECL(CscMatrix)
struct CscMatrix {
    int size LLJBASH_DEFAULT_VALUE(-1);
    long max_nnz LLJBASH_DEFAULT_VALUE(0);
    long* col_ptr LLJBASH_DEFAULT_VALUE(nullptr);
    int* row_idx LLJBASH_DEFAULT_VALUE(nullptr);
    double* value LLJBASH_DEFAULT_VALUE(nullptr);
};
LLJBASH_STRUCT_TYPEDEF(CscMatrix);

#define CSC_MATRIX_DEFAULT LLJBASH_DECL(CSC_MATRIX_DEFAULT)
extern const CscMatrix CSC_MATRIX_DEFAULT;

#define SetupCscMatrix LLJBASH_DECL(SetupCscMatrix)
void SetupCscMatrix(CscMatrix* mat, int size, long max_nnz);

#define CopyCscMatrix LLJBASH_DECL(CopyCscMatrix)
void CopyCscMatrix(CscMatrix* dst, const CscMatrix* src);

#define DestroyCscMatrix LLJBASH_DECL(DestroyCscMatrix)
void DestroyCscMatrix(CscMatrix* mat);

#define GetCscNonzeros LLJBASH_DECL(GetCscNonzeros)
inline long GetCscNonzeros(const CscMatrix* mat) {
    return mat->col_ptr[mat->size];
}

#define ReadCscMatrixMM1 LLJBASH_DECL(ReadCscMatrixMM1)
void ReadCscMatrixMM1(CscMatrix* mat, const char* filename);

#ifdef __cplusplus
}
#endif
