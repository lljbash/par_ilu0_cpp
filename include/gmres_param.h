#pragma once

#include "lljbash_decl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GmresParameters LLJBASH_DECL(GmresParameters)
struct GmresParameters {
    double tolerance LLJBASH_DEFAULT_VALUE(1e-8);
    int krylov_subspace_dimension LLJBASH_DEFAULT_VALUE(60);
    int max_iterations LLJBASH_DEFAULT_VALUE(200);
};
LLJBASH_STRUCT_TYPEDEF(GmresParameters);

#ifdef __cplusplus
}
#endif
