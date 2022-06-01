#pragma once

#ifndef __cplusplus
#include <stdbool.h>
#else
#include <cstdbool>
#endif

#include "csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IluSolverHdl LLJBASH_DECL(IluSolverHdl)
typedef void* IluSolverHdl;

#define IluSolverCreate LLJBASH_DECL(IluSolverCreate)
IluSolverHdl IluSolverCreate(int nthreads);

#define IluSolverDestroy LLJBASH_DECL(IluSolverDestroy)
void IluSolverDestroy(IluSolverHdl hdl);

#define IluSolverGetMatrix LLJBASH_DECL(IluSolverGetMatrix)
CsrMatrix* IluSolverGetMatrix(IluSolverHdl hdl);

#define IluSolverSetup LLJBASH_DECL(IluSolverSetup)
int IluSolverSetup(IluSolverHdl hdl);

#define IluSolverFactorize LLJBASH_DECL(IluSolverFactorize)
int IluSolverFactorize(IluSolverHdl hdl);

#ifdef __cplusplus
}
#endif
