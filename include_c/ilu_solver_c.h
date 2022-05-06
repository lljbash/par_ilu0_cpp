#pragma once

#ifndef __cplusplus
#include <stdbool.h>
#else
#include <cstdbool>
#endif

#include "csc_matrix.h"

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
CscMatrix* IluSolverGetMatrix(IluSolverHdl hdl);

#define IluSolverFactorize LLJBASH_DECL(IluSolverFactorize)
int IluSolverFactorize(IluSolverHdl hdl, bool different_structure);

#ifdef __cplusplus
}
#endif
