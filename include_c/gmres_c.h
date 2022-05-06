#pragma once

#include "gmres_param.h"
#include "ilu_solver_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GmresHdl LLJBASH_DECL(GmresHdl)
typedef void* GmresHdl;

#define GmresCreate LLJBASH_DECL(GmresCreate)
GmresHdl GmresCreate();

#define GmresDestroy LLJBASH_DECL(GmresDestroy)
void GmresDestroy(GmresHdl hdl);

#define GmresGetParameters LLJBASH_DECL(GmresGetParameters)
GmresParameters* GmresGetParameters(GmresHdl hdl);

#define GmresSetPreconditioner LLJBASH_DECL(GmresSetPreconditioner)
void GmresSetPreconditioner(GmresHdl hdl, IluSolverHdl ilu);

#define GmresSolve LLJBASH_DECL(GmresSolve)
bool GmresSolve(GmresHdl hdl, const CscMatrix* mat, const double* rhs, double* sol, int* iter);

#ifdef __cplusplus
}
#endif
