#ifndef PAR_ILU0_C_H
#define PAR_ILU0_C_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#include "itsol/globheads.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* ParILU0SolverHandler;

ParILU0SolverHandler par_ilu0_create_solver(int nthreads);
void par_ilu0_destroy_solver(ParILU0SolverHandler hdl);
void par_ilu0_import_matrix(ParILU0SolverHandler hdl, csptr csmat);
int par_ilu0_factorize(ParILU0SolverHandler hdl, bool different_structure);
void par_ilu0_export_matrix(ParILU0SolverHandler hdl, iluptr lu);

#ifdef __cplusplus
}
#endif

#endif /* end of include guard: PAR_ILU0_C_H */
