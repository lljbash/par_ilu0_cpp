#pragma once

#ifndef LLJBASH_DECL
#define LLJBASH_DECL(name) LLJBASH_##name
#endif

#ifndef LLJBASH_DEFAULT_VALUE
#ifdef __cplusplus
#define LLJBASH_DEFAULT_VALUE(a) = a
#else
#define LLJBASH_DEFAULT_VALUE(a)
#endif
#endif

#ifndef LLJBASH_STRUCT_TYPEDEF
#ifndef __cplusplus
#define LLJBASH_STRUCT_TYPEDEF(s) typedef struct s s
#else
#define LLJBASH_STRUCT_TYPEDEF(s)
#endif
#endif
