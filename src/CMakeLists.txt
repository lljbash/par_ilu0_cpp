if (DEFINED ENV{MKLROOT})
    message(STATUS "MKL found: " $ENV{MKLROOT})
else()
    message(FATAL_ERROR "MKL not found")
endif()

find_library(SuiteSparseAMD NAMES amd REQUIRED)
message(STATUS "SuiteSparse AMD found: " ${SuiteSparseAMD})

file(GLOB SRCS "*.cpp")
add_library(par-ilu0-gmres SHARED ${SRCS})
target_compile_options(par-ilu0-gmres PUBLIC -m64 -fopenmp -Wall -Wextra -Wconversion -Wno-sign-conversion)
target_link_libraries(par-ilu0-gmres PUBLIC ${SuiteSparseAMD} -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
target_include_directories(par-ilu0-gmres
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE $ENV{MKLROOT}/include)

if (${SHOW_ALGORITHM})
    target_compile_definitions(par-ilu0-gmres PRIVATE SHOW_ALGORITHM)
endif()

if (${USE_MKL})
    target_compile_definitions(par-ilu0-gmres PRIVATE USE_MKL_ILU=1 USE_MKL_SV=1)
endif()
