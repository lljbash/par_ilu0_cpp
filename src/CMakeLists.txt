if (DEFINED ENV{MKLROOT})
    message(STATUS "MKL found: " $ENV{MKLROOT})
else()
    message(FATAL_ERROR "MKL not found")
endif()

file(GLOB SRCS "*.cpp")
add_library(par-ilu0-gmres SHARED ${SRCS})
target_compile_options(par-ilu0-gmres PUBLIC -qopenmp -Wall -Wextra -Wconversion -Wno-sign-conversion)
target_link_libraries(par-ilu0-gmres PUBLIC -liomp5 -L$ENV{MKLROOT}/lib/intel64 -lmkl_rt -lpthread -lm -ldl)
target_include_directories(par-ilu0-gmres
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE $ENV{MKLROOT}/include)

if (${SHOW_ALGORITHM})
    target_compile_definitions(par-ilu0-gmres PRIVATE SHOW_ALGORITHM)
endif()