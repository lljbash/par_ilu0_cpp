set(BLA_PREFER_PKGCONFIG ON)
find_package(BLAS REQUIRED)
message(STATUS "BLAS found: ${BLAS_LIBRARIES}")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP REQUIRED)

file(GLOB SRCS "*.cpp")
add_library(par-ilu0-gmres SHARED ${SRCS})
target_compile_options(par-ilu0-gmres PUBLIC -Wall -Wextra -Wconversion -Wno-sign-conversion)
target_link_libraries(par-ilu0-gmres PUBLIC ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads)
target_include_directories(par-ilu0-gmres PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (${SHOW_ALGORITHM})
    target_compile_definitions(par-ilu0-gmres PRIVATE SHOW_ALGORITHM)
endif()
