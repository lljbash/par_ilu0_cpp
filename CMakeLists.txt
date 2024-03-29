#set(CMAKE_C_COMPILER "icc")
#set(CMAKE_CXX_COMPILER "icpc")
#set(CMAKE_C_COMPILER "gcc-12")
#set(CMAKE_CXX_COMPILER "g++-12")

project(par_ilu0_gmres C CXX)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# uncomment to modify default settings
#set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_CXX_FLAGS_RELEASE "-fast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -ffast-math -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
#set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

if (CMAKE_BUILD_TYPE STREQUAL Release)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    if(ipo_supported)
        message(STATUS "IPO / LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO / LTO not supported: <${error}>")
    endif()
endif()

option(SHOW_ALGORITHM "show algorithm" OFF)
option(USE_MKL "use mkl" OFF)
option(USE_LEVEL "use levelization" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(src)
add_subdirectory(src_c)
add_subdirectory(test)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    message(STATUS "OPTFLAGS: " ${CMAKE_CXX_FLAGS_DEBUG})
elseif (CMAKE_BUILD_TYPE STREQUAL Release)
    message(STATUS "OPTFLAGS: " ${CMAKE_CXX_FLAGS_RELEASE})
elseif (CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
    message(STATUS "OPTFLAGS: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif (CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
    message(STATUS "OPTFLAGS: " ${CMAKE_CXX_FLAGS_MINSIZEREL})
else()
    message(STATUS "OPTFLAGS: " ${CMAKE_CXX_FLAGS})
endif()
