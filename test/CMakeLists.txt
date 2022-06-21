add_executable(test_c "test.c")
target_link_libraries(test_c par-ilu0-gmres_c)

add_executable(test_amd "amd_test.c")
target_link_libraries(test_amd par-ilu0-gmres_c)

add_executable(benchmark "benchmark.cpp")
target_link_libraries(benchmark par-ilu0-gmres)
