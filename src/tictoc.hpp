#pragma once

#include <cstdint>

namespace lljbash {

inline uint64_t g_cycles;

inline uint64_t Rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ (
            "xorl %%eax, %%eax\n"
            "cpuid\n"
            "rdtsc\n"
            : "=a" (lo), "=d" (hi)
            :
            : "%ebx", "%ecx");
    return (uint64_t) hi << 32 | lo;
}

inline void Tic() {
    g_cycles = Rdtsc();
}

inline double Toc(uint64_t tic = g_cycles) {
    static constexpr double cpu_frequency = 2.4e9;
    return (double)(Rdtsc() - tic) / cpu_frequency;
}

} // namespace lljbash
