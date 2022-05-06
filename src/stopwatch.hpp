#pragma once

#include <chrono>

namespace lljbash {

class Stopwatch {
    using clock = std::chrono::steady_clock;
    std::chrono::time_point<clock> start_tp_;

public:
    Stopwatch() : start_tp_{clock::now()} {}

    std::chrono::duration<double> elapsed() const {
        return std::chrono::duration<double>(clock::now() - start_tp_);
    }

    void reset() {
        start_tp_ = clock ::now();
    }
};

} // namespace lljbash
