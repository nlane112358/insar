#pragma once

#include "insar/core/types.hpp"

#include <vector>

namespace insar::math {

class Fft1DPlan {
public:
    Fft1DPlan() = default;
    explicit Fft1DPlan(int size);

    void init(int size);
    void execute(Complex* data, bool inverse) const;

    int size() const noexcept { return n_; }

private:
    int n_ = 0;
    std::vector<int> bit_rev_;
    std::vector<Complex> twiddles_;
};

void transpose_blocked(const Complex* src, Complex* dst, int rows, int cols);

} // namespace insar::math
