#include "insar/math/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace insar::math {

Fft1DPlan::Fft1DPlan(int size) {
    init(size);
}

void Fft1DPlan::init(int size) {
    if (size <= 0 || (size & (size - 1)) != 0) {
        throw std::invalid_argument("Fft1DPlan size must be a positive power of 2");
    }

    n_ = size;
    int log_n = 0;
    while ((1 << log_n) < n_) {
        ++log_n;
    }

    bit_rev_.resize(n_);
    for (int i = 0; i < n_; ++i) {
        int r = 0;
        for (int b = 0; b < log_n; ++b) {
            r = (r << 1) | ((i >> b) & 1);
        }
        bit_rev_[i] = r;
    }

    twiddles_.resize(n_ / 2);
    for (int k = 0; k < n_ / 2; ++k) {
        const double angle = -2.0 * static_cast<double>(kPi) * k / n_;
        twiddles_[k] = Complex(static_cast<float>(std::cos(angle)),
                               static_cast<float>(std::sin(angle)));
    }
}

void Fft1DPlan::execute(Complex* data, bool inverse) const {
    for (int i = 0; i < n_; ++i) {
        const int j = bit_rev_[i];
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    for (int len = 2; len <= n_; len <<= 1) {
        const int half = len >> 1;
        const int step = n_ / len;
        for (int i = 0; i < n_; i += len) {
            for (int j = 0; j < half; ++j) {
                Complex w = twiddles_[j * step];
                if (inverse) {
                    w = std::conj(w);
                }
                const Complex u = data[i + j];
                const Complex v = data[i + j + half] * w;
                data[i + j] = u + v;
                data[i + j + half] = u - v;
            }
        }
    }
}

void transpose_blocked(const Complex* src, Complex* dst, int rows, int cols) {
    constexpr int kTile = 32;
    #pragma omp parallel for schedule(static)
    for (int r0 = 0; r0 < rows; r0 += kTile) {
        const int r_end = std::min(r0 + kTile, rows);
        for (int c0 = 0; c0 < cols; c0 += kTile) {
            const int c_end = std::min(c0 + kTile, cols);
            for (int r = r0; r < r_end; ++r) {
                for (int c = c0; c < c_end; ++c) {
                    dst[static_cast<size_t>(c) * rows + r] =
                        src[static_cast<size_t>(r) * cols + c];
                }
            }
        }
    }
}

} // namespace insar::math
