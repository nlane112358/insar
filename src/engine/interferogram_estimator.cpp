#include "insar/engine/interferogram_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace insar {

size_t InterferogramEstimator::compute(
    const SlcPair& pair,
    int out_w,
    int out_h,
    int stride_x,
    int stride_y,
    int look_rx,
    int look_ry,
    std::vector<float>& wrapped,
    std::vector<float>& coherence,
    std::vector<uint8_t>& valid_mask) const
{
    pair.validate();
    if (out_w <= 0 || out_h <= 0 || stride_x <= 0 || stride_y <= 0 || look_rx < 0 || look_ry < 0) {
        throw std::invalid_argument("InterferogramEstimator received invalid output or stride parameters");
    }
    const size_t out_n = static_cast<size_t>(out_w) * out_h;
    if (wrapped.size() < out_n || coherence.size() < out_n || valid_mask.size() < out_n) {
        throw std::invalid_argument("InterferogramEstimator output buffer size mismatch");
    }
    const int in_w = pair.width();
    const int in_h = pair.height();
    const auto& slc1 = pair.reference.pixels();
    const auto& slc2 = pair.secondary.pixels();

    const int rx = std::max(look_rx, stride_x / 2);
    const int ry = std::max(look_ry, stride_y / 2);
    const int min_samples = std::max(1, ((2 * rx + 1) * (2 * ry + 1)) / 4);

    size_t total_valid = 0;

    #pragma omp parallel for reduction(+:total_valid) schedule(static)
    for (int oy = 0; oy < out_h; ++oy) {
        const int cy = oy * stride_y + stride_y / 2;
        const int y0 = std::max(0, cy - ry);
        const int y1 = std::min(in_h - 1, cy + ry);

        for (int ox = 0; ox < out_w; ++ox) {
            const int cx = ox * stride_x + stride_x / 2;
            const int x0 = std::max(0, cx - rx);
            const int x1 = std::min(in_w - 1, cx + rx);

            float sum_re = 0.0f;
            float sum_im = 0.0f;
            float pow1 = 0.0f;
            float pow2 = 0.0f;
            int count = 0;

            for (int j = y0; j <= y1; ++j) {
                const size_t row = static_cast<size_t>(j) * in_w;
                for (int i = x0; i <= x1; ++i) {
                    const Complex a = slc1[row + i];
                    const Complex b = slc2[row + i];
                    if (!math::is_valid_complex(a) || !math::is_valid_complex(b)) {
                        continue;
                    }

                    sum_re += a.real() * b.real() + a.imag() * b.imag();
                    sum_im += a.imag() * b.real() - a.real() * b.imag();
                    pow1 += a.real() * a.real() + a.imag() * a.imag();
                    pow2 += b.real() * b.real() + b.imag() * b.imag();
                    ++count;
                }
            }

            const size_t idx = static_cast<size_t>(oy) * out_w + ox;
            if (count >= min_samples && pow1 > 0.0f && pow2 > 0.0f) {
                wrapped[idx] = std::atan2(sum_im, sum_re);
                const float denom = std::sqrt(pow1 * pow2) + 1e-12f;
                coherence[idx] = std::clamp(std::hypot(sum_re, sum_im) / denom, 0.0f, 1.0f);
                valid_mask[idx] = 1;
                ++total_valid;
            } else {
                wrapped[idx] = 0.0f;
                coherence[idx] = 0.0f;
                valid_mask[idx] = 0;
            }
        }
    }

    return total_valid;
}

} // namespace insar
