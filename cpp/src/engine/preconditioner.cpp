#include "insar/engine/preconditioner.hpp"

#include <cmath>
#include <stdexcept>

namespace insar {

NeumannPoissonPreconditioner::NeumannPoissonPreconditioner(int padded_width, int padded_height)
    : pw_(padded_width),
      ph_(padded_height),
      mw_(2 * padded_width),
      mh_(2 * padded_height)
{
    if (pw_ <= 0 || ph_ <= 0 || (pw_ & (pw_ - 1)) != 0 || (ph_ & (ph_ - 1)) != 0) {
        throw std::invalid_argument("NeumannPoissonPreconditioner dimensions must be positive powers of 2");
    }

    buf_a_.resize(static_cast<size_t>(mh_) * mw_);
    buf_b_.resize(static_cast<size_t>(mw_) * mh_);

    fft_w_.init(mw_);
    fft_h_.init(mh_);

    cos_kx_.resize(mw_);
    for (int kx = 0; kx < mw_; ++kx) {
        cos_kx_[kx] = std::cos(math::kPi * static_cast<float>(kx) / static_cast<float>(pw_));
    }

    cos_ky_.resize(mh_);
    for (int ky = 0; ky < mh_; ++ky) {
        cos_ky_[ky] = std::cos(math::kPi * static_cast<float>(ky) / static_cast<float>(ph_));
    }
}

void NeumannPoissonPreconditioner::apply(const std::vector<float>& in, std::vector<float>& out) {
    const size_t expected = static_cast<size_t>(pw_) * ph_;
    if (in.size() < expected || out.size() < expected) {
        throw std::invalid_argument("NeumannPoissonPreconditioner buffer size mismatch");
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ph_; ++y) {
        const size_t row_top = static_cast<size_t>(y) * mw_;
        const size_t row_bot = static_cast<size_t>(mh_ - 1 - y) * mw_;
        for (int x = 0; x < pw_; ++x) {
            const Complex v(in[static_cast<size_t>(y) * pw_ + x], 0.0f);
            buf_a_[row_top + x] = v;
            buf_a_[row_top + (mw_ - 1 - x)] = v;
            buf_a_[row_bot + x] = v;
            buf_a_[row_bot + (mw_ - 1 - x)] = v;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < mh_; ++y) {
        fft_w_.execute(&buf_a_[static_cast<size_t>(y) * mw_], false);
    }

    math::transpose_blocked(buf_a_.data(), buf_b_.data(), mh_, mw_);

    const float inv_area = 1.0f / (static_cast<float>(mw_) * static_cast<float>(mh_));
    #pragma omp parallel for schedule(static)
    for (int kx = 0; kx < mw_; ++kx) {
        Complex* col = &buf_b_[static_cast<size_t>(kx) * mh_];
        fft_h_.execute(col, false);

        const float cx = cos_kx_[kx];
        for (int ky = 0; ky < mh_; ++ky) {
            if (kx == 0 && ky == 0) {
                col[0] = Complex(0.0f, 0.0f);
            } else {
                const float eig = 4.0f - 2.0f * cx - 2.0f * cos_ky_[ky];
                col[ky] *= (inv_area / eig);
            }
        }

        fft_h_.execute(col, true);
    }

    math::transpose_blocked(buf_b_.data(), buf_a_.data(), mw_, mh_);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ph_; ++y) {
        Complex* row = &buf_a_[static_cast<size_t>(y) * mw_];
        fft_w_.execute(row, true);
        const size_t out_row = static_cast<size_t>(y) * pw_;
        for (int x = 0; x < pw_; ++x) {
            out[out_row + x] = row[x].real();
        }
    }
}

} // namespace insar
