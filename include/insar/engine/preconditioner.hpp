#pragma once

#include "insar/math/fft.hpp"

#include <vector>

namespace insar {

class IPreconditioner {
public:
    virtual ~IPreconditioner() = default;
    virtual void apply(const std::vector<float>& in, std::vector<float>& out) = 0;
};

class NeumannPoissonPreconditioner final : public IPreconditioner {
public:
    NeumannPoissonPreconditioner(int padded_width, int padded_height);

    void apply(const std::vector<float>& in, std::vector<float>& out) override;

    int padded_width() const noexcept { return pw_; }
    int padded_height() const noexcept { return ph_; }

private:
    int pw_ = 0;
    int ph_ = 0;
    int mw_ = 0;
    int mh_ = 0;

    std::vector<Complex> buf_a_;
    std::vector<Complex> buf_b_;
    std::vector<float> cos_kx_;
    std::vector<float> cos_ky_;
    math::Fft1DPlan fft_w_;
    math::Fft1DPlan fft_h_;
};

} // namespace insar
