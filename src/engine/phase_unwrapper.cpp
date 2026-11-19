#include "insar/engine/phase_unwrapper.hpp"

#include "insar/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace insar {

PcgPhaseUnwrapper::PcgPhaseUnwrapper(int out_width, int out_height)
    : PcgPhaseUnwrapper(
          out_width,
          out_height,
          std::make_unique<NeumannPoissonPreconditioner>(
              math::next_power_of_two(out_width),
              math::next_power_of_two(out_height)))
{
}

PcgPhaseUnwrapper::PcgPhaseUnwrapper(
    int out_width,
    int out_height,
    std::unique_ptr<IPreconditioner> preconditioner)
    : w_(out_width),
      h_(out_height),
      pw_(math::next_power_of_two(out_width)),
      ph_(math::next_power_of_two(out_height)),
      pn_(static_cast<size_t>(pw_) * ph_),
      laplacian_(pw_, ph_),
      preconditioner_(std::move(preconditioner)),
      phi_pad_(pn_),
      r_(pn_),
      z_(pn_),
      p_(pn_),
      Ap_(pn_)
{
    if (w_ <= 0 || h_ <= 0) {
        throw std::invalid_argument("PcgPhaseUnwrapper dimensions must be positive");
    }
    if (!preconditioner_) {
        throw std::invalid_argument("PcgPhaseUnwrapper requires a non-null preconditioner");
    }
}

UnwrapResult PcgPhaseUnwrapper::unwrap(
    const std::vector<float>& wrapped_phase,
    const std::vector<float>& coherence,
    const std::vector<uint8_t>& valid_mask,
    std::vector<float>& unwrapped_phase,
    int max_iters,
    float tol,
    float min_edge_weight)
{
    UnwrapResult result;
    const size_t out_n = static_cast<size_t>(w_) * h_;
    if (unwrapped_phase.size() != out_n) {
        unwrapped_phase.assign(out_n, 0.0f);
    }

    std::fill(phi_pad_.begin(), phi_pad_.end(), 0.0f);
    laplacian_.build(w_, h_, wrapped_phase, coherence, valid_mask, min_edge_weight, r_);

    const double b_norm = std::sqrt(dot_product(r_, r_));
    if (b_norm > 1e-12) {
        preconditioner_->apply(r_, z_);
        std::copy(z_.begin(), z_.end(), p_.begin());
        double rz_old = dot_product(r_, z_);

        for (int iter = 1; iter <= max_iters; ++iter) {
            laplacian_.apply(p_, Ap_);
            const double pAp = dot_product(p_, Ap_);
            if (pAp <= 1e-18) {
                break;
            }

            const float alpha = static_cast<float>(rz_old / pAp);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(pn_); ++i) {
                phi_pad_[i] += alpha * p_[i];
                r_[i] -= alpha * Ap_[i];
            }

            const double r_norm = std::sqrt(dot_product(r_, r_));
            result.iterations = iter;
            result.rel_residual = r_norm / b_norm;
            if (result.rel_residual < tol) {
                break;
            }

            preconditioner_->apply(r_, z_);
            const double rz_new = dot_product(r_, z_);
            const float beta = static_cast<float>(rz_new / (rz_old + 1e-30));

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(pn_); ++i) {
                p_[i] = z_[i] + beta * p_[i];
            }
            rz_old = rz_new;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h_; ++y) {
        const float* src_row = &phi_pad_[static_cast<size_t>(y) * pw_];
        float* dst_row = &unwrapped_phase[static_cast<size_t>(y) * w_];
        std::copy(src_row, src_row + w_, dst_row);
    }

    return result;
}

double PcgPhaseUnwrapper::dot_product(const std::vector<float>& a, const std::vector<float>& b) const {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(pn_); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

} // namespace insar
