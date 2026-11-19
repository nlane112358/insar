#pragma once

#include "insar/engine/preconditioner.hpp"
#include "insar/engine/weighted_laplacian.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace insar {

struct UnwrapResult {
    int iterations = 0;
    double rel_residual = 0.0;
};

class IPhaseUnwrapper {
public:
    virtual ~IPhaseUnwrapper() = default;

    virtual UnwrapResult unwrap(
        const std::vector<float>& wrapped_phase,
        const std::vector<float>& coherence,
        const std::vector<uint8_t>& valid_mask,
        std::vector<float>& unwrapped_phase,
        int max_iters,
        float tol,
        float min_edge_weight) = 0;
};

class PcgPhaseUnwrapper final : public IPhaseUnwrapper {
public:
    PcgPhaseUnwrapper(int out_width, int out_height);
    PcgPhaseUnwrapper(
        int out_width,
        int out_height,
        std::unique_ptr<IPreconditioner> preconditioner);

    UnwrapResult unwrap(
        const std::vector<float>& wrapped_phase,
        const std::vector<float>& coherence,
        const std::vector<uint8_t>& valid_mask,
        std::vector<float>& unwrapped_phase,
        int max_iters,
        float tol,
        float min_edge_weight) override;

    int padded_width() const noexcept { return pw_; }
    int padded_height() const noexcept { return ph_; }

private:
    double dot_product(const std::vector<float>& a, const std::vector<float>& b) const;

    int w_ = 0;
    int h_ = 0;
    int pw_ = 0;
    int ph_ = 0;
    size_t pn_ = 0;

    WeightedLaplacianSystem laplacian_;
    std::unique_ptr<IPreconditioner> preconditioner_;

    std::vector<float> phi_pad_;
    std::vector<float> r_;
    std::vector<float> z_;
    std::vector<float> p_;
    std::vector<float> Ap_;
};

} // namespace insar
