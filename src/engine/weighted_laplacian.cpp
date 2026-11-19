#include "insar/engine/weighted_laplacian.hpp"

#include "insar/core/types.hpp"

#include <algorithm>
#include <stdexcept>

namespace insar {

namespace {

struct EdgeTerm {
    float weight = 0.0f;
    float diff = 0.0f;
};

inline EdgeTerm evaluate_edge(
    int x0, int y0, bool v0,
    int x1, int y1, bool v1,
    int valid_w,
    const std::vector<float>& psi,
    const std::vector<float>& coh,
    float min_weight) noexcept
{
    if (v0 && v1) {
        const size_t i0 = static_cast<size_t>(y0) * valid_w + x0;
        const size_t i1 = static_cast<size_t>(y1) * valid_w + x1;
        const float c = std::min(coh[i0], coh[i1]);
        return {
            std::max(min_weight, c * c * c),
            math::wrap_phase(psi[i1] - psi[i0])
        };
    }
    return {min_weight, 0.0f};
}

} // namespace

WeightedLaplacianSystem::WeightedLaplacianSystem(int padded_width, int padded_height)
    : pw_(padded_width),
      ph_(padded_height),
      pn_(static_cast<size_t>(padded_width) * padded_height),
      wx_(pn_),
      wy_(pn_)
{
    if (pw_ <= 0 || ph_ <= 0) {
        throw std::invalid_argument("WeightedLaplacianSystem dimensions must be positive");
    }
}

void WeightedLaplacianSystem::build(
    int valid_width,
    int valid_height,
    const std::vector<float>& wrapped_phase,
    const std::vector<float>& coherence,
    const std::vector<uint8_t>& valid_mask,
    float min_weight,
    std::vector<float>& rhs)
{
    if (valid_width <= 0 || valid_height <= 0 || valid_width > pw_ || valid_height > ph_) {
        throw std::invalid_argument("WeightedLaplacianSystem valid dimensions out of bounds");
    }
    const size_t valid_n = static_cast<size_t>(valid_width) * valid_height;
    if (wrapped_phase.size() < valid_n || coherence.size() < valid_n || valid_mask.size() < valid_n) {
        throw std::invalid_argument("WeightedLaplacianSystem input buffer size mismatch");
    }
    if (rhs.size() < pn_) {
        rhs.resize(pn_);
    }

    auto is_valid = [&](int x, int y) noexcept -> bool {
        return (x >= 0 && x < valid_width && y >= 0 && y < valid_height) &&
               (valid_mask[static_cast<size_t>(y) * valid_width + x] != 0);
    };

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ph_; ++y) {
        for (int x = 0; x < pw_; ++x) {
            const size_t pidx = static_cast<size_t>(y) * pw_ + x;
            const bool v_here = is_valid(x, y);

            const EdgeTerm right = (x + 1 < pw_)
                ? evaluate_edge(x, y, v_here, x + 1, y, is_valid(x + 1, y),
                                valid_width, wrapped_phase, coherence, min_weight)
                : EdgeTerm{};
            wx_[pidx] = right.weight;

            const EdgeTerm down = (y + 1 < ph_)
                ? evaluate_edge(x, y, v_here, x, y + 1, is_valid(x, y + 1),
                                valid_width, wrapped_phase, coherence, min_weight)
                : EdgeTerm{};
            wy_[pidx] = down.weight;

            const EdgeTerm left = (x > 0)
                ? evaluate_edge(x - 1, y, is_valid(x - 1, y), x, y, v_here,
                                valid_width, wrapped_phase, coherence, min_weight)
                : EdgeTerm{};

            const EdgeTerm up = (y > 0)
                ? evaluate_edge(x, y - 1, is_valid(x, y - 1), x, y, v_here,
                                valid_width, wrapped_phase, coherence, min_weight)
                : EdgeTerm{};

            rhs[pidx] = (-right.weight * right.diff + left.weight * left.diff) +
                        (-down.weight * down.diff + up.weight * up.diff);
        }
    }
}

void WeightedLaplacianSystem::apply(const std::vector<float>& p, std::vector<float>& Ap) const {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ph_; ++y) {
        for (int x = 0; x < pw_; ++x) {
            const size_t idx = static_cast<size_t>(y) * pw_ + x;
            const float center = p[idx];
            float acc = 0.0f;

            if (x + 1 < pw_) acc += wx_[idx] * (center - p[idx + 1]);
            if (x > 0)       acc += wx_[idx - 1] * (center - p[idx - 1]);
            if (y + 1 < ph_) acc += wy_[idx] * (center - p[idx + pw_]);
            if (y > 0)       acc += wy_[idx - pw_] * (center - p[idx - pw_]);

            Ap[idx] = acc;
        }
    }
}

} // namespace insar
