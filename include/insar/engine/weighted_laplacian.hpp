#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace insar {

class WeightedLaplacianSystem {
public:
    WeightedLaplacianSystem(int padded_width, int padded_height);

    void build(
        int valid_width,
        int valid_height,
        const std::vector<float>& wrapped_phase,
        const std::vector<float>& coherence,
        const std::vector<uint8_t>& valid_mask,
        float min_weight,
        std::vector<float>& rhs);

    void apply(const std::vector<float>& p, std::vector<float>& Ap) const;

    int padded_width() const noexcept { return pw_; }
    int padded_height() const noexcept { return ph_; }
    size_t padded_size() const noexcept { return pn_; }

private:
    int pw_ = 0;
    int ph_ = 0;
    size_t pn_ = 0;
    std::vector<float> wx_;
    std::vector<float> wy_;
};

} // namespace insar
