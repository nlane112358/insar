#pragma once

#include "insar/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace insar {

class InterferogramEstimator {
public:
    size_t compute(
        const SlcPair& pair,
        int out_w,
        int out_h,
        int stride_x,
        int stride_y,
        int look_rx,
        int look_ry,
        std::vector<float>& wrapped,
        std::vector<float>& coherence,
        std::vector<uint8_t>& valid_mask) const;
};

} // namespace insar
