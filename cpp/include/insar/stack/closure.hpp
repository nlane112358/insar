#pragma once

#include "insar/stack/stack_types.hpp"

#include <cstdint>
#include <vector>

namespace insar::stack {

ClosureResult evaluate_triplet_closure(
    const StackOptions& opt,
    const RasterMeta& meta,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& ref_mask,
    std::vector<uint8_t>& bad_mask_out,
    std::vector<float>& residual_map_out);

} // namespace insar::stack
