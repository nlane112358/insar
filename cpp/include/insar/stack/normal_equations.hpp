#pragma once

#include "insar/stack/stack_types.hpp"

#include <cstddef>
#include <vector>

namespace insar::stack {

void update_state_incremental(
    std::vector<double>& state,
    size_t npix,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    double t0_days,
    double t1_days,
    double origin_days,
    float min_coh,
    bool redundant_pair,
    double weight_scale);

DerivedFields derive_products(
    const std::vector<double>& state,
    size_t npix,
    const StackMetadata& smeta,
    float min_coh);

} // namespace insar::stack
