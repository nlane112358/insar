#pragma once

#include "insar/stack/stack_types.hpp"

#include <cstdint>
#include <vector>

namespace insar::stack {

std::vector<uint8_t> dilate_mask_box(
    const std::vector<uint8_t>& src,
    int width,
    int height,
    int radius);

std::vector<uint8_t> build_reference_mask(
    const StackOptions& opt,
    const RasterMeta& meta,
    ReferencePinResult& ref_info);

void apply_reference_pinning(
    std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& ref_mask,
    float min_coh,
    ReferencePinResult& ref_info);

std::vector<uint8_t> extract_reference_connected_component(
    int width,
    int height,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& ref_mask,
    float min_coh);

std::vector<uint8_t> build_deform_exclusion_mask(
    const StackOptions& opt,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const std::vector<double>& state_bands,
    const std::vector<uint8_t>& ref_mask,
    const std::vector<uint8_t>& ref_cc,
    const std::vector<float>& disp,
    const std::vector<float>& coh);

RampResult remove_robust_ramp_and_dem(
    int width,
    int height,
    std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<float>& dem,
    const std::vector<uint8_t>& deform_excl_in,
    const std::vector<uint8_t>& ref_mask,
    const std::vector<uint8_t>& ref_cc,
    int order,
    float min_coh,
    float ramp_mask_sigma,
    int ramp_mask_dilate_px);

} // namespace insar::stack
