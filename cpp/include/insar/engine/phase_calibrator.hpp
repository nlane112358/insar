#pragma once

#include "insar/core/types.hpp"

#include <cstdint>
#include <vector>

namespace insar {

class PhaseCalibrator {
public:
    static std::vector<float> to_displacement_mm(
        const std::vector<float>& unwrapped_phase,
        const std::vector<float>& coherence,
        const std::vector<uint8_t>& valid_mask,
        float wavelength_mm,
        float min_ref_coherence);

    static std::vector<float> to_displacement_mm(
        const std::vector<float>& unwrapped_phase,
        const std::vector<float>& coherence,
        const std::vector<uint8_t>& valid_mask,
        const GeoMetadata& metadata,
        float wavelength_mm,
        float min_ref_coherence,
        const ReferenceRegionConfig& ref_config,
        size_t* ref_pixels_out = nullptr,
        bool* ref_pinned_out = nullptr);
};

} // namespace insar
