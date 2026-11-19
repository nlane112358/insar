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

    static double compute_coherent_rmse_mm(
        const std::vector<float>& unwrapped_phase,
        const std::vector<uint8_t>& valid_mask,
        const SyntheticTruth& truth,
        float wavelength_mm,
        float min_true_coherence = 0.5f);
};

} // namespace insar
