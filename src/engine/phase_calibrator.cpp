#include "insar/engine/phase_calibrator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace insar {

std::vector<float> PhaseCalibrator::to_displacement_mm(
    const std::vector<float>& unwrapped_phase,
    const std::vector<float>& coherence,
    const std::vector<uint8_t>& valid_mask,
    float wavelength_mm,
    float min_ref_coherence)
{
    const float rad_to_mm = wavelength_mm / (4.0f * math::kPi);

    std::vector<float> ref_samples;
    ref_samples.reserve(unwrapped_phase.size());
    for (size_t i = 0; i < unwrapped_phase.size(); ++i) {
        if (valid_mask[i] && coherence[i] > min_ref_coherence) {
            ref_samples.push_back(unwrapped_phase[i]);
        }
    }
    if (ref_samples.empty()) {
        for (size_t i = 0; i < unwrapped_phase.size(); ++i) {
            if (valid_mask[i]) {
                ref_samples.push_back(unwrapped_phase[i]);
            }
        }
    }

    float median_ref = 0.0f;
    if (!ref_samples.empty()) {
        const size_t mid = ref_samples.size() / 2;
        std::nth_element(ref_samples.begin(), ref_samples.begin() + mid, ref_samples.end());
        median_ref = ref_samples[mid];
    }

    std::vector<float> disp_mm(unwrapped_phase.size(), 0.0f);
    for (size_t i = 0; i < unwrapped_phase.size(); ++i) {
        if (valid_mask[i]) {
            disp_mm[i] = (unwrapped_phase[i] - median_ref) * rad_to_mm;
        }
    }
    return disp_mm;
}

double PhaseCalibrator::compute_coherent_rmse_mm(
    const std::vector<float>& unwrapped_phase,
    const std::vector<uint8_t>& valid_mask,
    const SyntheticTruth& truth,
    float wavelength_mm,
    float min_true_coherence)
{
    const float rad_to_mm = wavelength_mm / (4.0f * math::kPi);
    double mean_diff = 0.0;
    size_t valid_count = 0;

    for (size_t i = 0; i < unwrapped_phase.size(); ++i) {
        if (valid_mask[i] && truth.coherence[i] > min_true_coherence) {
            mean_diff += (unwrapped_phase[i] - truth.phase_rad[i]);
            ++valid_count;
        }
    }
    if (valid_count == 0) {
        return 0.0;
    }
    mean_diff /= static_cast<double>(valid_count);

    double sq_err_mm = 0.0;
    for (size_t i = 0; i < unwrapped_phase.size(); ++i) {
        if (valid_mask[i] && truth.coherence[i] > min_true_coherence) {
            const double err_mm = ((unwrapped_phase[i] - truth.phase_rad[i]) - mean_diff) * rad_to_mm;
            sq_err_mm += err_mm * err_mm;
        }
    }
    return std::sqrt(sq_err_mm / static_cast<double>(valid_count));
}

} // namespace insar
