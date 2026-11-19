#include "insar/engine/phase_calibrator.hpp"

#include "insar/io/gdal_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace insar {

std::vector<float> PhaseCalibrator::to_displacement_mm(
    const std::vector<float>& unwrapped_phase,
    const std::vector<float>& coherence,
    const std::vector<uint8_t>& valid_mask,
    float wavelength_mm,
    float min_ref_coherence)
{
    GeoMetadata empty_meta;
    ReferenceRegionConfig disabled_ref;
    return to_displacement_mm(
        unwrapped_phase, coherence, valid_mask, empty_meta,
        wavelength_mm, min_ref_coherence, disabled_ref, nullptr, nullptr);
}

std::vector<float> PhaseCalibrator::to_displacement_mm(
    const std::vector<float>& unwrapped_phase,
    const std::vector<float>& coherence,
    const std::vector<uint8_t>& valid_mask,
    const GeoMetadata& metadata,
    float wavelength_mm,
    float min_ref_coherence,
    const ReferenceRegionConfig& ref_config,
    size_t* ref_pixels_out,
    bool* ref_pinned_out)
{
    const float rad_to_mm = wavelength_mm / (4.0f * math::kPi);

    std::vector<float> ref_samples;
    bool pinned = false;

    if (ref_config.enabled && metadata.width > 0 && metadata.height > 0 &&
        static_cast<size_t>(metadata.width) * static_cast<size_t>(metadata.height) == unwrapped_phase.size()) {
        double cx = static_cast<double>(ref_config.px);
        double cy = static_cast<double>(ref_config.py);
        double r_px = static_cast<double>(std::max(1, ref_config.radius_px));
        bool coords_ok = !ref_config.is_lonlat;

        if (ref_config.is_lonlat) {
            io::GeoCoordinateTransformer transformer(metadata);
            if (transformer.lonlat_to_pixel(ref_config.lon, ref_config.lat, cx, cy)) {
                r_px = std::max(1.0, transformer.meters_to_pixels(ref_config.radius_m, ref_config.lat));
                coords_ok = true;
            }
        }

        if (coords_ok) {
            const int x0 = std::max(0, static_cast<int>(std::floor(cx - r_px)));
            const int x1 = std::min(metadata.width - 1, static_cast<int>(std::ceil(cx + r_px)));
            const int y0 = std::max(0, static_cast<int>(std::floor(cy - r_px)));
            const int y1 = std::min(metadata.height - 1, static_cast<int>(std::ceil(cy + r_px)));
            const double r2 = r_px * r_px;

            for (int y = y0; y <= y1; ++y) {
                const double dy = (y + 0.5) - cy;
                for (int x = x0; x <= x1; ++x) {
                    const double dx = (x + 0.5) - cx;
                    if (dx * dx + dy * dy > r2) {
                        continue;
                    }
                    const size_t idx = static_cast<size_t>(y) * metadata.width + x;
                    if (valid_mask[idx] && std::isfinite(unwrapped_phase[idx]) &&
                        coherence[idx] >= min_ref_coherence) {
                        ref_samples.push_back(unwrapped_phase[idx]);
                    }
                }
            }
            // Relax coherence inside reference region if strict threshold found too few pixels
            if (ref_samples.size() < 3) {
                ref_samples.clear();
                for (int y = y0; y <= y1; ++y) {
                    const double dy = (y + 0.5) - cy;
                    for (int x = x0; x <= x1; ++x) {
                        const double dx = (x + 0.5) - cx;
                        if (dx * dx + dy * dy > r2) {
                            continue;
                        }
                        const size_t idx = static_cast<size_t>(y) * metadata.width + x;
                        if (valid_mask[idx] && std::isfinite(unwrapped_phase[idx]) &&
                            coherence[idx] >= 0.25f) {
                            ref_samples.push_back(unwrapped_phase[idx]);
                        }
                    }
                }
            }
            pinned = !ref_samples.empty();
            if (!pinned) {
                std::fprintf(stderr,
                    "Warning: Reference region contained 0 valid pixels; falling back to scene median.\n");
            }
        }
    }

    if (ref_samples.empty()) {
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
    }

    float median_ref = 0.0f;
    if (!ref_samples.empty()) {
        const size_t mid = ref_samples.size() / 2;
        std::nth_element(ref_samples.begin(), ref_samples.begin() + mid, ref_samples.end());
        median_ref = ref_samples[mid];
    }

    if (ref_pixels_out != nullptr) {
        *ref_pixels_out = ref_samples.size();
    }
    if (ref_pinned_out != nullptr) {
        *ref_pinned_out = pinned;
    }

    std::vector<float> disp_mm(unwrapped_phase.size(), 0.0f);
    for (size_t i = 0; i < unwrapped_phase.size(); ++i) {
        if (valid_mask[i]) {
            disp_mm[i] = (unwrapped_phase[i] - median_ref) * rad_to_mm;
        }
    }
    return disp_mm;
}

} // namespace insar
