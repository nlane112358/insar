#include "insar/io/synthetic_slc_source.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace insar {

SyntheticSlcSource::SyntheticSlcSource(
    int width,
    int height,
    float wavelength_mm,
    uint32_t seed)
    : width_(width),
      height_(height),
      wavelength_mm_(wavelength_mm),
      seed_(seed)
{
    if (width_ <= 0 || height_ <= 0) {
        throw std::invalid_argument("Synthetic scene dimensions must be positive");
    }
    if (wavelength_mm_ <= 0.0f) {
        throw std::invalid_argument("Radar wavelength must be positive");
    }
}

SlcPair SyntheticSlcSource::load() {
    const size_t n = static_cast<size_t>(width_) * height_;
    std::vector<Complex> slc1(n);
    std::vector<Complex> slc2(n);

    SyntheticTruth truth;
    truth.phase_rad.resize(n);
    truth.coherence.resize(n);

    GeoMetadata meta;
    meta.width = width_;
    meta.height = height_;
    meta.has_georef = true;
    // Arbitrary UTM Zone 14N geotransform (Mexico City area) for QGIS sanity testing
    meta.geotransform = {480000.0, 10.0, 0.0, 2155000.0, 0.0, -10.0};
    meta.projection_wkt = "EPSG:32614";

    std::mt19937 rng(seed_);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const float mm_to_rad = (4.0f * math::kPi) / wavelength_mm_;
    const float nan_f = std::numeric_limits<float>::quiet_NaN();

    for (int y = 0; y < height_; ++y) {
        const float yn = (static_cast<float>(y) / height_) - 0.5f;
        for (int x = 0; x < width_; ++x) {
            const float xn = (static_cast<float>(x) / width_) - 0.5f;
            const size_t idx = static_cast<size_t>(y) * width_ + x;

            // Simulate a skewed satellite burst border with NaNs
            const float skewed_y = yn + 0.15f * xn;
            if (std::abs(skewed_y) > 0.42f || std::abs(xn) > 0.46f) {
                slc1[idx] = Complex(nan_f, nan_f);
                slc2[idx] = Complex(nan_f, nan_f);
                truth.phase_rad[idx] = nan_f;
                truth.coherence[idx] = 0.0f;
                continue;
            }

            const float r1 = (xn + 0.12f) * (xn + 0.12f) / 0.018f +
                             (yn - 0.05f) * (yn - 0.05f) / 0.014f;
            const float r2 = (xn - 0.20f) * (xn - 0.20f) / 0.008f +
                             (yn + 0.18f) * (yn + 0.18f) / 0.010f;
            const float disp_mm =
                -95.0f * std::exp(-r1) - 40.0f * std::exp(-r2) + 12.0f * xn - 8.0f * yn;
            const float phi = disp_mm * mm_to_rad;
            truth.phase_rad[idx] = phi;

            const float r_lake = (xn - 0.15f) * (xn - 0.15f) / 0.012f +
                                 (yn + 0.15f) * (yn + 0.15f) / 0.012f;
            const float rho = (r_lake < 1.0f) ? 0.15f : 0.90f;
            truth.coherence[idx] = rho;

            const float a_re = gauss(rng);
            const float a_im = gauss(rng);
            const float n_re = gauss(rng);
            const float n_im = gauss(rng);

            const float c = std::cos(phi);
            const float s = std::sin(phi);
            const float rot_re = a_re * c + a_im * s;
            const float rot_im = a_im * c - a_re * s;
            const float noise_scale = std::sqrt(1.0f - rho * rho);

            slc1[idx] = Complex(a_re, a_im);
            slc2[idx] = Complex(rho * rot_re + noise_scale * n_re,
                                rho * rot_im + noise_scale * n_im);
        }
    }

    truth_ = std::move(truth);

    SlcRaster ref(meta, std::move(slc1));
    SlcRaster sec(meta, std::move(slc2));
    return SlcPair{std::move(ref), std::move(sec)};
}

std::string SyntheticSlcSource::describe(int threads) const {
    std::ostringstream oss;
    oss << "Running synthetic mode (" << width_ << "x" << height_
        << ", skewed NaN border) on " << threads << " OpenMP threads";
    return oss.str();
}

} // namespace insar
