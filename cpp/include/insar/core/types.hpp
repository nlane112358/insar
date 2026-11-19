#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace insar {

using Complex = std::complex<float>;

namespace math {

inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kTwoPi = 6.28318530717958647692f;

inline float wrap_phase(float delta) noexcept {
    delta = std::fmod(delta + kPi, kTwoPi);
    if (delta < 0.0f) {
        delta += kTwoPi;
    }
    return delta - kPi;
}

inline int next_power_of_two(int value) noexcept {
    int p = 1;
    while (p < value) {
        p <<= 1;
    }
    return p;
}

inline bool is_valid_complex(const Complex& z) noexcept {
    const float re = z.real();
    const float im = z.imag();
    return std::isfinite(re) && std::isfinite(im) && (re * re + im * im > 0.0f);
}

} // namespace math

struct GeoMetadata {
    int width = 0;
    int height = 0;
    bool has_georef = false;
    std::array<double, 6> geotransform = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
    std::string projection_wkt;

    size_t pixel_count() const noexcept;
    void pixel_to_map(double px, double py, double& map_x, double& map_y) const noexcept;
    GeoMetadata cropped(int x_off, int y_off, int win_w, int win_h) const;
    GeoMetadata multilooked(int out_w, int out_h, int stride_x, int stride_y) const;
};

struct CropWindow {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct Wgs84BBox {
    bool valid = false;
    double min_lon = 0.0;
    double min_lat = 0.0;
    double max_lon = 0.0;
    double max_lat = 0.0;
};

class SlcRaster {
public:
    SlcRaster() = default;
    SlcRaster(GeoMetadata metadata, std::vector<Complex> pixels);

    int width() const noexcept { return metadata_.width; }
    int height() const noexcept { return metadata_.height; }
    size_t size() const noexcept { return pixels_.size(); }

    const GeoMetadata& metadata() const noexcept { return metadata_; }
    GeoMetadata& metadata() noexcept { return metadata_; }

    const std::vector<Complex>& pixels() const noexcept { return pixels_; }
    std::vector<Complex>& pixels() noexcept { return pixels_; }

    const Complex& at(int x, int y) const noexcept {
        return pixels_[static_cast<size_t>(y) * metadata_.width + x];
    }

private:
    GeoMetadata metadata_;
    std::vector<Complex> pixels_;
};

struct SlcPair {
    SlcRaster reference;
    SlcRaster secondary;

    int width() const noexcept { return reference.width(); }
    int height() const noexcept { return reference.height(); }

    void validate() const;
};

struct ReferenceRegionConfig {
    bool enabled = false;
    bool is_lonlat = false;
    double lon = 0.0;
    double lat = 0.0;
    double radius_m = 1000.0;
    int px = 0;
    int py = 0;
    int radius_px = 10;
};

struct ProcessingConfig {
    int stride_x = 1;
    int stride_y = 1;
    int look_rx = 3;
    int look_ry = 3;
    int max_iters = 40;
    float tolerance = 1e-4f;
    float min_edge_weight = 1e-5f;
    float wavelength_mm = 55.4658f;
    float min_ref_coherence = 0.65f;
    ReferenceRegionConfig reference_region;
};

struct ProcessingStats {
    double ifg_ms = 0.0;
    double unwrap_ms = 0.0;
    int iterations = 0;
    double rel_residual = 0.0;
    size_t valid_pixels = 0;
    size_t ref_pixels_used = 0;
    bool ref_region_pinned = false;
};

struct InsarProducts {
    GeoMetadata metadata;
    std::vector<float> wrapped_phase;
    std::vector<float> coherence;
    std::vector<float> unwrapped_phase;
    std::vector<float> displacement_mm;
    std::vector<uint8_t> valid_mask;
    ProcessingStats stats;
};

} // namespace insar
