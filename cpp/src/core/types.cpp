#include "insar/core/types.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace insar {

size_t GeoMetadata::pixel_count() const noexcept {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    return static_cast<size_t>(width) * static_cast<size_t>(height);
}

void GeoMetadata::pixel_to_map(double px, double py, double& map_x, double& map_y) const noexcept {
    map_x = geotransform[0] + px * geotransform[1] + py * geotransform[2];
    map_y = geotransform[3] + px * geotransform[4] + py * geotransform[5];
}

GeoMetadata GeoMetadata::cropped(int x_off, int y_off, int win_w, int win_h) const {
    GeoMetadata out = *this;
    out.width = win_w;
    out.height = win_h;
    if (out.has_georef) {
        double orig_x = 0.0;
        double orig_y = 0.0;
        pixel_to_map(static_cast<double>(x_off), static_cast<double>(y_off), orig_x, orig_y);
        out.geotransform[0] = orig_x;
        out.geotransform[3] = orig_y;
    }
    return out;
}

GeoMetadata GeoMetadata::multilooked(int out_w, int out_h, int stride_x, int stride_y) const {
    GeoMetadata out = *this;
    out.width = out_w;
    out.height = out_h;
    if (out.has_georef) {
        out.geotransform[1] *= stride_x;
        out.geotransform[2] *= stride_y;
        out.geotransform[4] *= stride_x;
        out.geotransform[5] *= stride_y;
    }
    return out;
}

SlcRaster::SlcRaster(GeoMetadata metadata, std::vector<Complex> pixels)
    : metadata_(std::move(metadata)), pixels_(std::move(pixels)) {
    const size_t expected = metadata_.pixel_count();
    if (metadata_.width <= 0 || metadata_.height <= 0) {
        throw std::invalid_argument("SlcRaster dimensions must be positive");
    }
    if (pixels_.size() != expected) {
        throw std::invalid_argument("SlcRaster pixel buffer size does not match metadata dimensions");
    }
}

void SlcPair::validate() const {
    if (reference.width() <= 0 || reference.height() <= 0) {
        throw std::runtime_error("Reference SLC has invalid dimensions");
    }
    if (reference.width() != secondary.width() || reference.height() != secondary.height()) {
        std::ostringstream oss;
        oss << "SLC dimensions mismatch (" << reference.width() << "x" << reference.height()
            << " vs " << secondary.width() << "x" << secondary.height()
            << "). Verify both files share the same OPERA Burst ID.";
        throw std::runtime_error(oss.str());
    }
}

} // namespace insar
