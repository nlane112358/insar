#include "insar/io/gdal_utils.hpp"

#include <cpl_error.h>
#include <cpl_string.h>

#include <algorithm>
#include <cmath>
#include <mutex>

namespace insar::io {

void ensure_gdal_registered() {
    static std::once_flag flag;
    std::call_once(flag, []() { GDALAllRegister(); });
}

void GdalDatasetDeleter::operator()(GDALDataset* ds) const noexcept {
    if (ds) {
        GDALClose(ds);
    }
}

void OgrTransformDeleter::operator()(OGRCoordinateTransformation* ct) const noexcept {
    if (ct) {
        OGRCoordinateTransformation::DestroyCT(ct);
    }
}

CplQuietErrorScope::CplQuietErrorScope() {
    CPLPushErrorHandler(CPLQuietErrorHandler);
}

CplQuietErrorScope::~CplQuietErrorScope() {
    CPLPopErrorHandler();
}

CslStringList::~CslStringList() {
    if (list_) {
        CSLDestroy(list_);
    }
}

void CslStringList::set(const char* key, const char* value) {
    list_ = CSLSetNameValue(list_, key, value);
}

GeoCoordinateTransformer::GeoCoordinateTransformer(const GeoMetadata& meta)
    : meta_(meta)
{
    if (meta_.has_georef && !meta_.projection_wkt.empty()) {
        OGRSpatialReference src_srs;
        OGRSpatialReference dst_srs;
        if (src_srs.SetFromUserInput(meta_.projection_wkt.c_str()) == OGRERR_NONE) {
            src_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            dst_srs.SetWellKnownGeogCS("WGS84");
            dst_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            is_projected_ = src_srs.IsProjected() != 0;
            if (is_projected_) {
                const double lin = src_srs.GetLinearUnits(nullptr);
                linear_units_m_ = (lin > 0.0) ? lin : 1.0;
            }
            ct_.reset(OGRCreateCoordinateTransformation(&src_srs, &dst_srs));
            inv_ct_.reset(OGRCreateCoordinateTransformation(&dst_srs, &src_srs));
        }
    }
}

bool GeoCoordinateTransformer::pixel_to_lonlat(
    double px, double py, double& lon, double& lat) const
{
    if (!ct_) {
        return false;
    }
    double x = 0.0;
    double y = 0.0;
    meta_.pixel_to_map(px, py, x, y);
    if (!ct_->Transform(1, &x, &y)) {
        return false;
    }
    lon = x;
    lat = y;
    return std::isfinite(lon) && std::isfinite(lat);
}

bool GeoCoordinateTransformer::lonlat_to_pixel(
    double lon, double lat, double& px, double& py) const
{
    if (!inv_ct_) {
        return false;
    }
    double mx = lon;
    double my = lat;
    if (!inv_ct_->Transform(1, &mx, &my)) {
        return false;
    }
    const auto& gt = meta_.geotransform;
    const double det = gt[1] * gt[5] - gt[2] * gt[4];
    if (std::abs(det) < 1e-18) {
        return false;
    }
    const double dx = mx - gt[0];
    const double dy = my - gt[3];
    px = (gt[5] * dx - gt[2] * dy) / det;
    py = (-gt[4] * dx + gt[1] * dy) / det;
    return std::isfinite(px) && std::isfinite(py);
}

double GeoCoordinateTransformer::meters_to_pixels(double radius_m, double lat_deg) const {
    const auto& gt = meta_.geotransform;
    const double pix_x = std::hypot(gt[1], gt[4]);
    const double pix_y = std::hypot(gt[2], gt[5]);
    const double pix_units = std::max(1e-12, 0.5 * (pix_x + pix_y));
    if (is_projected_) {
        const double crs_units = radius_m / linear_units_m_;
        return crs_units / pix_units;
    }
    const double cos_lat = std::max(0.2, std::cos(lat_deg * 3.14159265358979323846 / 180.0));
    const double deg_units = radius_m / (111320.0 * cos_lat);
    return deg_units / pix_units;
}

Wgs84BBox GeoCoordinateTransformer::compute_wgs84_bbox() const {
    Wgs84BBox bbox;
    if (!ct_) {
        return bbox;
    }

    const double corners[4][2] = {
        {0.0, 0.0},
        {static_cast<double>(meta_.width), 0.0},
        {static_cast<double>(meta_.width), static_cast<double>(meta_.height)},
        {0.0, static_cast<double>(meta_.height)}
    };

    bbox.min_lon = 180.0;
    bbox.max_lon = -180.0;
    bbox.min_lat = 90.0;
    bbox.max_lat = -90.0;
    bbox.valid = true;

    for (const auto& c : corners) {
        double lon = 0.0;
        double lat = 0.0;
        if (pixel_to_lonlat(c[0], c[1], lon, lat)) {
            bbox.min_lon = std::min(bbox.min_lon, lon);
            bbox.max_lon = std::max(bbox.max_lon, lon);
            bbox.min_lat = std::min(bbox.min_lat, lat);
            bbox.max_lat = std::max(bbox.max_lat, lat);
        } else {
            bbox.valid = false;
        }
    }
    return bbox;
}

} // namespace insar::io
