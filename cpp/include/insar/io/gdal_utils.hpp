#pragma once

#include "insar/core/types.hpp"

#include <gdal_priv.h>
#include <ogr_feature.h>
#include <ogr_geometry.h>
#include <ogr_spatialref.h>

#include <memory>

namespace insar::io {

void ensure_gdal_registered();

struct GdalDatasetDeleter {
    void operator()(GDALDataset* ds) const noexcept;
};
using UniqueGdalDataset = std::unique_ptr<GDALDataset, GdalDatasetDeleter>;

struct OgrTransformDeleter {
    void operator()(OGRCoordinateTransformation* ct) const noexcept;
};
using UniqueOgrTransform = std::unique_ptr<OGRCoordinateTransformation, OgrTransformDeleter>;

struct OgrGeometryDeleter {
    void operator()(OGRGeometry* g) const noexcept;
};
using UniqueOgrGeom = std::unique_ptr<OGRGeometry, OgrGeometryDeleter>;

struct OgrFeatureDeleter {
    void operator()(OGRFeature* f) const noexcept;
};
using UniqueOgrFeature = std::unique_ptr<OGRFeature, OgrFeatureDeleter>;

double meters_to_crs_units(const OGRSpatialReference& srs, double meters, double center_lat_deg);

class CplQuietErrorScope {
public:
    CplQuietErrorScope();
    ~CplQuietErrorScope();

    CplQuietErrorScope(const CplQuietErrorScope&) = delete;
    CplQuietErrorScope& operator=(const CplQuietErrorScope&) = delete;
};

class CslStringList {
public:
    CslStringList() = default;
    ~CslStringList();

    CslStringList(const CslStringList&) = delete;
    CslStringList& operator=(const CslStringList&) = delete;

    void set(const char* key, const char* value);
    char** raw() const noexcept { return list_; }

private:
    char** list_ = nullptr;
};

class GeoCoordinateTransformer {
public:
    explicit GeoCoordinateTransformer(const GeoMetadata& meta);

    bool is_valid() const noexcept { return ct_ != nullptr; }
    bool is_projected() const noexcept { return is_projected_; }
    bool pixel_to_lonlat(double px, double py, double& lon, double& lat) const;
    bool lonlat_to_pixel(double lon, double lat, double& px, double& py) const;
    double meters_to_pixels(double radius_m, double lat_deg) const;
    Wgs84BBox compute_wgs84_bbox() const;

private:
    GeoMetadata meta_;
    bool is_projected_ = false;
    double linear_units_m_ = 1.0;
    UniqueOgrTransform ct_;
    UniqueOgrTransform inv_ct_;
};

} // namespace insar::io
