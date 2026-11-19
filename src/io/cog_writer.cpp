#include "insar/io/cog_writer.hpp"

#include "insar/io/gdal_utils.hpp"

#include <cpl_conv.h>

#include <cstdio>
#include <limits>
#include <vector>

namespace insar::io {

bool GdalCogWriter::write(
    const std::string& path,
    const std::vector<float>& data,
    const std::vector<uint8_t>& valid_mask,
    const GeoMetadata& meta) const
{
    if (meta.width <= 0 || meta.height <= 0 ||
        data.size() != meta.pixel_count() || valid_mask.size() != meta.pixel_count()) {
        std::fprintf(stderr, "Error: Raster data size does not match metadata dimensions.\n");
        return false;
    }

    ensure_gdal_registered();

    GDALDriver* mem_drv = GetGDALDriverManager()->GetDriverByName("MEM");
    if (!mem_drv) {
        std::fprintf(stderr, "Error: GDAL MEM driver unavailable.\n");
        return false;
    }

    UniqueGdalDataset mem_ds(mem_drv->Create(
        "", meta.width, meta.height, 1, GDT_Float32, nullptr));
    if (!mem_ds) {
        return false;
    }

    if (meta.has_georef) {
        auto gt = meta.geotransform;
        mem_ds->SetGeoTransform(gt.data());
    }
    if (!meta.projection_wkt.empty()) {
        OGRSpatialReference srs;
        if (srs.SetFromUserInput(meta.projection_wkt.c_str()) == OGRERR_NONE) {
            char* wkt_out = nullptr;
            srs.exportToWkt(&wkt_out);
            if (wkt_out) {
                mem_ds->SetProjection(wkt_out);
                CPLFree(wkt_out);
            }
        }
    }

    const float nan_f = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> masked(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        masked[i] = valid_mask[i] ? data[i] : nan_f;
    }

    GDALRasterBand* band = mem_ds->GetRasterBand(1);
    band->SetNoDataValue(std::numeric_limits<double>::quiet_NaN());
    if (band->RasterIO(GF_Write, 0, 0, meta.width, meta.height,
                       masked.data(), meta.width, meta.height,
                       GDT_Float32, 0, 0) != CE_None) {
        return false;
    }

    GDALDriver* out_drv = GetGDALDriverManager()->GetDriverByName("COG");
    CslStringList options;
    if (out_drv) {
        options.set("COMPRESS", "DEFLATE");
        options.set("PREDICTOR", "YES");
        options.set("BLOCKSIZE", "512");
        options.set("NUM_THREADS", "ALL_CPUS");
        options.set("BIGTIFF", "IF_SAFER");
    } else {
        out_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!out_drv) {
            std::fprintf(stderr, "Error: Neither GDAL COG nor GTiff driver available.\n");
            return false;
        }
        options.set("TILED", "YES");
        options.set("COMPRESS", "DEFLATE");
        options.set("PREDICTOR", "3");
        options.set("BIGTIFF", "IF_SAFER");
    }

    UniqueGdalDataset out_ds(out_drv->CreateCopy(
        path.c_str(), mem_ds.get(), FALSE, options.raw(), nullptr, nullptr));
    if (!out_ds) {
        std::fprintf(stderr, "Error: Could not create COG '%s'.\n", path.c_str());
        return false;
    }
    return true;
}

} // namespace insar::io
