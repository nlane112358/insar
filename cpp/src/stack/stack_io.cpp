#include "insar/stack/stack_io.hpp"

#include "insar/io/gdal_utils.hpp"

#include <cpl_conv.h>
#include <cpl_string.h>
#include <ogr_api.h>
#include <ogr_spatialref.h>
#include <ogrsf_frmts.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace insar::stack {

using insar::io::CplQuietErrorScope;
using insar::io::CslStringList;
using insar::io::UniqueGdalDataset;
using insar::io::UniqueOgrFeature;
using insar::io::UniqueOgrGeom;
using insar::io::UniqueOgrTransform;

std::vector<float> read_single_band_f32(const std::string& path, RasterMeta& meta_out) {
    insar::io::ensure_gdal_registered();
    UniqueGdalDataset ds(static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly)));
    if (!ds) {
        throw std::runtime_error("Failed to open raster: " + path);
    }
    meta_out.width = ds->GetRasterXSize();
    meta_out.height = ds->GetRasterYSize();
    if (ds->GetGeoTransform(meta_out.geotransform) != CE_None) {
        throw std::runtime_error("Raster has no GeoTransform: " + path);
    }
    const char* proj = ds->GetProjectionRef();
    meta_out.wkt = (proj != nullptr) ? proj : "";
    GDALRasterBand* band = ds->GetRasterBand(1);
    if (band == nullptr) {
        throw std::runtime_error("Raster missing band 1: " + path);
    }
    int has_nodata = 0;
    double nodata = band->GetNoDataValue(&has_nodata);

    const size_t n = static_cast<size_t>(meta_out.width) * static_cast<size_t>(meta_out.height);
    std::vector<float> buf(n, kNanF);
    CPLErr err = band->RasterIO(
        GF_Read, 0, 0, meta_out.width, meta_out.height,
        buf.data(), meta_out.width, meta_out.height,
        GDT_Float32, 0, 0, nullptr);
    if (err != CE_None) {
        throw std::runtime_error("RasterIO read failed on: " + path);
    }
    if (has_nodata && std::isfinite(nodata)) {
        const float nd = static_cast<float>(nodata);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            if (buf[i] == nd) {
                buf[i] = kNanF;
            }
        }
    }
    return buf;
}

bool grids_match(const RasterMeta& a, const RasterMeta& b) {
    if (a.width != b.width || a.height != b.height) {
        return false;
    }
    for (int k = 0; k < 6; ++k) {
        double tol = std::max(1e-4, std::abs(a.geotransform[k]) * 1e-6);
        if (std::abs(a.geotransform[k] - b.geotransform[k]) > tol) {
            return false;
        }
    }
    return true;
}

void assert_compatible_grid(const RasterMeta& a, const RasterMeta& b, const char* label) {
    if (a.width != b.width || a.height != b.height) {
        throw std::runtime_error(
            std::string("Dimension mismatch for ") + label + ": expected " +
            std::to_string(a.width) + "x" + std::to_string(a.height) + ", got " +
            std::to_string(b.width) + "x" + std::to_string(b.height));
    }
    for (int k = 0; k < 6; ++k) {
        double tol = std::max(1e-4, std::abs(a.geotransform[k]) * 1e-6);
        if (std::abs(a.geotransform[k] - b.geotransform[k]) > tol) {
            throw std::runtime_error(std::string("GeoTransform mismatch for ") + label);
        }
    }
}

std::vector<float> read_raster_aligned_f32(
    const std::string& path,
    const RasterMeta& target_meta,
    GDALResampleAlg resample_alg
) {
    insar::io::ensure_gdal_registered();
    UniqueGdalDataset src_ds(static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly)));
    if (!src_ds) {
        throw std::runtime_error("Failed to open auxiliary raster: " + path);
    }
    RasterMeta src_meta;
    src_meta.width = src_ds->GetRasterXSize();
    src_meta.height = src_ds->GetRasterYSize();
    const bool has_gt = (src_ds->GetGeoTransform(src_meta.geotransform) == CE_None);
    const char* src_proj = src_ds->GetProjectionRef();
    src_meta.wkt = (src_proj != nullptr) ? src_proj : "";

    bool same_crs = (src_meta.wkt == target_meta.wkt);
    if (!same_crs && !src_meta.wkt.empty() && !target_meta.wkt.empty()) {
        OGRSpatialReference s1, s2;
        if (s1.importFromWkt(src_meta.wkt.c_str()) == OGRERR_NONE &&
            s2.importFromWkt(target_meta.wkt.c_str()) == OGRERR_NONE) {
            same_crs = (s1.IsSame(&s2) != 0);
        }
    }
    if (has_gt && grids_match(src_meta, target_meta) && (same_crs || src_meta.wkt.empty() || target_meta.wkt.empty())) {
        src_ds.reset();
        RasterMeta dummy;
        return read_single_band_f32(path, dummy);
    }

    GDALDriver* mem_drv = GetGDALDriverManager()->GetDriverByName("MEM");
    if (mem_drv == nullptr) {
        throw std::runtime_error("GDAL MEM driver unavailable for reprojection");
    }
    UniqueGdalDataset dst_ds(mem_drv->Create("", target_meta.width, target_meta.height, 1, GDT_Float32, nullptr));
    if (!dst_ds) {
        throw std::runtime_error("Failed creating MEM dataset for aligned raster: " + path);
    }
    double gt[6];
    std::memcpy(gt, target_meta.geotransform, sizeof(gt));
    dst_ds->SetGeoTransform(gt);
    if (!target_meta.wkt.empty()) {
        dst_ds->SetProjection(target_meta.wkt.c_str());
    }

    const size_t npix = static_cast<size_t>(target_meta.width) * static_cast<size_t>(target_meta.height);
    std::vector<float> buf(npix, kNanF);
    GDALRasterBand* dst_band = dst_ds->GetRasterBand(1);
    dst_band->SetNoDataValue(kNanF);
    if (dst_band->RasterIO(
            GF_Write, 0, 0, target_meta.width, target_meta.height,
            buf.data(), target_meta.width, target_meta.height,
            GDT_Float32, 0, 0, nullptr) != CE_None) {
        throw std::runtime_error("Failed initializing aligned raster buffer");
    }

    GDALWarpOptions* warp_opts = GDALCreateWarpOptions();
    warp_opts->hSrcDS = src_ds.get();
    warp_opts->hDstDS = dst_ds.get();
    warp_opts->nBandCount = 1;
    warp_opts->panSrcBands = static_cast<int*>(CPLMalloc(sizeof(int)));
    warp_opts->panDstBands = static_cast<int*>(CPLMalloc(sizeof(int)));
    warp_opts->panSrcBands[0] = 1;
    warp_opts->panDstBands[0] = 1;
    warp_opts->eResampleAlg = resample_alg;
    warp_opts->eWorkingDataType = GDT_Float32;
    warp_opts->pfnProgress = GDALDummyProgress;

    int has_src_nd = 0;
    double src_nd = src_ds->GetRasterBand(1)->GetNoDataValue(&has_src_nd);
    if (has_src_nd) {
        warp_opts->padfSrcNoDataReal = static_cast<double*>(CPLMalloc(sizeof(double)));
        warp_opts->padfSrcNoDataReal[0] = src_nd;
    }
    warp_opts->padfDstNoDataReal = static_cast<double*>(CPLMalloc(sizeof(double)));
    warp_opts->padfDstNoDataReal[0] = kNanD;

    CslStringList warp_str_opts;
    warp_str_opts.set("INIT_DEST", "NO_DATA");
    warp_str_opts.set("NUM_THREADS", "ALL_CPUS");
    warp_opts->papszWarpOptions = warp_str_opts.raw();

    const char* effective_src_wkt = src_meta.wkt.empty() ? nullptr : src_meta.wkt.c_str();
    const char* effective_dst_wkt = target_meta.wkt.empty() ? nullptr : target_meta.wkt.c_str();
    warp_opts->pTransformerArg = GDALCreateGenImgProjTransformer(
        src_ds.get(), effective_src_wkt, dst_ds.get(), effective_dst_wkt, FALSE, 0.0, 1);
    if (warp_opts->pTransformerArg == nullptr) {
        warp_opts->papszWarpOptions = nullptr;
        GDALDestroyWarpOptions(warp_opts);
        throw std::runtime_error("Failed creating GDAL image projection transformer for: " + path);
    }
    warp_opts->pfnTransformer = GDALGenImgProjTransform;

    GDALWarpOperation warp_op;
    CPLErr init_err = warp_op.Initialize(warp_opts);
    CPLErr warp_err = CE_Failure;
    if (init_err == CE_None) {
        warp_err = warp_op.ChunkAndWarpImage(0, 0, target_meta.width, target_meta.height);
    }
    GDALDestroyGenImgProjTransformer(warp_opts->pTransformerArg);
    warp_opts->pTransformerArg = nullptr;
    warp_opts->papszWarpOptions = nullptr;
    GDALDestroyWarpOptions(warp_opts);
    if (init_err != CE_None || warp_err != CE_None) {
        throw std::runtime_error("GDAL warp failed while aligning raster: " + path);
    }

    if (dst_band->RasterIO(
            GF_Read, 0, 0, target_meta.width, target_meta.height,
            buf.data(), target_meta.width, target_meta.height,
            GDT_Float32, 0, 0, nullptr) != CE_None) {
        throw std::runtime_error("Failed reading warped raster buffer: " + path);
    }
    return buf;
}

void write_cog_f32(
    const fs::path& out_path,
    const RasterMeta& meta,
    const std::vector<float>& data,
    const char* description
) {
    insar::io::ensure_gdal_registered();
    GDALDriver* mem_drv = GetGDALDriverManager()->GetDriverByName("MEM");
    if (mem_drv == nullptr) {
        throw std::runtime_error("GDAL MEM driver unavailable");
    }
    UniqueGdalDataset mem_ds(mem_drv->Create("", meta.width, meta.height, 1, GDT_Float32, nullptr));
    if (!mem_ds) {
        throw std::runtime_error("Failed to create MEM dataset for " + out_path.string());
    }
    double gt[6];
    std::memcpy(gt, meta.geotransform, sizeof(gt));
    mem_ds->SetGeoTransform(gt);
    if (!meta.wkt.empty()) {
        mem_ds->SetProjection(meta.wkt.c_str());
    }
    GDALRasterBand* band = mem_ds->GetRasterBand(1);
    band->SetDescription(description);
    band->SetNoDataValue(kNanF);
    CPLErr io_err = band->RasterIO(
        GF_Write, 0, 0, meta.width, meta.height,
        const_cast<float*>(data.data()), meta.width, meta.height,
        GDT_Float32, 0, 0, nullptr);
    if (io_err != CE_None) {
        throw std::runtime_error("MEM RasterIO write failed for " + out_path.string());
    }

    GDALDriver* cog_drv = GetGDALDriverManager()->GetDriverByName("COG");
    if (cog_drv != nullptr) {
        CslStringList opts;
        opts.set("COMPRESS", "DEFLATE");
        opts.set("PREDICTOR", "YES");
        opts.set("BLOCKSIZE", "512");
        opts.set("NUM_THREADS", "ALL_CPUS");
        opts.set("BIGTIFF", "IF_SAFER");
        UniqueGdalDataset out_ds(cog_drv->CreateCopy(out_path.c_str(), mem_ds.get(), FALSE, opts.raw(), nullptr, nullptr));
        if (out_ds) {
            return;
        }
    }
    GDALDriver* gtiff_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (gtiff_drv == nullptr) {
        throw std::runtime_error("Neither COG nor GTiff GDAL driver is available");
    }
    CslStringList opts;
    opts.set("TILED", "YES");
    opts.set("COMPRESS", "DEFLATE");
    opts.set("PREDICTOR", "YES");
    opts.set("NUM_THREADS", "ALL_CPUS");
    UniqueGdalDataset out_ds(gtiff_drv->CreateCopy(out_path.c_str(), mem_ds.get(), FALSE, opts.raw(), nullptr, nullptr));
    if (!out_ds) {
        throw std::runtime_error("Failed to write output raster: " + out_path.string());
    }
}

void read_state_raster(
    const fs::path& state_path,
    RasterMeta& meta,
    StackMetadata& smeta,
    std::vector<double>& state_bands
) {
    insar::io::ensure_gdal_registered();
    UniqueGdalDataset ds(static_cast<GDALDataset*>(GDALOpen(state_path.c_str(), GA_ReadOnly)));
    if (!ds) {
        throw std::runtime_error("Failed to open existing stack state: " + state_path.string());
    }
    if (ds->GetRasterCount() != kNumStateBands) {
        throw std::runtime_error("Unexpected band count in stack state raster: " + state_path.string());
    }
    meta.width = ds->GetRasterXSize();
    meta.height = ds->GetRasterYSize();
    if (ds->GetGeoTransform(meta.geotransform) != CE_None) {
        throw std::runtime_error("Stack state missing GeoTransform");
    }
    const char* proj = ds->GetProjectionRef();
    meta.wkt = (proj != nullptr) ? proj : "";

    auto read_meta_d = [&](const char* key, double fallback) -> double {
        const char* val = ds->GetMetadataItem(key);
        return (val != nullptr) ? std::stod(val) : fallback;
    };
    auto read_meta_i = [&](const char* key, int fallback) -> int {
        const char* val = ds->GetMetadataItem(key);
        return (val != nullptr) ? std::stoi(val) : fallback;
    };
    smeta.origin_days = read_meta_d("STACK_ORIGIN_DAYS", 0.0);
    smeta.first_t0_days = read_meta_d("STACK_FIRST_T0_DAYS", 0.0);
    smeta.last_t1_days = read_meta_d("STACK_LAST_T1_DAYS", 0.0);
    smeta.sum_dt_days = read_meta_d("STACK_SUM_DT_DAYS", 0.0);
    smeta.pair_count = read_meta_i("STACK_PAIR_COUNT", 0);
    smeta.redundant_pair_count = read_meta_i("STACK_REDUNDANT_PAIR_COUNT", 0);
    smeta.rejected_pair_count = read_meta_i("STACK_REJECTED_PAIR_COUNT", 0);

    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    state_bands.assign(static_cast<size_t>(kNumStateBands) * npix, 0.0);
    for (int b = 0; b < kNumStateBands; ++b) {
        GDALRasterBand* band = ds->GetRasterBand(b + 1);
        CPLErr err = band->RasterIO(
            GF_Read, 0, 0, meta.width, meta.height,
            state_bands.data() + static_cast<size_t>(b) * npix,
            meta.width, meta.height, GDT_Float64, 0, 0, nullptr);
        if (err != CE_None) {
            throw std::runtime_error("Failed reading state band " + std::to_string(b + 1));
        }
    }
}

void write_state_raster(
    const fs::path& state_path,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const std::vector<double>& state_bands
) {
    insar::io::ensure_gdal_registered();
    if (state_path.has_parent_path()) {
        fs::create_directories(state_path.parent_path());
    }
    const fs::path tmp_path = state_path.string() + ".tmp.tif";
    GDALDriver* gtiff_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (gtiff_drv == nullptr) {
        throw std::runtime_error("GTiff GDAL driver unavailable");
    }
    CslStringList opts;
    opts.set("TILED", "YES");
    opts.set("COMPRESS", "DEFLATE");
    opts.set("PREDICTOR", "3");
    opts.set("BLOCKXSIZE", "512");
    opts.set("BLOCKYSIZE", "512");
    opts.set("NUM_THREADS", "ALL_CPUS");
    opts.set("BIGTIFF", "IF_SAFER");

    UniqueGdalDataset ds(gtiff_drv->Create(
        tmp_path.c_str(), meta.width, meta.height, kNumStateBands, GDT_Float64, opts.raw()));
    if (!ds) {
        throw std::runtime_error("Failed creating stack state raster: " + tmp_path.string());
    }
    double gt[6];
    std::memcpy(gt, meta.geotransform, sizeof(gt));
    ds->SetGeoTransform(gt);
    if (!meta.wkt.empty()) {
        ds->SetProjection(meta.wkt.c_str());
    }
    ds->SetMetadataItem("STACK_ORIGIN_DAYS", std::to_string(smeta.origin_days).c_str());
    ds->SetMetadataItem("STACK_FIRST_T0_DAYS", std::to_string(smeta.first_t0_days).c_str());
    ds->SetMetadataItem("STACK_LAST_T1_DAYS", std::to_string(smeta.last_t1_days).c_str());
    ds->SetMetadataItem("STACK_SUM_DT_DAYS", std::to_string(smeta.sum_dt_days).c_str());
    ds->SetMetadataItem("STACK_PAIR_COUNT", std::to_string(smeta.pair_count).c_str());
    ds->SetMetadataItem("STACK_REDUNDANT_PAIR_COUNT", std::to_string(smeta.redundant_pair_count).c_str());
    ds->SetMetadataItem("STACK_REJECTED_PAIR_COUNT", std::to_string(smeta.rejected_pair_count).c_str());

    const char* band_names[kNumStateBands] = {
        "sum_w", "S_11", "S_12", "S_22", "b_1", "b_2", "S_yy",
        "cum_disp_raw", "valid_dt_yr", "sum_coh", "valid_count", "footprint_count", "latest_disp"
    };
    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    for (int b = 0; b < kNumStateBands; ++b) {
        GDALRasterBand* band = ds->GetRasterBand(b + 1);
        band->SetDescription(band_names[b]);
        if (b == kBandLatestDisp) {
            band->SetNoDataValue(kNanD);
        }
        CPLErr err = band->RasterIO(
            GF_Write, 0, 0, meta.width, meta.height,
            const_cast<double*>(state_bands.data() + static_cast<size_t>(b) * npix),
            meta.width, meta.height, GDT_Float64, 0, 0, nullptr);
        if (err != CE_None) {
            throw std::runtime_error("Failed writing state band " + std::to_string(b + 1));
        }
    }
    ds.reset();
    fs::rename(tmp_path, state_path);
}

std::vector<uint8_t> rasterize_vector_mask(const std::string& vector_path, const RasterMeta& meta) {
    insar::io::ensure_gdal_registered();
    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    std::vector<uint8_t> mask(npix, 0);
    if (meta.wkt.empty()) {
        throw std::runtime_error("Raster has empty CRS WKT; cannot rasterize vector mask: " + vector_path);
    }

    UniqueGdalDataset vds(static_cast<GDALDataset*>(
        GDALOpenEx(vector_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    if (!vds) {
        throw std::runtime_error("Failed to open vector mask: " + vector_path);
    }

    OGRSpatialReference dst_srs;
    if (dst_srs.importFromWkt(meta.wkt.c_str()) != OGRERR_NONE) {
        throw std::runtime_error("Invalid raster CRS WKT");
    }
    dst_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    GDALDriver* mem_drv = GetGDALDriverManager()->GetDriverByName("MEM");
    if (mem_drv == nullptr) {
        throw std::runtime_error("MEM driver unavailable");
    }
    UniqueGdalDataset mem_ds(mem_drv->Create("", meta.width, meta.height, 1, GDT_Byte, nullptr));
    if (!mem_ds) {
        throw std::runtime_error("Failed allocating rasterized mask buffer");
    }
    double gt[6];
    std::memcpy(gt, meta.geotransform, sizeof(gt));
    mem_ds->SetGeoTransform(gt);
    mem_ds->SetProjection(meta.wkt.c_str());
    if (mem_ds->GetRasterBand(1)->RasterIO(
            GF_Write, 0, 0, meta.width, meta.height, mask.data(),
            meta.width, meta.height, GDT_Byte, 0, 0, nullptr) != CE_None) {
        throw std::runtime_error("Failed initializing rasterized mask buffer");
    }

    std::vector<UniqueOgrGeom> owned_geoms;
    std::vector<OGRGeometryH> geom_handles;
    std::vector<double> burn_vals;

    for (int li = 0; li < vds->GetLayerCount(); ++li) {
        OGRLayer* layer = vds->GetLayer(li);
        if (layer == nullptr) continue;
        layer->ResetReading();

        OGRSpatialReference src_srs;
        if (layer->GetSpatialRef() != nullptr) {
            src_srs = *(layer->GetSpatialRef());
        } else {
            src_srs.SetWellKnownGeogCS("WGS84");
        }
        src_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        UniqueOgrTransform to_dst(OGRCreateCoordinateTransformation(&src_srs, &dst_srs));
        if (!to_dst) {
            throw std::runtime_error("Failed creating vector->raster transform for: " + vector_path);
        }

        OGRFeature* raw_feat = nullptr;
        while ((raw_feat = layer->GetNextFeature()) != nullptr) {
            UniqueOgrFeature feat(raw_feat);
            OGRGeometry* g = feat->GetGeometryRef();
            if (g == nullptr || g->IsEmpty()) continue;
            UniqueOgrGeom g_dst(g->clone());
            if (g_dst->transform(to_dst.get()) == OGRERR_NONE) {
                geom_handles.push_back(OGRGeometry::ToHandle(g_dst.get()));
                burn_vals.push_back(1.0);
                owned_geoms.push_back(std::move(g_dst));
            }
        }
    }

    if (!geom_handles.empty()) {
        int bands[1] = {1};
        CslStringList r_opts;
        r_opts.set("ALL_TOUCHED", "TRUE");
        CPLErr err = GDALRasterizeGeometries(
            GDALDataset::ToHandle(mem_ds.get()), 1, bands,
            static_cast<int>(geom_handles.size()), geom_handles.data(),
            nullptr, nullptr, burn_vals.data(), r_opts.raw(), nullptr, nullptr);
        if (err != CE_None) {
            throw std::runtime_error("GDALRasterizeGeometries failed on: " + vector_path);
        }
        if (mem_ds->GetRasterBand(1)->RasterIO(
                GF_Read, 0, 0, meta.width, meta.height, mask.data(),
                meta.width, meta.height, GDT_Byte, 0, 0, nullptr) != CE_None) {
            throw std::runtime_error("Failed reading rasterized mask buffer");
        }
    }
    return mask;
}

std::vector<uint8_t> load_mask_any(const std::string& path, const RasterMeta& meta) {
    insar::io::ensure_gdal_registered();
    UniqueGdalDataset vds;
    {
        CplQuietErrorScope quiet;
        vds.reset(static_cast<GDALDataset*>(
            GDALOpenEx(path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    }
    if (vds) {
        vds.reset();
        return rasterize_vector_mask(path, meta);
    }
    std::vector<float> r = read_raster_aligned_f32(path, meta, GRA_NearestNeighbour);
    std::vector<uint8_t> mask(r.size(), 0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < r.size(); ++i) {
        if (std::isfinite(r[i]) && r[i] > 0.5f) {
            mask[i] = 1;
        }
    }
    return mask;
}

} // namespace insar::stack
