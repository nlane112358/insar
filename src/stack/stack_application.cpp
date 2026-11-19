#include "insar/stack/stack_application.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpl_conv.h"
#include "cpl_string.h"
#include "gdal_alg.h"
#include "gdal_priv.h"
#include "gdalwarper.h"
#include "ogr_api.h"
#include "ogr_spatialref.h"
#include "ogrsf_frmts.h"

namespace fs = std::filesystem;

namespace {

constexpr double kDaysPerYear = 365.25;
constexpr double kPi = 3.14159265358979323846;
constexpr float kNanF = std::numeric_limits<float>::quiet_NaN();
constexpr double kNanD = std::numeric_limits<double>::quiet_NaN();

enum StateBands : int {
    kBandSumW = 0,
    kBandS11 = 1,
    kBandS12 = 2,
    kBandS22 = 3,
    kBandB1 = 4,
    kBandB2 = 5,
    kBandSyy = 6,
    kBandCumDispRaw = 7,
    kBandValidDtYr = 8,
    kBandSumCoh = 9,
    kBandValidCount = 10,
    kBandFootprintCount = 11,
    kBandLatestDisp = 12,
    kNumStateBands = 13
};

struct GdalDatasetDeleter {
    void operator()(GDALDataset* ds) const {
        if (ds != nullptr) {
            GDALClose(ds);
        }
    }
};
using UniqueGdalDataset = std::unique_ptr<GDALDataset, GdalDatasetDeleter>;

struct OgrGeomDeleter {
    void operator()(OGRGeometry* g) const {
        if (g != nullptr) {
            OGRGeometryFactory::destroyGeometry(g);
        }
    }
};
using UniqueOgrGeom = std::unique_ptr<OGRGeometry, OgrGeomDeleter>;

struct OgrCtDeleter {
    void operator()(OGRCoordinateTransformation* ct) const {
        if (ct != nullptr) {
            OGRCoordinateTransformation::DestroyCT(ct);
        }
    }
};
using UniqueOgrCt = std::unique_ptr<OGRCoordinateTransformation, OgrCtDeleter>;

struct RasterMeta {
    int width = 0;
    int height = 0;
    double geotransform[6] = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
    std::string wkt;
};

struct StackMetadata {
    double origin_days = 0.0;
    double first_t0_days = 0.0;
    double last_t1_days = 0.0;
    double sum_dt_days = 0.0;
    int pair_count = 0;
    int redundant_pair_count = 0;
    int rejected_pair_count = 0;
};

struct ClosureTripletSpec {
    std::string disp_a_path;
    std::string disp_b_path;
    std::string mode; // "add_sub" => d_P + d_a - d_b; "sum_sub" => d_a + d_b - d_P
};

struct ClosureResult {
    int triplets_evaluated = 0;
    size_t evaluated_coherent_pixels = 0;
    size_t masked_pixels = 0;
    double bad_fraction = 0.0;
    double rms_mm = kNanD;
    double thresh_mm = 10.0;
    double max_bad_frac = 0.35;
    std::string mode = "consensus";
    bool pair_rejected = false;
};

struct ReferencePinResult {
    bool configured = false;
    bool applied = false;
    std::string mode = "none";
    double lon = kNanD;
    double lat = kNanD;
    double radius_m = kNanD;
    size_t ref_pixels_used = 0;
    double offset_mm = 0.0;
};

struct RampResult {
    int order = 1;
    bool applied = false;
    double rms_before_mm = 0.0;
    double rms_after_mm = 0.0;
    std::vector<double> coeffs;
    size_t masked_deform_pixels = 0;
    double masked_deform_fraction = 0.0;
    bool dem_applied = false;
    double dem_slope_mm_per_km = kNanD;
    double dem_corr_before = kNanD;
    double dem_corr_after = kNanD;
    double dem_mean_m = kNanD;
    double dem_std_m = kNanD;
};

struct CorridorMetric {
    std::string asset_id;
    std::string name;
    std::string asset_class;
    std::string source_ref;
    std::string wgs84_geom_json;
    int pixel_count = 0;
    int valid_pixels = 0;
    int high_coh_pixels = 0;
    int alert_pixel_count = 0;
    int snr_gated_pixels = 0;
    double coverage_ratio = 0.0;
    double mean_coherence = kNanD;
    double mean_velocity_mm_yr = kNanD;
    double min_velocity_mm_yr = kNanD;
    double max_velocity_mm_yr = kNanD;
    double p95_abs_velocity_mm_yr = kNanD;
    double mean_snr = kNanD;
    double p95_snr = kNanD;
    double mean_cumulative_disp_mm = kNanD;
    double max_abs_cumulative_disp_mm = kNanD;
    double mean_acceleration_mm_yr2 = kNanD;
    double max_abs_acceleration_mm_yr2 = kNanD;
    double latest_interval_disp_mm = kNanD;
    bool maturity_gated = false;
    std::string risk_level = "no_coverage";
};

struct AlertPoint {
    int px = 0;
    int py = 0;
    double lon = 0.0;
    double lat = 0.0;
    double velocity_mm_yr = 0.0;
    double acceleration_mm_yr2 = kNanD;
    double cumulative_disp_mm = 0.0;
    double temporal_coherence = 0.0;
    double uncertainty_mm_yr = kNanD;
    double velocity_snr = kNanD;
    std::string severity = "critical";
    std::string corridor_asset_id;
    std::string corridor_name;
    double score = 0.0;
};

struct Options {
    std::string state_path;
    std::string disp_path;
    std::string coh_path;
    std::string out_dir;
    std::string corridors_path;
    std::string dem_path;
    std::string deform_mask_path;
    std::string prior_vel_path;
    std::string ref_region_path;
    bool has_ref_lonlat = false;
    double ref_lon = 0.0;
    double ref_lat = 0.0;
    double ref_radius_m = 1000.0;
    bool has_ref_px = false;
    int ref_px = 0;
    int ref_py = 0;
    int ref_radius_px = 10;

    double dt_days = -1.0;
    double t0_days = kNanD;
    double t1_days = kNanD;
    int ramp_order = 1;
    float min_coh = 0.35f;
    float ramp_deform_vel_mm_yr = 25.0f;
    float ramp_mask_sigma = 2.5f;
    int ramp_mask_dilate_px = 2;

    std::vector<ClosureTripletSpec> closure_triplets;
    float closure_thresh_mm = 10.0f;
    float closure_max_bad_frac = 0.35f;
    std::string closure_mode = "consensus";
    bool redundant_pair = false;
    double pair_weight_scale = 1.0;

    double buffer_m = 100.0;
    float vel_alert_mm_yr = 15.0f;
    float accel_alert_mm_yr2 = 25.0f;
    float min_alert_coh = 0.55f;
    float min_alert_snr = 2.5f;
    double min_alert_span_days = 60.0;
    int min_alert_passes = 0;

    bool export_ramp_disp = true;
    bool no_export = false;
    bool skip_cog = false;
};

std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (unsigned char c : input) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
                } else {
                    oss << static_cast<char>(c);
                }
        }
    }
    return oss.str();
}

std::string json_num(double v, int prec = 4) {
    if (!std::isfinite(v)) {
        return "null";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Sequential O(1) InSAR stacking, DEM tropospheric correction, triplet closure, and corridor analytics.\n\n"
        << "Required:\n"
        << "  --state <stack_state.tif>     Persistent Float64 sufficient-statistics raster\n"
        << "  --disp <displacement_mm.tif>  Input unwrapped pair displacement GeoTIFF (mm)\n"
        << "  --coh <coherence.tif>         Input interferometric coherence GeoTIFF [0..1]\n"
        << "  --dt-days <double>            Acquisition baseline duration (days), OR\n"
        << "  --t0-days <d0> --t1-days <d1> Explicit epoch bounds relative to stack t=0 (days)\n"
        << "  --out-dir <dir>               Output directory for derived COGs & corridor JSONs\n\n"
        << "Reference & Ramp / Tropospheric Correction:\n"
        << "  --ramp-order <-1|0|1|2>       Robust Huber/Tukey IRLS ramp order (-1=off, default 1)\n"
        << "  --min-coh <float>             Minimum coherence for ramp fit & stack update (default 0.35)\n"
        << "  --dem <dem.tif>               DEM GeoTIFF (local/SRTM/Copernicus) for elevation-correlated tropo removal\n"
        << "  --ref-lonlat <lon> <lat>      Pin zero datum to known stable WGS84 reference region\n"
        << "  --ref-radius-m <double>       Radius in meters for --ref-lonlat (default 1000.0)\n"
        << "  --ref-px <x> <y> <r>          Pin zero datum to pixel disk (x, y, radius_px)\n"
        << "  --ref-region <path>           Vector polygon (GeoJSON/GPKG) of known stable reference region\n"
        << "  --deform-mask <path>          Vector polygon or raster mask of high-deformation zones to exclude from ramp fit\n"
        << "  --prior-vel <vel.tif>         Optional prior velocity raster for masking |v| >= --ramp-deform-vel-mm-yr\n"
        << "  --ramp-deform-vel-mm-yr <f>   Velocity threshold (mm/yr) to auto-mask deforming pixels before ramp fit (default 25.0)\n"
        << "  --ramp-mask-sigma <float>     Single-pair residual MAD multiplier for masking deformation before final fit (default 2.5)\n"
        << "  --ramp-mask-dilate-px <int>   Dilation radius (px) around masked deformation zones (default 2)\n\n"
        << "Triplet Phase Closure & Redundant Pairs:\n"
        << "  --closure-triplet <a> <b> <m> Check triplet closure (m='add_sub': d+da-db, m='sum_sub': da+db-d); repeatable\n"
        << "  --closure-thresh-mm <float>   Per-pixel |closure| threshold in mm to mask unwrapping errors (default 10.0)\n"
        << "  --closure-max-bad-frac <f>    Reject entire pair if bad coherent pixel fraction exceeds this (default 0.35)\n"
        << "  --closure-mode <m>            'consensus' (majority fail, default) or 'strict' (any triplet fail)\n"
        << "  --redundant-pair              Update WLS velocity/accel normal eqs without double-counting cumulative span\n"
        << "  --pair-weight-scale <double>  Weight multiplier for pair normal equations (default 1.0)\n\n"
        << "Corridors & Alert Maturity / SNR Gating:\n"
        << "  --corridors <path>            Vector file (GeoJSON/GPKG) of infrastructure corridors\n"
        << "  --buffer-m <double>           Metric corridor buffer radius in meters (default 100.0)\n"
        << "  --vel-alert-mm-yr <float>     |Velocity| alert threshold in mm/yr (default 15.0)\n"
        << "  --accel-alert-mm-yr2 <float>  |Acceleration| alert threshold in mm/yr^2 (default 25.0)\n"
        << "  --min-alert-coh <float>       Minimum temporal coherence for alerts (default 0.55)\n"
        << "  --min-alert-snr <float>       Minimum velocity SNR |v|/sigma_v for alerts (default 2.5)\n"
        << "  --min-alert-span-days <d>     Minimum stack span (days) before issuing critical alerts (default 60.0)\n"
        << "  --min-alert-passes <int>      Minimum valid stacked pairs at a pixel for alerts (0=auto)\n"
        << "  --no-export                   Update state raster only without exporting outputs\n"
        << "  --skip-cog                    Export corridor JSONs/summary without encoding COGs\n"
        << "  --help                        Show this help text\n";
}

Options parse_args(int argc, char** argv) {
    Options opt;
    auto need_val = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("Missing value for ") + flag);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--state") {
            opt.state_path = need_val(i, "--state");
        } else if (arg == "--disp") {
            opt.disp_path = need_val(i, "--disp");
        } else if (arg == "--coh") {
            opt.coh_path = need_val(i, "--coh");
        } else if (arg == "--out-dir") {
            opt.out_dir = need_val(i, "--out-dir");
        } else if (arg == "--corridors") {
            opt.corridors_path = need_val(i, "--corridors");
        } else if (arg == "--dem") {
            opt.dem_path = need_val(i, "--dem");
        } else if (arg == "--deform-mask") {
            opt.deform_mask_path = need_val(i, "--deform-mask");
        } else if (arg == "--prior-vel") {
            opt.prior_vel_path = need_val(i, "--prior-vel");
        } else if (arg == "--ref-region") {
            opt.ref_region_path = need_val(i, "--ref-region");
        } else if (arg == "--ref-lonlat") {
            opt.has_ref_lonlat = true;
            opt.ref_lon = std::stod(need_val(i, "--ref-lonlat lon"));
            opt.ref_lat = std::stod(need_val(i, "--ref-lonlat lat"));
        } else if (arg == "--ref-radius-m") {
            opt.ref_radius_m = std::stod(need_val(i, "--ref-radius-m"));
        } else if (arg == "--ref-px") {
            opt.has_ref_px = true;
            opt.ref_px = std::stoi(need_val(i, "--ref-px x"));
            opt.ref_py = std::stoi(need_val(i, "--ref-px y"));
            opt.ref_radius_px = std::stoi(need_val(i, "--ref-px r"));
        } else if (arg == "--dt-days") {
            opt.dt_days = std::stod(need_val(i, "--dt-days"));
        } else if (arg == "--t0-days") {
            opt.t0_days = std::stod(need_val(i, "--t0-days"));
        } else if (arg == "--t1-days") {
            opt.t1_days = std::stod(need_val(i, "--t1-days"));
        } else if (arg == "--ramp-order") {
            opt.ramp_order = std::stoi(need_val(i, "--ramp-order"));
        } else if (arg == "--min-coh") {
            opt.min_coh = std::stof(need_val(i, "--min-coh"));
        } else if (arg == "--ramp-deform-vel-mm-yr") {
            opt.ramp_deform_vel_mm_yr = std::stof(need_val(i, "--ramp-deform-vel-mm-yr"));
        } else if (arg == "--ramp-mask-sigma") {
            opt.ramp_mask_sigma = std::stof(need_val(i, "--ramp-mask-sigma"));
        } else if (arg == "--ramp-mask-dilate-px") {
            opt.ramp_mask_dilate_px = std::stoi(need_val(i, "--ramp-mask-dilate-px"));
        } else if (arg == "--closure-triplet") {
            ClosureTripletSpec spec;
            spec.disp_a_path = need_val(i, "--closure-triplet disp_a");
            spec.disp_b_path = need_val(i, "--closure-triplet disp_b");
            spec.mode = need_val(i, "--closure-triplet mode");
            if (spec.mode != "add_sub" && spec.mode != "sum_sub") {
                throw std::runtime_error("--closure-triplet mode must be 'add_sub' or 'sum_sub'");
            }
            opt.closure_triplets.push_back(std::move(spec));
        } else if (arg == "--closure-thresh-mm") {
            opt.closure_thresh_mm = std::stof(need_val(i, "--closure-thresh-mm"));
        } else if (arg == "--closure-max-bad-frac") {
            opt.closure_max_bad_frac = std::stof(need_val(i, "--closure-max-bad-frac"));
        } else if (arg == "--closure-mode") {
            opt.closure_mode = need_val(i, "--closure-mode");
            if (opt.closure_mode != "consensus" && opt.closure_mode != "strict") {
                throw std::runtime_error("--closure-mode must be 'consensus' or 'strict'");
            }
        } else if (arg == "--redundant-pair") {
            opt.redundant_pair = true;
        } else if (arg == "--pair-weight-scale") {
            opt.pair_weight_scale = std::stod(need_val(i, "--pair-weight-scale"));
        } else if (arg == "--buffer-m") {
            opt.buffer_m = std::stod(need_val(i, "--buffer-m"));
        } else if (arg == "--vel-alert-mm-yr") {
            opt.vel_alert_mm_yr = std::stof(need_val(i, "--vel-alert-mm-yr"));
        } else if (arg == "--accel-alert-mm-yr2") {
            opt.accel_alert_mm_yr2 = std::stof(need_val(i, "--accel-alert-mm-yr2"));
        } else if (arg == "--min-alert-coh") {
            opt.min_alert_coh = std::stof(need_val(i, "--min-alert-coh"));
        } else if (arg == "--min-alert-snr") {
            opt.min_alert_snr = std::stof(need_val(i, "--min-alert-snr"));
        } else if (arg == "--min-alert-span-days") {
            opt.min_alert_span_days = std::stod(need_val(i, "--min-alert-span-days"));
        } else if (arg == "--min-alert-passes") {
            opt.min_alert_passes = std::stoi(need_val(i, "--min-alert-passes"));
        } else if (arg == "--no-export") {
            opt.no_export = true;
        } else if (arg == "--skip-cog") {
            opt.skip_cog = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opt.state_path.empty()) {
        throw std::runtime_error("--state is required");
    }
    if (!opt.no_export && opt.out_dir.empty()) {
        throw std::runtime_error("--out-dir is required unless --no-export is set");
    }
    bool have_disp = !opt.disp_path.empty();
    bool have_coh = !opt.coh_path.empty();
    if (have_disp != have_coh) {
        throw std::runtime_error("--disp and --coh must both be provided");
    }
    if (have_disp) {
        if (std::isfinite(opt.t0_days) && std::isfinite(opt.t1_days)) {
            if (opt.t1_days <= opt.t0_days) {
                throw std::runtime_error("--t1-days must exceed --t0-days");
            }
            opt.dt_days = opt.t1_days - opt.t0_days;
        } else if (!(opt.dt_days > 0.0)) {
            throw std::runtime_error("Provide --dt-days > 0 or valid --t0-days and --t1-days");
        }
    }
    if (opt.ramp_order < -1 || opt.ramp_order > 2) {
        throw std::runtime_error("--ramp-order must be -1, 0, 1, or 2");
    }
    if (opt.has_ref_lonlat && opt.ref_radius_m <= 0.0) {
        throw std::runtime_error("--ref-radius-m must be positive");
    }
    if (opt.has_ref_px && opt.ref_radius_px <= 0) {
        throw std::runtime_error("--ref-px radius must be positive");
    }
    if (opt.pair_weight_scale <= 0.0) {
        throw std::runtime_error("--pair-weight-scale must be positive");
    }
    if (opt.closure_thresh_mm <= 0.0f) {
        throw std::runtime_error("--closure-thresh-mm must be positive");
    }
    return opt;
}

std::vector<float> read_single_band_f32(const std::string& path, RasterMeta& meta_out) {
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
    GDALResampleAlg resample_alg = GRA_Bilinear
) {
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

    char** w_str = nullptr;
    w_str = CSLSetNameValue(w_str, "INIT_DEST", "NO_DATA");
    w_str = CSLSetNameValue(w_str, "NUM_THREADS", "ALL_CPUS");
    warp_opts->papszWarpOptions = w_str;

    const char* effective_src_wkt = src_meta.wkt.empty() ? nullptr : src_meta.wkt.c_str();
    const char* effective_dst_wkt = target_meta.wkt.empty() ? nullptr : target_meta.wkt.c_str();
    warp_opts->pTransformerArg = GDALCreateGenImgProjTransformer(
        src_ds.get(), effective_src_wkt, dst_ds.get(), effective_dst_wkt, FALSE, 0.0, 1);
    if (warp_opts->pTransformerArg == nullptr) {
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
        char** opts = nullptr;
        opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
        opts = CSLSetNameValue(opts, "PREDICTOR", "YES");
        opts = CSLSetNameValue(opts, "BLOCKSIZE", "512");
        opts = CSLSetNameValue(opts, "NUM_THREADS", "ALL_CPUS");
        opts = CSLSetNameValue(opts, "BIGTIFF", "IF_SAFER");
        UniqueGdalDataset out_ds(cog_drv->CreateCopy(out_path.c_str(), mem_ds.get(), FALSE, opts, nullptr, nullptr));
        CSLDestroy(opts);
        if (out_ds) {
            return;
        }
    }
    GDALDriver* gtiff_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (gtiff_drv == nullptr) {
        throw std::runtime_error("Neither COG nor GTiff GDAL driver is available");
    }
    char** opts = nullptr;
    opts = CSLSetNameValue(opts, "TILED", "YES");
    opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
    opts = CSLSetNameValue(opts, "PREDICTOR", "YES");
    opts = CSLSetNameValue(opts, "NUM_THREADS", "ALL_CPUS");
    UniqueGdalDataset out_ds(gtiff_drv->CreateCopy(out_path.c_str(), mem_ds.get(), FALSE, opts, nullptr, nullptr));
    CSLDestroy(opts);
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
    if (state_path.has_parent_path()) {
        fs::create_directories(state_path.parent_path());
    }
    const fs::path tmp_path = state_path.string() + ".tmp.tif";
    GDALDriver* gtiff_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (gtiff_drv == nullptr) {
        throw std::runtime_error("GTiff GDAL driver unavailable");
    }
    char** opts = nullptr;
    opts = CSLSetNameValue(opts, "TILED", "YES");
    opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
    opts = CSLSetNameValue(opts, "PREDICTOR", "3");
    opts = CSLSetNameValue(opts, "BLOCKXSIZE", "512");
    opts = CSLSetNameValue(opts, "BLOCKYSIZE", "512");
    opts = CSLSetNameValue(opts, "NUM_THREADS", "ALL_CPUS");
    opts = CSLSetNameValue(opts, "BIGTIFF", "IF_SAFER");

    UniqueGdalDataset ds(gtiff_drv->Create(
        tmp_path.c_str(), meta.width, meta.height, kNumStateBands, GDT_Float64, opts));
    CSLDestroy(opts);
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

double meters_to_crs_units(const OGRSpatialReference& srs, double meters, double center_lat_deg) {
    if (srs.IsProjected()) {
        double lin = srs.GetLinearUnits(nullptr);
        return (lin > 0.0) ? (meters / lin) : meters;
    }
    double cos_lat = std::max(0.2, std::cos(center_lat_deg * kPi / 180.0));
    return meters / (111320.0 * cos_lat);
}

std::vector<uint8_t> rasterize_vector_mask(const std::string& vector_path, const RasterMeta& meta) {
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
        UniqueOgrCt to_dst(OGRCreateCoordinateTransformation(&src_srs, &dst_srs));
        if (!to_dst) {
            throw std::runtime_error("Failed creating vector->raster transform for: " + vector_path);
        }

        OGRFeature* raw_feat = nullptr;
        while ((raw_feat = layer->GetNextFeature()) != nullptr) {
            std::unique_ptr<OGRFeature, void(*)(OGRFeature*)> feat(raw_feat, OGRFeature::DestroyFeature);
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
        char** r_opts = nullptr;
        r_opts = CSLSetNameValue(r_opts, "ALL_TOUCHED", "TRUE");
        CPLErr err = GDALRasterizeGeometries(
            GDALDataset::ToHandle(mem_ds.get()), 1, bands,
            static_cast<int>(geom_handles.size()), geom_handles.data(),
            nullptr, nullptr, burn_vals.data(), r_opts, nullptr, nullptr);
        CSLDestroy(r_opts);
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
    // Try vector first; if not a vector dataset, fall back to aligned raster
    CPLPushErrorHandler(CPLQuietErrorHandler);
    UniqueGdalDataset vds(static_cast<GDALDataset*>(
        GDALOpenEx(path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    CPLPopErrorHandler();
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

std::vector<uint8_t> dilate_mask_box(const std::vector<uint8_t>& src, int width, int height, int radius) {
    if (radius <= 0) {
        return src;
    }
    const size_t npix = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> horiz(npix, 0);
    std::vector<uint8_t> out(npix, 0);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const size_t row = static_cast<size_t>(y) * width;
        int acc = 0;
        const int r0 = std::min(width - 1, radius);
        for (int x = 0; x <= r0; ++x) {
            acc += src[row + x] ? 1 : 0;
        }
        for (int x = 0; x < width; ++x) {
            horiz[row + x] = (acc > 0) ? 1 : 0;
            const int leave = x - radius;
            const int enter = x + radius + 1;
            if (leave >= 0 && src[row + leave]) acc--;
            if (enter < width && src[row + enter]) acc++;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int x = 0; x < width; ++x) {
        int acc = 0;
        const int r0 = std::min(height - 1, radius);
        for (int y = 0; y <= r0; ++y) {
            acc += horiz[static_cast<size_t>(y) * width + x] ? 1 : 0;
        }
        for (int y = 0; y < height; ++y) {
            out[static_cast<size_t>(y) * width + x] = (acc > 0) ? 1 : 0;
            const int leave = y - radius;
            const int enter = y + radius + 1;
            if (leave >= 0 && horiz[static_cast<size_t>(leave) * width + x]) acc--;
            if (enter < height && horiz[static_cast<size_t>(enter) * width + x]) acc++;
        }
    }
    return out;
}

std::vector<uint8_t> build_reference_mask(
    const Options& opt,
    const RasterMeta& meta,
    ReferencePinResult& ref_info
) {
    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    std::vector<uint8_t> mask(npix, 0);

    if (!opt.ref_region_path.empty()) {
        ref_info.configured = true;
        ref_info.mode = "vector_region";
        return load_mask_any(opt.ref_region_path, meta);
    }

    if (opt.has_ref_lonlat) {
        ref_info.configured = true;
        ref_info.mode = "lonlat_disk";
        ref_info.lon = opt.ref_lon;
        ref_info.lat = opt.ref_lat;
        ref_info.radius_m = opt.ref_radius_m;

        if (meta.wkt.empty()) {
            throw std::runtime_error("Raster has empty CRS WKT; cannot project --ref-lonlat");
        }
        OGRSpatialReference wgs84, dst_srs;
        wgs84.SetWellKnownGeogCS("WGS84");
        wgs84.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        if (dst_srs.importFromWkt(meta.wkt.c_str()) != OGRERR_NONE) {
            throw std::runtime_error("Invalid raster CRS WKT for --ref-lonlat");
        }
        dst_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        UniqueOgrCt ct(OGRCreateCoordinateTransformation(&wgs84, &dst_srs));
        if (!ct) {
            throw std::runtime_error("Failed creating WGS84->raster transform for --ref-lonlat");
        }
        double mx = opt.ref_lon;
        double my = opt.ref_lat;
        if (!ct->Transform(1, &mx, &my)) {
            throw std::runtime_error("Failed transforming --ref-lonlat coordinates to raster CRS");
        }
        const double* gt = meta.geotransform;
        const double det = gt[1] * gt[5] - gt[2] * gt[4];
        if (std::abs(det) < 1e-18) {
            throw std::runtime_error("Degenerate GeoTransform for --ref-lonlat");
        }
        const double cx = (gt[5] * (mx - gt[0]) - gt[2] * (my - gt[3])) / det;
        const double cy = (-gt[4] * (mx - gt[0]) + gt[1] * (my - gt[3])) / det;
        const double crs_radius = meters_to_crs_units(dst_srs, opt.ref_radius_m, opt.ref_lat);
        const double pix_size = std::max(1e-12, 0.5 * (std::hypot(gt[1], gt[4]) + std::hypot(gt[2], gt[5])));
        const double r_px = std::max(1.0, crs_radius / pix_size);

        const int x0 = std::max(0, static_cast<int>(std::floor(cx - r_px)));
        const int x1 = std::min(meta.width - 1, static_cast<int>(std::ceil(cx + r_px)));
        const int y0 = std::max(0, static_cast<int>(std::floor(cy - r_px)));
        const int y1 = std::min(meta.height - 1, static_cast<int>(std::ceil(cy + r_px)));
        const double r2 = r_px * r_px;
        for (int y = y0; y <= y1; ++y) {
            const double dy = (y + 0.5) - cy;
            for (int x = x0; x <= x1; ++x) {
                const double dx = (x + 0.5) - cx;
                if (dx * dx + dy * dy <= r2) {
                    mask[static_cast<size_t>(y) * meta.width + x] = 1;
                }
            }
        }
        return mask;
    }

    if (opt.has_ref_px) {
        ref_info.configured = true;
        ref_info.mode = "pixel_disk";
        const double cx = static_cast<double>(opt.ref_px) + 0.5;
        const double cy = static_cast<double>(opt.ref_py) + 0.5;
        const double r_px = static_cast<double>(std::max(1, opt.ref_radius_px));
        const int x0 = std::max(0, static_cast<int>(std::floor(cx - r_px)));
        const int x1 = std::min(meta.width - 1, static_cast<int>(std::ceil(cx + r_px)));
        const int y0 = std::max(0, static_cast<int>(std::floor(cy - r_px)));
        const int y1 = std::min(meta.height - 1, static_cast<int>(std::ceil(cy + r_px)));
        const double r2 = r_px * r_px;
        for (int y = y0; y <= y1; ++y) {
            const double dy = (y + 0.5) - cy;
            for (int x = x0; x <= x1; ++x) {
                const double dx = (x + 0.5) - cx;
                if (dx * dx + dy * dy <= r2) {
                    mask[static_cast<size_t>(y) * meta.width + x] = 1;
                }
            }
        }
        return mask;
    }

    return mask;
}

void apply_reference_pinning(
    std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& ref_mask,
    float min_coh,
    ReferencePinResult& ref_info
) {
    if (!ref_info.configured) {
        return;
    }
    const size_t npix = disp.size();
    std::vector<float> samples;
    samples.reserve(4096);

    for (size_t i = 0; i < npix; ++i) {
        if (ref_mask[i] && std::isfinite(disp[i]) && std::isfinite(coh[i]) && coh[i] >= min_coh) {
            samples.push_back(disp[i]);
        }
    }
    if (samples.size() < 5) {
        samples.clear();
        for (size_t i = 0; i < npix; ++i) {
            if (ref_mask[i] && std::isfinite(disp[i]) && std::isfinite(coh[i]) && coh[i] >= 0.25f) {
                samples.push_back(disp[i]);
            }
        }
    }
    if (samples.empty()) {
        for (size_t i = 0; i < npix; ++i) {
            if (ref_mask[i] && std::isfinite(disp[i])) {
                samples.push_back(disp[i]);
            }
        }
    }
    if (samples.empty()) {
        std::cerr << "Warning: Configured reference region has 0 valid pixels on this pair; skipping pin.\n";
        ref_info.applied = false;
        ref_info.ref_pixels_used = 0;
        return;
    }

    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    const double offset = static_cast<double>(samples[mid]);
    ref_info.offset_mm = offset;
    ref_info.ref_pixels_used = samples.size();
    ref_info.applied = true;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        if (std::isfinite(disp[i])) {
            disp[i] = static_cast<float>(static_cast<double>(disp[i]) - offset);
        }
    }
}

ClosureResult evaluate_triplet_closure(
    const Options& opt,
    const RasterMeta& meta,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& ref_mask,
    std::vector<uint8_t>& bad_mask_out,
    std::vector<float>& residual_map_out
) {
    ClosureResult res;
    res.thresh_mm = opt.closure_thresh_mm;
    res.max_bad_frac = opt.closure_max_bad_frac;
    res.mode = opt.closure_mode;

    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    bad_mask_out.assign(npix, 0);
    residual_map_out.assign(npix, kNanF);

    if (opt.closure_triplets.empty()) {
        return res;
    }

    std::vector<uint16_t> eval_cnt(npix, 0);
    std::vector<uint16_t> fail_cnt(npix, 0);
    std::vector<double> sum_sq_res(npix, 0.0);
    std::vector<float> tmp_r(npix, kNanF);
    const bool has_ref = std::any_of(ref_mask.begin(), ref_mask.end(), [](uint8_t v) { return v != 0; });

    for (const auto& trip : opt.closure_triplets) {
        RasterMeta meta_a, meta_b;
        std::vector<float> da = read_single_band_f32(trip.disp_a_path, meta_a);
        std::vector<float> db = read_single_band_f32(trip.disp_b_path, meta_b);
        assert_compatible_grid(meta, meta_a, "closure triplet A");
        assert_compatible_grid(meta, meta_b, "closure triplet B");

        const bool is_add_sub = (trip.mode == "add_sub");
        std::fill(tmp_r.begin(), tmp_r.end(), kNanF);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; ++i) {
            if (std::isfinite(disp[i]) && std::isfinite(da[i]) && std::isfinite(db[i])) {
                tmp_r[i] = is_add_sub ? (disp[i] + da[i] - db[i]) : (da[i] + db[i] - disp[i]);
            }
        }

        // Remove constant datum offset per triplet using coherent reference (or scene) median
        std::vector<float> offset_samples;
        offset_samples.reserve(npix / 8);
        if (has_ref) {
            for (size_t i = 0; i < npix; ++i) {
                if (ref_mask[i] && std::isfinite(tmp_r[i]) && std::isfinite(coh[i]) && coh[i] >= opt.min_coh) {
                    offset_samples.push_back(tmp_r[i]);
                }
            }
        }
        if (offset_samples.size() < 10) {
            offset_samples.clear();
            const size_t step = std::max<size_t>(1, npix / 250000);
            for (size_t i = 0; i < npix; i += step) {
                if (std::isfinite(tmp_r[i]) && std::isfinite(coh[i]) && coh[i] >= opt.min_coh) {
                    offset_samples.push_back(tmp_r[i]);
                }
            }
        }
        double median_offset = 0.0;
        if (!offset_samples.empty()) {
            const size_t mid = offset_samples.size() / 2;
            std::nth_element(offset_samples.begin(), offset_samples.begin() + mid, offset_samples.end());
            median_offset = static_cast<double>(offset_samples[mid]);
        }

        res.triplets_evaluated++;
        const double thresh = static_cast<double>(opt.closure_thresh_mm);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; ++i) {
            if (!std::isfinite(tmp_r[i])) continue;
            const double rc = static_cast<double>(tmp_r[i]) - median_offset;
            eval_cnt[i]++;
            sum_sq_res[i] += rc * rc;
            if (std::abs(rc) > thresh) {
                fail_cnt[i]++;
            }
        }
    }

    const bool strict_mode = (opt.closure_mode == "strict");
    size_t eval_coh_cnt = 0;
    size_t bad_coh_cnt = 0;
    size_t masked_total = 0;
    double total_sq = 0.0;

    #pragma omp parallel for reduction(+:eval_coh_cnt, bad_coh_cnt, masked_total, total_sq) schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        if (eval_cnt[i] == 0) continue;
        const double rms_pix = std::sqrt(sum_sq_res[i] / static_cast<double>(eval_cnt[i]));
        residual_map_out[i] = static_cast<float>(rms_pix);
        const bool is_bad = strict_mode
            ? (fail_cnt[i] > 0)
            : (static_cast<int>(fail_cnt[i]) * 2 > static_cast<int>(eval_cnt[i]));
        if (is_bad) {
            bad_mask_out[i] = 1;
            masked_total++;
        }
        if (std::isfinite(coh[i]) && coh[i] >= opt.min_coh) {
            eval_coh_cnt++;
            total_sq += rms_pix * rms_pix;
            if (is_bad) {
                bad_coh_cnt++;
            }
        }
    }

    res.evaluated_coherent_pixels = eval_coh_cnt;
    res.masked_pixels = masked_total;
    if (eval_coh_cnt > 0) {
        res.bad_fraction = static_cast<double>(bad_coh_cnt) / static_cast<double>(eval_coh_cnt);
        res.rms_mm = std::sqrt(total_sq / static_cast<double>(eval_coh_cnt));
        res.pair_rejected = (res.bad_fraction > static_cast<double>(opt.closure_max_bad_frac));
    }
    return res;
}

bool solve_linear_system(int n, std::vector<double>& mat, std::vector<double>& rhs) {
    for (int i = 0; i < n; ++i) {
        int piv = i;
        double best = std::abs(mat[i * n + i]);
        for (int r = i + 1; r < n; ++r) {
            double v = std::abs(mat[r * n + i]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < 1e-12) {
            return false;
        }
        if (piv != i) {
            for (int c = i; c < n; ++c) {
                std::swap(mat[i * n + c], mat[piv * n + c]);
            }
            std::swap(rhs[i], rhs[piv]);
        }
        double inv_diag = 1.0 / mat[i * n + i];
        for (int c = i; c < n; ++c) {
            mat[i * n + c] *= inv_diag;
        }
        rhs[i] *= inv_diag;
        for (int r = 0; r < n; ++r) {
            if (r == i) continue;
            double factor = mat[r * n + i];
            if (factor == 0.0) continue;
            for (int c = i; c < n; ++c) {
                mat[r * n + c] -= factor * mat[i * n + c];
            }
            rhs[r] -= factor * rhs[i];
        }
    }
    return true;
}

int ramp_poly_param_count(int order) {
    if (order < 0) return 0;
    if (order == 0) return 1;
    if (order == 1) return 3;
    return 6;
}

double eval_surface_basis(
    int order,
    bool use_dem,
    double u,
    double v,
    double z_km,
    const std::vector<double>& c
) {
    double val = 0.0;
    int idx = 0;
    if (order < 0) {
        if (use_dem) {
            val += c[0] + c[1] * z_km;
        }
        return val;
    }
    val += c[idx++];
    if (order >= 1) {
        val += c[idx++] * u;
        val += c[idx++] * v;
    }
    if (order == 2) {
        val += c[idx++] * u * u;
        val += c[idx++] * u * v;
        val += c[idx++] * v * v;
    }
    if (use_dem) {
        val += c[idx] * z_km;
    }
    return val;
}

double compute_pearson_corr(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() < 3 || x.size() != y.size()) {
        return kNanD;
    }
    double sx = 0.0, sy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sx += x[i];
        sy += y[i];
    }
    const double mx = sx / static_cast<double>(x.size());
    const double my = sy / static_cast<double>(y.size());
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    const double denom = std::sqrt(sxx * syy);
    return (denom > 1e-12) ? (sxy / denom) : kNanD;
}

std::vector<uint8_t> build_deform_exclusion_mask(
    const Options& opt,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const std::vector<double>& state_bands,
    const std::vector<uint8_t>& ref_mask,
    const std::vector<uint8_t>& ref_cc,
    const std::vector<float>& disp,
    const std::vector<float>& coh
) {
    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    std::vector<uint8_t> excl(npix, 0);

    if (!opt.deform_mask_path.empty()) {
        excl = load_mask_any(opt.deform_mask_path, meta);
    }

    if (opt.ramp_deform_vel_mm_yr > 0.0f) {
        std::vector<double> abs_v_map(npix, kNanD);
        bool have_vel = false;
        if (!opt.prior_vel_path.empty()) {
            std::vector<float> pvel = read_raster_aligned_f32(opt.prior_vel_path, meta, GRA_Bilinear);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < npix; ++i) {
                if (std::isfinite(pvel[i])) {
                    abs_v_map[i] = std::abs(static_cast<double>(pvel[i]));
                }
            }
            have_vel = true;
        } else if (smeta.pair_count >= 2 && !state_bands.empty()) {
            const double* s11 = state_bands.data() + static_cast<size_t>(kBandS11) * npix;
            const double* b1 = state_bands.data() + static_cast<size_t>(kBandB1) * npix;
            const double* valid_cnt = state_bands.data() + static_cast<size_t>(kBandValidCount) * npix;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < npix; ++i) {
                if (valid_cnt[i] >= 2.0 && s11[i] > 1e-16) {
                    const double v = b1[i] / s11[i];
                    if (std::isfinite(v)) {
                        abs_v_map[i] = std::abs(v);
                    }
                }
            }
            have_vel = true;
        }

        if (have_vel) {
            const bool use_cc = !ref_cc.empty();
            std::vector<double> v_samples;
            v_samples.reserve(npix / 4);
            for (size_t i = 0; i < npix; ++i) {
                if (std::isfinite(abs_v_map[i]) && std::isfinite(disp[i]) &&
                    std::isfinite(coh[i]) && coh[i] >= opt.min_coh && (!use_cc || ref_cc[i])) {
                    v_samples.push_back(abs_v_map[i]);
                }
            }
            double eff_thresh = static_cast<double>(opt.ramp_deform_vel_mm_yr);
            // Adaptively cap velocity auto-masking at the top 45% fastest-deforming pixels so
            // noisy or broadly deforming scenes never fall back to 0% masked!
            if (v_samples.size() >= 64) {
                const size_t q55 = static_cast<size_t>(0.55 * static_cast<double>(v_samples.size() - 1));
                std::nth_element(v_samples.begin(), v_samples.begin() + q55, v_samples.end());
                eff_thresh = std::max(eff_thresh, v_samples[q55]);
            }
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < npix; ++i) {
                if (std::isfinite(abs_v_map[i]) && abs_v_map[i] >= eff_thresh) {
                    excl[i] = 1;
                }
            }
        }
    }

    if (opt.ramp_mask_dilate_px > 0) {
        bool any_excl = std::any_of(excl.begin(), excl.end(), [](uint8_t v) { return v != 0; });
        if (any_excl) {
            excl = dilate_mask_box(excl, meta.width, meta.height, opt.ramp_mask_dilate_px);
        }
    }

    // Never exclude explicitly configured stable reference pixels
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        if (ref_mask[i]) {
            excl[i] = 0;
        }
    }
    return excl;
}

std::vector<uint8_t> extract_reference_connected_component(
    int width,
    int height,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& ref_mask,
    float min_coh
) {
    const size_t npix = static_cast<size_t>(width) * static_cast<size_t>(height);
    const bool has_ref = std::any_of(ref_mask.begin(), ref_mask.end(), [](uint8_t v) { return v != 0; });
    if (!has_ref) {
        return {};
    }

    auto run_bfs = [&](float coh_thr) -> std::pair<std::vector<uint8_t>, size_t> {
        std::vector<uint8_t> visited(npix, 0);
        std::vector<size_t> queue;
        queue.reserve(npix / 4);
        for (size_t i = 0; i < npix; ++i) {
            if (ref_mask[i] && std::isfinite(disp[i]) && std::isfinite(coh[i]) && coh[i] >= coh_thr) {
                visited[i] = 1;
                queue.push_back(i);
            }
        }
        size_t head = 0;
        while (head < queue.size()) {
            const size_t cur = queue[head++];
            const int y = static_cast<int>(cur / static_cast<size_t>(width));
            const int x = static_cast<int>(cur % static_cast<size_t>(width));
            auto try_push = [&](int nx, int ny) {
                const size_t ni = static_cast<size_t>(ny) * width + nx;
                if (!visited[ni] && std::isfinite(disp[ni]) && std::isfinite(coh[ni]) && coh[ni] >= coh_thr) {
                    visited[ni] = 1;
                    queue.push_back(ni);
                }
            };
            if (x > 0) try_push(x - 1, y);
            if (x + 1 < width) try_push(x + 1, y);
            if (y > 0) try_push(x, y - 1);
            if (y + 1 < height) try_push(x, y + 1);
        }
        return {std::move(visited), queue.size()};
    };

    // First try strict bridge threshold (max(min_coh, 0.45)) so decorrelated mountain/water barriers
    // cannot leak disconnected integer-fringe islands into the ramp/DEM regression.
    const float strict_thr = std::max(min_coh, 0.45f);
    auto [cc, count] = run_bfs(strict_thr);
    if (count >= 256) {
        return cc;
    }
    if (strict_thr > min_coh + 1e-6f) {
        auto [cc2, count2] = run_bfs(min_coh);
        if (count2 >= 64) {
            return cc2;
        }
    }
    return {};
}

RampResult remove_robust_ramp_and_dem(
    int width,
    int height,
    std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<float>& dem,
    const std::vector<uint8_t>& deform_excl_in,
    const std::vector<uint8_t>& ref_mask,
    const std::vector<uint8_t>& ref_cc,
    int order,
    float min_coh,
    float ramp_mask_sigma,
    int ramp_mask_dilate_px
) {
    RampResult res;
    res.order = order;
    const bool have_dem = (dem.size() == disp.size());
    if (order < 0 && !have_dem) {
        return res;
    }

    const size_t npix = static_cast<size_t>(width) * static_cast<size_t>(height);
    const bool use_cc = !ref_cc.empty();

    std::vector<size_t> all_coh_idx;
    all_coh_idx.reserve(npix / 2);
    std::vector<size_t> unmasked_idx;
    unmasked_idx.reserve(npix / 2);

    double sum_sq_before = 0.0;
    size_t excluded_prior_cnt = 0;
    for (size_t i = 0; i < npix; ++i) {
        if (std::isfinite(disp[i]) && std::isfinite(coh[i]) && coh[i] >= min_coh) {
            if (use_cc && !ref_cc[i]) {
                continue;
            }
            all_coh_idx.push_back(i);
            sum_sq_before += static_cast<double>(disp[i]) * static_cast<double>(disp[i]);
            if (deform_excl_in[i]) {
                excluded_prior_cnt++;
            } else {
                unmasked_idx.push_back(i);
            }
        }
    }

    if (all_coh_idx.size() < 16) {
        return res;
    }
    res.rms_before_mm = std::sqrt(sum_sq_before / static_cast<double>(all_coh_idx.size()));

    // Guard against over-masking
    std::vector<size_t> fit_idx = unmasked_idx;
    if (fit_idx.size() < std::max<size_t>(30, all_coh_idx.size() / 10)) {
        fit_idx = all_coh_idx;
        excluded_prior_cnt = 0;
    }

    // Check DEM coverage & topographic relief over fit_idx
    bool use_dem = false;
    double dem_mean = 0.0;
    double dem_std = 0.0;
    if (have_dem) {
        size_t dem_valid = 0;
        double sum_z = 0.0;
        double sum_z2 = 0.0;
        for (size_t idx : fit_idx) {
            if (std::isfinite(dem[idx])) {
                const double z = static_cast<double>(dem[idx]);
                sum_z += z;
                sum_z2 += z * z;
                dem_valid++;
            }
        }
        if (dem_valid >= std::max<size_t>(30, fit_idx.size() / 2)) {
            dem_mean = sum_z / static_cast<double>(dem_valid);
            const double var = std::max(0.0, sum_z2 / static_cast<double>(dem_valid) - dem_mean * dem_mean);
            dem_std = std::sqrt(var);
            res.dem_mean_m = dem_mean;
            res.dem_std_m = dem_std;
            if (dem_std >= 5.0) {
                use_dem = true;
                std::vector<size_t> dem_fit_idx;
                dem_fit_idx.reserve(dem_valid);
                for (size_t idx : fit_idx) {
                    if (std::isfinite(dem[idx])) {
                        dem_fit_idx.push_back(idx);
                    }
                }
                fit_idx = std::move(dem_fit_idx);
            }
        }
    }

    if (order < 0 && !use_dem) {
        return res;
    }

    const int poly_params = (order < 0) ? 1 : ramp_poly_param_count(order);
    const int n_params = poly_params + (use_dem ? 1 : 0);
    if (fit_idx.size() < static_cast<size_t>(std::max(12, n_params * 4))) {
        return res;
    }

    const double inv_half_w = (width > 1) ? 2.0 / static_cast<double>(width - 1) : 0.0;
    const double inv_half_h = (height > 1) ? 2.0 / static_cast<double>(height - 1) : 0.0;

    // Helper to run IRLS passes on an index subset
    auto run_irls = [&](const std::vector<size_t>& indices, std::vector<double>& coeffs, double& mad_sigma, int iter_start, int iter_end) -> bool {
        const size_t n_ind = indices.size();
        const size_t sample_step = std::max<size_t>(1, n_ind / 200000);
        std::vector<double> abs_res_sample;
        abs_res_sample.reserve((n_ind + sample_step - 1) / sample_step);

        for (int iter = iter_start; iter < iter_end; ++iter) {
            std::vector<double> ata(static_cast<size_t>(n_params) * n_params, 0.0);
            std::vector<double> aty(n_params, 0.0);
            const double huber_delta = 1.345 * mad_sigma;
            const double tukey_c = 4.685 * mad_sigma;

            #pragma omp parallel
            {
                std::vector<double> local_ata(static_cast<size_t>(n_params) * n_params, 0.0);
                std::vector<double> local_aty(n_params, 0.0);
                double phi[8] = {0.0};

                #pragma omp for schedule(static)
                for (size_t k = 0; k < n_ind; ++k) {
                    const size_t idx = indices[k];
                    const int y = static_cast<int>(idx / static_cast<size_t>(width));
                    const int x = static_cast<int>(idx % static_cast<size_t>(width));
                    const double u = static_cast<double>(x) * inv_half_w - 1.0;
                    const double v = static_cast<double>(y) * inv_half_h - 1.0;
                    const double z_km = use_dem ? (static_cast<double>(dem[idx]) - dem_mean) / 1000.0 : 0.0;

                    const double g = static_cast<double>(coh[idx]);
                    double w = g * g * g;
                    const double val = static_cast<double>(disp[idx]);
                    if (iter > 0) {
                        const double r = std::abs(val - eval_surface_basis(order, use_dem, u, v, z_km, coeffs));
                        if (iter <= 2) {
                            if (r > huber_delta) {
                                w *= (huber_delta / r);
                            }
                        } else {
                            if (r >= tukey_c) {
                                continue;
                            }
                            const double t = r / tukey_c;
                            const double s = 1.0 - t * t;
                            w *= s * s;
                        }
                    }

                    int p = 0;
                    phi[p++] = 1.0;
                    if (order >= 1) {
                        phi[p++] = u;
                        phi[p++] = v;
                    }
                    if (order == 2) {
                        phi[p++] = u * u;
                        phi[p++] = u * v;
                        phi[p++] = v * v;
                    }
                    if (use_dem) {
                        phi[p++] = z_km;
                    }
                    for (int r = 0; r < n_params; ++r) {
                        const double wr = w * phi[r];
                        local_aty[r] += wr * val;
                        for (int c = r; c < n_params; ++c) {
                            local_ata[static_cast<size_t>(r) * n_params + c] += wr * phi[c];
                        }
                    }
                }

                #pragma omp critical
                {
                    for (int r = 0; r < n_params; ++r) {
                        aty[r] += local_aty[r];
                        for (int c = r; c < n_params; ++c) {
                            ata[static_cast<size_t>(r) * n_params + c] +=
                                local_ata[static_cast<size_t>(r) * n_params + c];
                        }
                    }
                }
            }

            for (int r = 0; r < n_params; ++r) {
                ata[static_cast<size_t>(r) * n_params + r] += 1e-10;
                for (int c = 0; c < r; ++c) {
                    ata[static_cast<size_t>(r) * n_params + c] =
                        ata[static_cast<size_t>(c) * n_params + r];
                }
            }
            // Mild physical ridge damping on DEM tropospheric slope so low-relief basins cannot
            // produce unphysical +/-150 mm/km slopes while preserving unbiased slopes over real relief.
            if (use_dem) {
                const int dem_col = n_params - 1;
                const double sum_w_eff = ata[0];
                ata[static_cast<size_t>(dem_col) * n_params + dem_col] += sum_w_eff * 0.0025;
            }
            if (!solve_linear_system(n_params, ata, aty)) {
                return (iter > 0);
            }
            coeffs = aty;

            if (iter + 1 < iter_end) {
                abs_res_sample.clear();
                for (size_t k = 0; k < n_ind; k += sample_step) {
                    const size_t idx = indices[k];
                    const int y = static_cast<int>(idx / static_cast<size_t>(width));
                    const int x = static_cast<int>(idx % static_cast<size_t>(width));
                    const double u = static_cast<double>(x) * inv_half_w - 1.0;
                    const double v = static_cast<double>(y) * inv_half_h - 1.0;
                    const double z_km = use_dem ? (static_cast<double>(dem[idx]) - dem_mean) / 1000.0 : 0.0;
                    const double pred = eval_surface_basis(order, use_dem, u, v, z_km, coeffs);
                    abs_res_sample.push_back(std::abs(static_cast<double>(disp[idx]) - pred));
                }
                const size_t mid = abs_res_sample.size() / 2;
                std::nth_element(abs_res_sample.begin(), abs_res_sample.begin() + mid, abs_res_sample.end());
                mad_sigma = std::max(0.05, 1.4826 * abs_res_sample[mid]);
            }
        }
        return true;
    };

    std::vector<double> coeffs(n_params, 0.0);
    double mad_sigma = 1.0;

    // Stage 1: Initial Huber IRLS fit on prior-unmasked pixels
    if (!run_irls(fit_idx, coeffs, mad_sigma, 0, 2)) {
        return res;
    }

    // Stage 2: Explicit single-pair residual masking of localized high-deformation bowls + dilation
    std::vector<size_t> stable_idx = fit_idx;
    size_t resid_masked_cnt = 0;
    if (ramp_mask_sigma > 0.0f) {
        const double resid_thresh = static_cast<double>(ramp_mask_sigma) * mad_sigma;
        std::vector<uint8_t> anom_mask(npix, 0);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < fit_idx.size(); ++k) {
            const size_t idx = fit_idx[k];
            if (ref_mask[idx]) continue;
            const int y = static_cast<int>(idx / static_cast<size_t>(width));
            const int x = static_cast<int>(idx % static_cast<size_t>(width));
            const double u = static_cast<double>(x) * inv_half_w - 1.0;
            const double v = static_cast<double>(y) * inv_half_h - 1.0;
            const double z_km = use_dem ? (static_cast<double>(dem[idx]) - dem_mean) / 1000.0 : 0.0;
            const double r = std::abs(static_cast<double>(disp[idx]) - eval_surface_basis(order, use_dem, u, v, z_km, coeffs));
            if (r > resid_thresh) {
                anom_mask[idx] = 1;
            }
        }
        if (ramp_mask_dilate_px > 0) {
            anom_mask = dilate_mask_box(anom_mask, width, height, std::min(2, ramp_mask_dilate_px));
        }
        std::vector<size_t> filtered;
        filtered.reserve(fit_idx.size());
        for (size_t idx : fit_idx) {
            if (!anom_mask[idx] || ref_mask[idx]) {
                filtered.push_back(idx);
            } else {
                resid_masked_cnt++;
            }
        }
        if (filtered.size() >= std::max<size_t>(30, fit_idx.size() / 5)) {
            stable_idx = std::move(filtered);
        } else {
            resid_masked_cnt = 0;
        }
    }

    // Stage 3: Final Tukey biweight IRLS refinement strictly on stable background pixels
    run_irls(stable_idx, coeffs, mad_sigma, 2, 6);

    // Pre-correction DEM correlation diagnostic on stable pixels
    if (use_dem) {
        const size_t step = std::max<size_t>(1, stable_idx.size() / 100000);
        std::vector<double> zs, d_before, d_after;
        zs.reserve(stable_idx.size() / step + 1);
        d_before.reserve(zs.capacity());
        d_after.reserve(zs.capacity());
        for (size_t k = 0; k < stable_idx.size(); k += step) {
            const size_t idx = stable_idx[k];
            const int y = static_cast<int>(idx / static_cast<size_t>(width));
            const int x = static_cast<int>(idx % static_cast<size_t>(width));
            const double u = static_cast<double>(x) * inv_half_w - 1.0;
            const double v = static_cast<double>(y) * inv_half_h - 1.0;
            const double z_km = (static_cast<double>(dem[idx]) - dem_mean) / 1000.0;
            const double val = static_cast<double>(disp[idx]);
            zs.push_back(z_km);
            d_before.push_back(val);
            d_after.push_back(val - eval_surface_basis(order, true, u, v, z_km, coeffs));
        }
        res.dem_corr_before = compute_pearson_corr(zs, d_before);
        res.dem_corr_after = compute_pearson_corr(zs, d_after);
        res.dem_applied = true;
        res.dem_slope_mm_per_km = coeffs.back();
    }

    double sum_sq_after = 0.0;
    #pragma omp parallel for reduction(+:sum_sq_after) schedule(static)
    for (size_t idx = 0; idx < npix; ++idx) {
        if (!std::isfinite(disp[idx])) continue;
        const int y = static_cast<int>(idx / static_cast<size_t>(width));
        const int x = static_cast<int>(idx % static_cast<size_t>(width));
        const double u = static_cast<double>(x) * inv_half_w - 1.0;
        const double v = static_cast<double>(y) * inv_half_h - 1.0;
        const bool pix_has_dem = use_dem && std::isfinite(dem[idx]);
        const double z_km = pix_has_dem ? (static_cast<double>(dem[idx]) - dem_mean) / 1000.0 : 0.0;
        const double pred = eval_surface_basis(order, pix_has_dem, u, v, z_km, coeffs);
        const double corrected = static_cast<double>(disp[idx]) - pred;
        disp[idx] = static_cast<float>(corrected);
        if (std::isfinite(coh[idx]) && coh[idx] >= min_coh && (!use_cc || ref_cc[idx])) {
            sum_sq_after += corrected * corrected;
        }
    }

    res.applied = true;
    res.coeffs = std::move(coeffs);
    res.rms_after_mm = std::sqrt(sum_sq_after / static_cast<double>(all_coh_idx.size()));
    res.masked_deform_pixels = excluded_prior_cnt + resid_masked_cnt;
    res.masked_deform_fraction = static_cast<double>(res.masked_deform_pixels) / static_cast<double>(all_coh_idx.size());
    return res;
}

void update_state_incremental(
    std::vector<double>& state,
    size_t npix,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    double t0_days,
    double t1_days,
    double origin_days,
    float min_coh,
    bool redundant_pair,
    double weight_scale
) {
    const double tau_yr = (t1_days - t0_days) / kDaysPerYear;
    const double m_yr = (0.5 * (t0_days + t1_days) - origin_days) / kDaysPerYear;
    const double x1 = tau_yr;
    const double x2 = m_yr * tau_yr;
    const double x11 = x1 * x1;
    const double x12 = x1 * x2;
    const double x22 = x2 * x2;

    double* sum_w = state.data() + static_cast<size_t>(kBandSumW) * npix;
    double* s11 = state.data() + static_cast<size_t>(kBandS11) * npix;
    double* s12 = state.data() + static_cast<size_t>(kBandS12) * npix;
    double* s22 = state.data() + static_cast<size_t>(kBandS22) * npix;
    double* b1 = state.data() + static_cast<size_t>(kBandB1) * npix;
    double* b2 = state.data() + static_cast<size_t>(kBandB2) * npix;
    double* syy = state.data() + static_cast<size_t>(kBandSyy) * npix;
    double* cum_raw = state.data() + static_cast<size_t>(kBandCumDispRaw) * npix;
    double* valid_dt = state.data() + static_cast<size_t>(kBandValidDtYr) * npix;
    double* sum_coh = state.data() + static_cast<size_t>(kBandSumCoh) * npix;
    double* valid_cnt = state.data() + static_cast<size_t>(kBandValidCount) * npix;
    double* fp_cnt = state.data() + static_cast<size_t>(kBandFootprintCount) * npix;
    double* latest = state.data() + static_cast<size_t>(kBandLatestDisp) * npix;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        if (!redundant_pair) {
            latest[i] = kNanD;
        }
        const double c = static_cast<double>(coh[i]);
        const double d = static_cast<double>(disp[i]);
        if (!std::isfinite(c)) {
            continue;
        }
        const double g = std::clamp(c, 0.0, 1.0);
        if (std::isfinite(d)) {
            if (!redundant_pair || fp_cnt[i] == 0.0) {
                fp_cnt[i] += 1.0;
                sum_coh[i] += g;
            }
        }
        if (!std::isfinite(d) || c < static_cast<double>(min_coh)) {
            continue;
        }
        const double w = std::max(1e-4, g * g * weight_scale);
        sum_w[i] += w;
        s11[i] += w * x11;
        s12[i] += w * x12;
        s22[i] += w * x22;
        b1[i] += w * x1 * d;
        b2[i] += w * x2 * d;
        syy[i] += w * d * d;
        valid_cnt[i] += 1.0;
        if (!redundant_pair) {
            cum_raw[i] += d;
            valid_dt[i] += tau_yr;
            latest[i] = d;
        }
    }
}

struct DerivedFields {
    std::vector<float> vel;
    std::vector<float> accel;
    std::vector<float> cum_disp;
    std::vector<float> temp_coh;
    std::vector<float> unc;
    std::vector<float> snr;
    std::vector<float> latest_disp;
    std::vector<double> vel_d;
    std::vector<double> accel_d;
    std::vector<double> cum_d;
    std::vector<double> coh_d;
    std::vector<double> unc_d;
    std::vector<double> snr_d;
    std::vector<double> latest_d;
    std::vector<int> valid_cnt;
    size_t valid_stack_pixels = 0;
    double mean_vel = kNanD;
    double p95_abs_vel = kNanD;
    double mean_coh = kNanD;
};

DerivedFields derive_products(
    const std::vector<double>& state,
    size_t npix,
    const StackMetadata& smeta,
    float min_coh
) {
    DerivedFields out;
    out.vel.assign(npix, kNanF);
    out.accel.assign(npix, kNanF);
    out.cum_disp.assign(npix, kNanF);
    out.temp_coh.assign(npix, kNanF);
    out.unc.assign(npix, kNanF);
    out.snr.assign(npix, kNanF);
    out.latest_disp.assign(npix, kNanF);

    out.vel_d.assign(npix, kNanD);
    out.accel_d.assign(npix, kNanD);
    out.cum_d.assign(npix, kNanD);
    out.coh_d.assign(npix, kNanD);
    out.unc_d.assign(npix, kNanD);
    out.snr_d.assign(npix, kNanD);
    out.latest_d.assign(npix, kNanD);
    out.valid_cnt.assign(npix, 0);

    const double span_days = std::max(smeta.sum_dt_days, smeta.last_t1_days - smeta.first_t0_days);
    const double total_span_yr = span_days / kDaysPerYear;
    const double* s11 = state.data() + static_cast<size_t>(kBandS11) * npix;
    const double* s12 = state.data() + static_cast<size_t>(kBandS12) * npix;
    const double* s22 = state.data() + static_cast<size_t>(kBandS22) * npix;
    const double* b1 = state.data() + static_cast<size_t>(kBandB1) * npix;
    const double* b2 = state.data() + static_cast<size_t>(kBandB2) * npix;
    const double* syy = state.data() + static_cast<size_t>(kBandSyy) * npix;
    const double* cum_raw = state.data() + static_cast<size_t>(kBandCumDispRaw) * npix;
    const double* valid_dt = state.data() + static_cast<size_t>(kBandValidDtYr) * npix;
    const double* sum_coh = state.data() + static_cast<size_t>(kBandSumCoh) * npix;
    const double* valid_cnt = state.data() + static_cast<size_t>(kBandValidCount) * npix;
    const double* fp_cnt = state.data() + static_cast<size_t>(kBandFootprintCount) * npix;
    const double* latest = state.data() + static_cast<size_t>(kBandLatestDisp) * npix;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        int fc = static_cast<int>(std::llround(fp_cnt[i]));
        if (fc <= 0) {
            continue;
        }
        double tcoh = sum_coh[i] / static_cast<double>(fc);
        out.coh_d[i] = tcoh;
        out.temp_coh[i] = static_cast<float>(tcoh);

        int vc = static_cast<int>(std::llround(valid_cnt[i]));
        out.valid_cnt[i] = vc;
        if (std::isfinite(latest[i])) {
            out.latest_d[i] = latest[i];
            out.latest_disp[i] = static_cast<float>(latest[i]);
        }

        if (vc < 1 || s11[i] <= 1e-16 || tcoh < static_cast<double>(min_coh) * 0.5) {
            continue;
        }
        if (smeta.pair_count >= 3 && vc < std::max(2, static_cast<int>(std::ceil(0.35 * smeta.pair_count)))) {
            continue;
        }
        double v_lin = b1[i] / s11[i];
        double cdisp = cum_raw[i];
        if (total_span_yr > valid_dt[i] + 1e-9) {
            cdisp += (total_span_yr - valid_dt[i]) * v_lin;
        }

        out.vel_d[i] = v_lin;
        out.vel[i] = static_cast<float>(v_lin);
        out.cum_d[i] = cdisp;
        out.cum_disp[i] = static_cast<float>(cdisp);

        if (vc >= 2) {
            double wssr = std::max(0.0, syy[i] - v_lin * b1[i]);
            double se_v = std::sqrt(wssr / (static_cast<double>(vc - 1) * s11[i]));
            out.unc_d[i] = se_v;
            out.unc[i] = static_cast<float>(se_v);
            double snr_val = (se_v > 1e-9)
                ? (std::abs(v_lin) / se_v)
                : ((std::abs(v_lin) > 1e-6) ? 1e6 : 0.0);
            out.snr_d[i] = snr_val;
            out.snr[i] = static_cast<float>(std::min(snr_val, 1e6));
        }

        if (vc >= 3 && s11[i] > 0.0 && s22[i] > 0.0) {
            double det = s11[i] * s22[i] - s12[i] * s12[i];
            double norm_det = det / (s11[i] * s22[i]);
            if (norm_det > 1e-6) {
                double a_hat = (s11[i] * b2[i] - s12[i] * b1[i]) / det;
                out.accel_d[i] = a_hat;
                out.accel[i] = static_cast<float>(a_hat);
            }
        }
    }

    std::vector<double> abs_vels;
    abs_vels.reserve(npix / 4);
    double sum_v = 0.0;
    double sum_c = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        if (std::isfinite(out.vel_d[i])) {
            out.valid_stack_pixels++;
            sum_v += out.vel_d[i];
            sum_c += out.coh_d[i];
            abs_vels.push_back(std::abs(out.vel_d[i]));
        }
    }
    if (!abs_vels.empty()) {
        out.mean_vel = sum_v / static_cast<double>(abs_vels.size());
        out.mean_coh = sum_c / static_cast<double>(abs_vels.size());
        size_t p95_idx = static_cast<size_t>(std::floor(0.95 * static_cast<double>(abs_vels.size() - 1)));
        std::nth_element(abs_vels.begin(), abs_vels.begin() + p95_idx, abs_vels.end());
        out.p95_abs_vel = abs_vels[p95_idx];
    }
    return out;
}

std::string get_feature_str(OGRFeature* feat, const char* field, const std::string& fallback = "") {
    int idx = feat->GetFieldIndex(field);
    if (idx < 0 || !feat->IsFieldSetAndNotNull(idx)) {
        return fallback;
    }
    const char* val = feat->GetFieldAsString(idx);
    return (val != nullptr && *val != '\0') ? std::string(val) : fallback;
}

std::string classify_corridor_risk(
    int valid_pixels,
    int alert_pixel_count,
    double mean_vel,
    double p95_abs_vel,
    float vel_alert_mm_yr,
    bool stack_mature,
    bool& maturity_gated_out
) {
    maturity_gated_out = false;
    if (valid_pixels <= 0 || !std::isfinite(p95_abs_vel)) {
        return "no_coverage";
    }
    const int min_crit_alerts = (valid_pixels >= 100) ? 3 : 1;
    const bool meets_critical =
        alert_pixel_count >= min_crit_alerts &&
        (p95_abs_vel >= 2.0 * vel_alert_mm_yr || alert_pixel_count >= 10 || std::abs(mean_vel) >= vel_alert_mm_yr);
    if (meets_critical) {
        if (stack_mature) {
            return "critical";
        }
        maturity_gated_out = true;
        return "warning";
    }
    if (alert_pixel_count > 0 || p95_abs_vel >= vel_alert_mm_yr) {
        return "warning";
    }
    if (p95_abs_vel >= 0.5 * vel_alert_mm_yr) {
        return "watch";
    }
    return "nominal";
}

void process_corridors(
    const Options& opt,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const DerivedFields& df,
    bool stack_mature,
    std::vector<CorridorMetric>& metrics_out,
    std::vector<AlertPoint>& alerts_out,
    size_t& total_snr_gated_out
) {
    total_snr_gated_out = 0;
    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    if (opt.corridors_path.empty()) {
        return;
    }
    if (meta.wkt.empty()) {
        throw std::runtime_error("Stack raster has empty CRS WKT; cannot project corridors");
    }

    UniqueGdalDataset vds(static_cast<GDALDataset*>(
        GDALOpenEx(opt.corridors_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    if (!vds) {
        throw std::runtime_error("Failed to open corridors vector file: " + opt.corridors_path);
    }

    OGRSpatialReference dst_srs;
    if (dst_srs.importFromWkt(meta.wkt.c_str()) != OGRERR_NONE) {
        throw std::runtime_error("Invalid raster CRS WKT");
    }
    dst_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    OGRSpatialReference wgs84_srs;
    wgs84_srs.SetWellKnownGeogCS("WGS84");
    wgs84_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    UniqueOgrCt to_wgs84(OGRCreateCoordinateTransformation(&dst_srs, &wgs84_srs));
    if (!to_wgs84) {
        throw std::runtime_error("Failed to create raster->WGS84 coordinate transform");
    }

    const double rx0 = meta.geotransform[0];
    const double dx = meta.geotransform[1];
    const double ry0 = meta.geotransform[3];
    const double dy = meta.geotransform[5];
    const double r_xmin = std::min(rx0, rx0 + meta.width * dx);
    const double r_xmax = std::max(rx0, rx0 + meta.width * dx);
    const double r_ymin = std::min(ry0, ry0 + meta.height * dy);
    const double r_ymax = std::max(ry0, ry0 + meta.height * dy);

    double cx = 0.5 * (r_xmin + r_xmax);
    double cy = 0.5 * (r_ymin + r_ymax);
    to_wgs84->Transform(1, &cx, &cy);
    const double buf_units = (opt.buffer_m > 0.0) ? meters_to_crs_units(dst_srs, opt.buffer_m, cy) : 0.0;

    GDALDriver* mem_drv = GetGDALDriverManager()->GetDriverByName("MEM");
    if (mem_drv == nullptr) {
        throw std::runtime_error("MEM driver unavailable");
    }

    std::vector<int32_t> owner_corridor(npix, -1);
    std::vector<double> owner_best_vel(npix, 0.0);
    const int total_pairs_any = smeta.pair_count + smeta.redundant_pair_count;
    const int min_alert_passes = (opt.min_alert_passes > 0)
        ? opt.min_alert_passes
        : ((total_pairs_any >= 2 || opt.min_alert_snr > 0.0f) ? 2 : 1);

    const int n_layers = vds->GetLayerCount();
    int feature_seq = 0;
    for (int li = 0; li < n_layers; ++li) {
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

        UniqueOgrCt to_dst(OGRCreateCoordinateTransformation(&src_srs, &dst_srs));
        UniqueOgrCt src_to_wgs(OGRCreateCoordinateTransformation(&src_srs, &wgs84_srs));
        if (!to_dst) {
            throw std::runtime_error("Failed to create vector->raster coordinate transform");
        }

        OGRFeature* raw_feat = nullptr;
        while ((raw_feat = layer->GetNextFeature()) != nullptr) {
            std::unique_ptr<OGRFeature, void(*)(OGRFeature*)> feat(raw_feat, OGRFeature::DestroyFeature);
            OGRGeometry* g_orig = feat->GetGeometryRef();
            if (g_orig == nullptr || g_orig->IsEmpty()) {
                continue;
            }
            feature_seq++;

            CorridorMetric cm;
            cm.asset_id = get_feature_str(feat.get(), "asset_id", "corridor_" + std::to_string(feature_seq));
            cm.name = get_feature_str(feat.get(), "name", cm.asset_id);
            cm.asset_class = get_feature_str(feat.get(), "asset_class", "infrastructure");
            cm.source_ref = get_feature_str(feat.get(), "ref", "");

            UniqueOgrGeom g_wgs(g_orig->clone());
            if (src_to_wgs && g_wgs->transform(src_to_wgs.get()) == OGRERR_NONE) {
                char* gj = g_wgs->exportToJson();
                if (gj != nullptr) {
                    cm.wgs84_geom_json = gj;
                    CPLFree(gj);
                }
            }
            if (cm.wgs84_geom_json.empty()) {
                cm.wgs84_geom_json = "null";
            }

            UniqueOgrGeom g_dst(g_orig->clone());
            if (g_dst->transform(to_dst.get()) != OGRERR_NONE) {
                metrics_out.push_back(std::move(cm));
                continue;
            }
            UniqueOgrGeom g_buf;
            if (buf_units > 0.0) {
                g_buf.reset(g_dst->Buffer(buf_units, 4));
            } else {
                g_buf.reset(g_dst->clone());
            }
            if (!g_buf || g_buf->IsEmpty()) {
                metrics_out.push_back(std::move(cm));
                continue;
            }

            OGREnvelope env;
            g_buf->getEnvelope(&env);
            if (env.MaxX < r_xmin || env.MinX > r_xmax || env.MaxY < r_ymin || env.MinY > r_ymax) {
                metrics_out.push_back(std::move(cm));
                continue;
            }

            int x0 = std::clamp(static_cast<int>(std::floor((env.MinX - rx0) / dx)) - 1, 0, meta.width - 1);
            int x1 = std::clamp(static_cast<int>(std::ceil((env.MaxX - rx0) / dx)) + 1, 0, meta.width - 1);
            int py_a = static_cast<int>(std::floor((env.MaxY - ry0) / dy));
            int py_b = static_cast<int>(std::ceil((env.MinY - ry0) / dy));
            if (dy > 0) {
                py_a = static_cast<int>(std::floor((env.MinY - ry0) / dy));
                py_b = static_cast<int>(std::ceil((env.MaxY - ry0) / dy));
            }
            int y0 = std::clamp(std::min(py_a, py_b) - 1, 0, meta.height - 1);
            int y1 = std::clamp(std::max(py_a, py_b) + 1, 0, meta.height - 1);
            const int pw = x1 - x0 + 1;
            const int ph = y1 - y0 + 1;
            if (pw <= 0 || ph <= 0) {
                metrics_out.push_back(std::move(cm));
                continue;
            }

            UniqueGdalDataset sub_ds(mem_drv->Create("", pw, ph, 1, GDT_Byte, nullptr));
            if (!sub_ds) {
                throw std::runtime_error("Failed allocating corridor rasterization buffer");
            }
            double sub_gt[6] = {
                rx0 + x0 * dx, dx, 0.0,
                ry0 + y0 * dy, 0.0, dy
            };
            sub_ds->SetGeoTransform(sub_gt);
            if (!meta.wkt.empty()) {
                sub_ds->SetProjection(meta.wkt.c_str());
            }
            std::vector<uint8_t> mask(static_cast<size_t>(pw) * static_cast<size_t>(ph), 0);
            if (sub_ds->GetRasterBand(1)->RasterIO(
                    GF_Write, 0, 0, pw, ph, mask.data(), pw, ph, GDT_Byte, 0, 0, nullptr) != CE_None) {
                throw std::runtime_error("Failed initializing corridor mask buffer");
            }

            OGRGeometryH geoms[1] = { OGRGeometry::ToHandle(g_buf.get()) };
            int bands[1] = {1};
            double burn[1] = {1.0};
            char** r_opts = nullptr;
            r_opts = CSLSetNameValue(r_opts, "ALL_TOUCHED", "TRUE");
            CPLErr r_err = GDALRasterizeGeometries(
                GDALDataset::ToHandle(sub_ds.get()), 1, bands,
                1, geoms, nullptr, nullptr, burn, r_opts, nullptr, nullptr);
            CSLDestroy(r_opts);
            if (r_err != CE_None) {
                throw std::runtime_error("GDALRasterizeGeometries failed for corridor " + cm.name);
            }
            if (sub_ds->GetRasterBand(1)->RasterIO(
                    GF_Read, 0, 0, pw, ph, mask.data(), pw, ph, GDT_Byte, 0, 0, nullptr) != CE_None) {
                throw std::runtime_error("Failed reading corridor mask buffer");
            }

            const int corridor_idx = static_cast<int>(metrics_out.size());
            std::vector<double> abs_vels;
            std::vector<double> snrs;
            double sum_vel = 0.0, sum_coh = 0.0, sum_cum = 0.0, sum_acc = 0.0, sum_lat = 0.0, sum_snr = 0.0;
            double min_v = std::numeric_limits<double>::infinity();
            double max_v = -std::numeric_limits<double>::infinity();
            double max_abs_cum = 0.0, max_abs_acc = 0.0;
            int acc_cnt = 0, lat_cnt = 0, snr_cnt = 0;

            for (int sy = 0; sy < ph; ++sy) {
                const int gy = y0 + sy;
                for (int sx = 0; sx < pw; ++sx) {
                    if (mask[static_cast<size_t>(sy) * pw + sx] == 0) continue;
                    const size_t gi = static_cast<size_t>(gy) * meta.width + (x0 + sx);
                    cm.pixel_count++;
                    if (!std::isfinite(df.vel_d[gi])) continue;

                    cm.valid_pixels++;
                    const double v = df.vel_d[gi];
                    const double av = std::abs(v);
                    const double c = df.coh_d[gi];
                    const double cd = df.cum_d[gi];
                    abs_vels.push_back(av);
                    sum_vel += v;
                    sum_coh += c;
                    sum_cum += cd;
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    max_abs_cum = std::max(max_abs_cum, std::abs(cd));

                    const bool has_snr = std::isfinite(df.snr_d[gi]);
                    if (has_snr) {
                        const double s_capped = std::min(df.snr_d[gi], 1e4);
                        snrs.push_back(s_capped);
                        sum_snr += s_capped;
                        snr_cnt++;
                    }
                    if (std::isfinite(df.accel_d[gi])) {
                        sum_acc += df.accel_d[gi];
                        max_abs_acc = std::max(max_abs_acc, std::abs(df.accel_d[gi]));
                        acc_cnt++;
                    }
                    if (std::isfinite(df.latest_d[gi])) {
                        sum_lat += df.latest_d[gi];
                        lat_cnt++;
                    }

                    const bool high_coh = (c >= static_cast<double>(opt.min_alert_coh));
                    if (high_coh) {
                        cm.high_coh_pixels++;
                    }
                    const bool exceeds_motion =
                        (av >= static_cast<double>(opt.vel_alert_mm_yr) ||
                         (std::isfinite(df.accel_d[gi]) && std::abs(df.accel_d[gi]) >= static_cast<double>(opt.accel_alert_mm_yr2)));
                    const bool passes_snr = (opt.min_alert_snr <= 0.0f)
                        ? true
                        : (has_snr && df.snr_d[gi] >= static_cast<double>(opt.min_alert_snr));

                    if (high_coh && df.valid_cnt[gi] >= min_alert_passes && exceeds_motion) {
                        if (!passes_snr) {
                            cm.snr_gated_pixels++;
                            total_snr_gated_out++;
                        } else {
                            cm.alert_pixel_count++;
                            const double score = av;
                            if (owner_corridor[gi] < 0 || score > owner_best_vel[gi]) {
                                owner_corridor[gi] = corridor_idx;
                                owner_best_vel[gi] = score;
                            }
                        }
                    }
                }
            }

            if (cm.pixel_count > 0) {
                cm.coverage_ratio = static_cast<double>(cm.valid_pixels) / static_cast<double>(cm.pixel_count);
            }
            if (cm.valid_pixels > 0) {
                cm.mean_coherence = sum_coh / static_cast<double>(cm.valid_pixels);
                cm.mean_velocity_mm_yr = sum_vel / static_cast<double>(cm.valid_pixels);
                cm.min_velocity_mm_yr = min_v;
                cm.max_velocity_mm_yr = max_v;
                cm.mean_cumulative_disp_mm = sum_cum / static_cast<double>(cm.valid_pixels);
                cm.max_abs_cumulative_disp_mm = max_abs_cum;
                size_t p95 = static_cast<size_t>(std::floor(0.95 * static_cast<double>(abs_vels.size() - 1)));
                std::nth_element(abs_vels.begin(), abs_vels.begin() + p95, abs_vels.end());
                cm.p95_abs_velocity_mm_yr = abs_vels[p95];
                if (snr_cnt > 0) {
                    cm.mean_snr = sum_snr / static_cast<double>(snr_cnt);
                    size_t p95s = static_cast<size_t>(std::floor(0.95 * static_cast<double>(snrs.size() - 1)));
                    std::nth_element(snrs.begin(), snrs.begin() + p95s, snrs.end());
                    cm.p95_snr = snrs[p95s];
                }
                if (acc_cnt > 0) {
                    cm.mean_acceleration_mm_yr2 = sum_acc / static_cast<double>(acc_cnt);
                    cm.max_abs_acceleration_mm_yr2 = max_abs_acc;
                }
                if (lat_cnt > 0) {
                    cm.latest_interval_disp_mm = sum_lat / static_cast<double>(lat_cnt);
                }
            }
            cm.risk_level = classify_corridor_risk(
                cm.valid_pixels, cm.alert_pixel_count, cm.mean_velocity_mm_yr,
                cm.p95_abs_velocity_mm_yr, opt.vel_alert_mm_yr, stack_mature, cm.maturity_gated);
            metrics_out.push_back(std::move(cm));
        }
    }

    std::vector<AlertPoint> candidates;
    for (int py = 0; py < meta.height; ++py) {
        for (int px = 0; px < meta.width; ++px) {
            const size_t gi = static_cast<size_t>(py) * meta.width + px;
            int cid = owner_corridor[gi];
            if (cid < 0) continue;
            AlertPoint pt;
            pt.px = px;
            pt.py = py;
            pt.velocity_mm_yr = df.vel_d[gi];
            pt.acceleration_mm_yr2 = df.accel_d[gi];
            pt.cumulative_disp_mm = df.cum_d[gi];
            pt.temporal_coherence = df.coh_d[gi];
            pt.uncertainty_mm_yr = df.unc_d[gi];
            pt.velocity_snr = df.snr_d[gi];
            pt.severity = stack_mature ? "critical" : "provisional";
            pt.corridor_asset_id = metrics_out[cid].asset_id;
            pt.corridor_name = metrics_out[cid].name;
            pt.score = std::abs(pt.velocity_mm_yr) +
                (std::isfinite(pt.acceleration_mm_yr2) ? 0.25 * std::abs(pt.acceleration_mm_yr2) : 0.0);
            candidates.push_back(std::move(pt));
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const AlertPoint& a, const AlertPoint& b) {
        return a.score > b.score;
    });

    constexpr int kNmsRadiusPx = 5;
    constexpr size_t kMaxCorridorAlerts = 250;
    for (auto& pt : candidates) {
        bool too_close = false;
        for (const auto& kept : alerts_out) {
            int ddx = pt.px - kept.px;
            int ddy = pt.py - kept.py;
            if (ddx * ddx + ddy * ddy <= kNmsRadiusPx * kNmsRadiusPx) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;
        double wx = rx0 + (pt.px + 0.5) * dx;
        double wy = ry0 + (pt.py + 0.5) * dy;
        to_wgs84->Transform(1, &wx, &wy);
        pt.lon = wx;
        pt.lat = wy;
        alerts_out.push_back(std::move(pt));
        if (alerts_out.size() >= kMaxCorridorAlerts) {
            break;
        }
    }
}

void write_corridor_outputs(
    const fs::path& out_dir,
    const Options& opt,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const RampResult& ramp,
    const ReferencePinResult& ref_pin,
    const ClosureResult& closure,
    const DerivedFields& df,
    bool stack_mature,
    double calendar_span_days,
    size_t total_snr_gated,
    const std::vector<CorridorMetric>& metrics,
    const std::vector<AlertPoint>& alerts
) {
    if (!opt.corridors_path.empty()) {
        std::ofstream rep(out_dir / "corridor_report.geojson");
        rep << "{\n  \"type\": \"FeatureCollection\",\n  \"name\": \"corridor_report\",\n  \"features\": [\n";
        for (size_t i = 0; i < metrics.size(); ++i) {
            const auto& m = metrics[i];
            rep << "    {\n"
                << "      \"type\": \"Feature\",\n"
                << "      \"properties\": {\n"
                << "        \"asset_id\": \"" << json_escape(m.asset_id) << "\",\n"
                << "        \"name\": \"" << json_escape(m.name) << "\",\n"
                << "        \"asset_class\": \"" << json_escape(m.asset_class) << "\",\n"
                << "        \"ref\": \"" << json_escape(m.source_ref) << "\",\n"
                << "        \"risk_level\": \"" << m.risk_level << "\",\n"
                << "        \"maturity_gated\": " << (m.maturity_gated ? "true" : "false") << ",\n"
                << "        \"pixel_count\": " << m.pixel_count << ",\n"
                << "        \"valid_pixels\": " << m.valid_pixels << ",\n"
                << "        \"high_coh_pixels\": " << m.high_coh_pixels << ",\n"
                << "        \"alert_pixel_count\": " << m.alert_pixel_count << ",\n"
                << "        \"snr_gated_pixels\": " << m.snr_gated_pixels << ",\n"
                << "        \"coverage_ratio\": " << json_num(m.coverage_ratio, 4) << ",\n"
                << "        \"mean_coherence\": " << json_num(m.mean_coherence, 4) << ",\n"
                << "        \"mean_velocity_mm_yr\": " << json_num(m.mean_velocity_mm_yr, 3) << ",\n"
                << "        \"min_velocity_mm_yr\": " << json_num(m.min_velocity_mm_yr, 3) << ",\n"
                << "        \"max_velocity_mm_yr\": " << json_num(m.max_velocity_mm_yr, 3) << ",\n"
                << "        \"p95_abs_velocity_mm_yr\": " << json_num(m.p95_abs_velocity_mm_yr, 3) << ",\n"
                << "        \"mean_snr\": " << json_num(m.mean_snr, 3) << ",\n"
                << "        \"p95_snr\": " << json_num(m.p95_snr, 3) << ",\n"
                << "        \"mean_cumulative_disp_mm\": " << json_num(m.mean_cumulative_disp_mm, 3) << ",\n"
                << "        \"max_abs_cumulative_disp_mm\": " << json_num(m.max_abs_cumulative_disp_mm, 3) << ",\n"
                << "        \"mean_acceleration_mm_yr2\": " << json_num(m.mean_acceleration_mm_yr2, 3) << ",\n"
                << "        \"max_abs_acceleration_mm_yr2\": " << json_num(m.max_abs_acceleration_mm_yr2, 3) << ",\n"
                << "        \"latest_interval_disp_mm\": " << json_num(m.latest_interval_disp_mm, 3) << "\n"
                << "      },\n"
                << "      \"geometry\": " << m.wgs84_geom_json << "\n"
                << "    }" << (i + 1 < metrics.size() ? "," : "") << "\n";
        }
        rep << "  ]\n}\n";

        std::ofstream af(out_dir / "corridor_alerts.geojson");
        af << "{\n  \"type\": \"FeatureCollection\",\n  \"name\": \"corridor_alerts\",\n  \"features\": [\n";
        for (size_t i = 0; i < alerts.size(); ++i) {
            const auto& a = alerts[i];
            af << "    {\n"
               << "      \"type\": \"Feature\",\n"
               << "      \"geometry\": {\"type\": \"Point\", \"coordinates\": ["
               << std::fixed << std::setprecision(6) << a.lon << ", " << a.lat << "]},\n"
               << "      \"properties\": {\n"
               << "        \"corridor_asset_id\": \"" << json_escape(a.corridor_asset_id) << "\",\n"
               << "        \"corridor_name\": \"" << json_escape(a.corridor_name) << "\",\n"
               << "        \"severity\": \"" << a.severity << "\",\n"
               << "        \"stack_mature\": " << (stack_mature ? "true" : "false") << ",\n"
               << "        \"velocity_mm_yr\": " << json_num(a.velocity_mm_yr, 3) << ",\n"
               << "        \"acceleration_mm_yr2\": " << json_num(a.acceleration_mm_yr2, 3) << ",\n"
               << "        \"cumulative_disp_mm\": " << json_num(a.cumulative_disp_mm, 3) << ",\n"
               << "        \"temporal_coherence\": " << json_num(a.temporal_coherence, 4) << ",\n"
               << "        \"uncertainty_mm_yr\": " << json_num(a.uncertainty_mm_yr, 3) << ",\n"
               << "        \"velocity_snr\": " << json_num(std::min(a.velocity_snr, 1e4), 3) << ",\n"
               << "        \"pixel_x\": " << a.px << ",\n"
               << "        \"pixel_y\": " << a.py << "\n"
               << "      }\n"
               << "    }" << (i + 1 < alerts.size() ? "," : "") << "\n";
        }
        af << "  ]\n}\n";
    }

    int crit_cnt = 0, warn_cnt = 0, watch_cnt = 0, maturity_gated_corridors = 0;
    for (const auto& m : metrics) {
        if (m.risk_level == "critical") crit_cnt++;
        else if (m.risk_level == "warning") warn_cnt++;
        else if (m.risk_level == "watch") watch_cnt++;
        if (m.maturity_gated) maturity_gated_corridors++;
    }

    std::ofstream sf(out_dir / "stack_summary.json");
    sf << "{\n"
       << "  \"width\": " << meta.width << ",\n"
       << "  \"height\": " << meta.height << ",\n"
       << "  \"pair_count\": " << smeta.pair_count << ",\n"
       << "  \"redundant_pair_count\": " << smeta.redundant_pair_count << ",\n"
       << "  \"rejected_pair_count\": " << smeta.rejected_pair_count << ",\n"
       << "  \"first_t0_days\": " << json_num(smeta.first_t0_days, 3) << ",\n"
       << "  \"last_t1_days\": " << json_num(smeta.last_t1_days, 3) << ",\n"
       << "  \"total_span_days\": " << json_num(smeta.sum_dt_days, 3) << ",\n"
       << "  \"calendar_span_days\": " << json_num(calendar_span_days, 3) << ",\n"
       << "  \"valid_stack_pixels\": " << df.valid_stack_pixels << ",\n"
       << "  \"mean_temporal_coherence\": " << json_num(df.mean_coh, 4) << ",\n"
       << "  \"mean_velocity_mm_yr\": " << json_num(df.mean_vel, 3) << ",\n"
       << "  \"p95_abs_velocity_mm_yr\": " << json_num(df.p95_abs_vel, 3) << ",\n"
       << "  \"reference_region\": {\n"
       << "    \"configured\": " << (ref_pin.configured ? "true" : "false") << ",\n"
       << "    \"applied\": " << (ref_pin.applied ? "true" : "false") << ",\n"
       << "    \"mode\": \"" << json_escape(ref_pin.mode) << "\",\n"
       << "    \"lon\": " << json_num(ref_pin.lon, 6) << ",\n"
       << "    \"lat\": " << json_num(ref_pin.lat, 6) << ",\n"
       << "    \"radius_m\": " << json_num(ref_pin.radius_m, 2) << ",\n"
       << "    \"ref_pixels_used\": " << ref_pin.ref_pixels_used << ",\n"
       << "    \"offset_mm\": " << json_num(ref_pin.offset_mm, 4) << "\n"
       << "  },\n"
       << "  \"ramp_removal\": {\n"
       << "    \"order\": " << ramp.order << ",\n"
       << "    \"applied\": " << (ramp.applied ? "true" : "false") << ",\n"
       << "    \"rms_before_mm\": " << json_num(ramp.rms_before_mm, 4) << ",\n"
       << "    \"rms_after_mm\": " << json_num(ramp.rms_after_mm, 4) << ",\n"
       << "    \"masked_deform_pixels\": " << ramp.masked_deform_pixels << ",\n"
       << "    \"masked_deform_fraction\": " << json_num(ramp.masked_deform_fraction, 4) << ",\n"
       << "    \"dem_applied\": " << (ramp.dem_applied ? "true" : "false") << ",\n"
       << "    \"dem_slope_mm_per_km\": " << json_num(ramp.dem_slope_mm_per_km, 4) << ",\n"
       << "    \"dem_corr_before\": " << json_num(ramp.dem_corr_before, 4) << ",\n"
       << "    \"dem_corr_after\": " << json_num(ramp.dem_corr_after, 4) << ",\n"
       << "    \"dem_mean_m\": " << json_num(ramp.dem_mean_m, 2) << ",\n"
       << "    \"dem_std_m\": " << json_num(ramp.dem_std_m, 2) << ",\n"
       << "    \"coeffs\": [";
    for (size_t i = 0; i < ramp.coeffs.size(); ++i) {
        sf << json_num(ramp.coeffs[i], 5) << (i + 1 < ramp.coeffs.size() ? ", " : "");
    }
    sf << "]\n  },\n"
       << "  \"triplet_closure\": {\n"
       << "    \"triplets_evaluated\": " << closure.triplets_evaluated << ",\n"
       << "    \"mode\": \"" << json_escape(closure.mode) << "\",\n"
       << "    \"thresh_mm\": " << json_num(closure.thresh_mm, 3) << ",\n"
       << "    \"max_bad_frac\": " << json_num(closure.max_bad_frac, 4) << ",\n"
       << "    \"evaluated_coherent_pixels\": " << closure.evaluated_coherent_pixels << ",\n"
       << "    \"masked_pixels\": " << closure.masked_pixels << ",\n"
       << "    \"bad_fraction\": " << json_num(closure.bad_fraction, 4) << ",\n"
       << "    \"rms_mm\": " << json_num(closure.rms_mm, 4) << ",\n"
       << "    \"pair_rejected\": " << (closure.pair_rejected ? "true" : "false") << "\n"
       << "  },\n"
       << "  \"maturity_gating\": {\n"
       << "    \"min_alert_snr\": " << json_num(opt.min_alert_snr, 3) << ",\n"
       << "    \"min_alert_span_days\": " << json_num(opt.min_alert_span_days, 2) << ",\n"
       << "    \"actual_span_days\": " << json_num(calendar_span_days, 2) << ",\n"
       << "    \"stack_mature\": " << (stack_mature ? "true" : "false") << ",\n"
       << "    \"snr_gated_pixels\": " << total_snr_gated << ",\n"
       << "    \"maturity_gated_corridors\": " << maturity_gated_corridors << "\n"
       << "  },\n"
       << "  \"corridors\": {\n"
       << "    \"buffer_m\": " << json_num(opt.buffer_m, 2) << ",\n"
       << "    \"total_features\": " << metrics.size() << ",\n"
       << "    \"critical_count\": " << crit_cnt << ",\n"
       << "    \"warning_count\": " << warn_cnt << ",\n"
       << "    \"watch_count\": " << watch_cnt << ",\n"
       << "    \"corridor_alerts\": " << alerts.size() << "\n"
       << "  }\n"
       << "}\n";
}

}  // namespace

namespace insar::stack {

int run_cli(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        GDALAllRegister();
        OGRRegisterAll();

        RasterMeta meta;
        StackMetadata smeta;
        std::vector<double> state_bands;
        const bool have_existing = fs::exists(opt.state_path);
        if (have_existing) {
            read_state_raster(opt.state_path, meta, smeta, state_bands);
        }

        RampResult ramp;
        ramp.order = opt.ramp_order;
        ReferencePinResult ref_pin;
        ClosureResult closure;
        std::vector<float> ramp_disp;
        std::vector<float> closure_res_map;

        if (!opt.disp_path.empty()) {
            RasterMeta disp_meta, coh_meta;
            ramp_disp = read_single_band_f32(opt.disp_path, disp_meta);
            std::vector<float> coh = read_single_band_f32(opt.coh_path, coh_meta);
            assert_compatible_grid(disp_meta, coh_meta, "coherence");

            const size_t npix = static_cast<size_t>(disp_meta.width) * static_cast<size_t>(disp_meta.height);
            if (!have_existing) {
                meta = disp_meta;
                state_bands.assign(static_cast<size_t>(kNumStateBands) * npix, 0.0);
                std::fill(
                    state_bands.data() + static_cast<size_t>(kBandLatestDisp) * npix,
                    state_bands.data() + static_cast<size_t>(kNumStateBands) * npix,
                    kNanD);
            } else {
                assert_compatible_grid(meta, disp_meta, "displacement");
            }

            double t0 = std::isfinite(opt.t0_days)
                ? opt.t0_days
                : (have_existing ? smeta.last_t1_days : 0.0);
            double t1 = std::isfinite(opt.t1_days) ? opt.t1_days : (t0 + opt.dt_days);

            std::vector<uint8_t> ref_mask = build_reference_mask(opt, meta, ref_pin);

            // 1. Triplet phase closure screening BEFORE ramp fitting or stack state update
            std::vector<uint8_t> closure_bad;
            closure = evaluate_triplet_closure(opt, meta, ramp_disp, coh, ref_mask, closure_bad, closure_res_map);

            if (closure.pair_rejected) {
                smeta.rejected_pair_count += 1;
                if (have_existing) {
                    write_state_raster(opt.state_path, meta, smeta, state_bands);
                }
                std::cout << "insar_stack REJECTED pair (closure bad_frac="
                          << std::fixed << std::setprecision(3) << closure.bad_fraction
                          << " > " << closure.max_bad_frac
                          << ", rms=" << std::setprecision(2) << closure.rms_mm << "mm)\n";
            } else {
                // Mask individual pixels that failed triplet phase closure
                if (closure.masked_pixels > 0) {
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < npix; ++i) {
                        if (closure_bad[i]) {
                            ramp_disp[i] = kNanF;
                        }
                    }
                }

                // 2. Load optional DEM aligned to stack grid
                std::vector<float> dem;
                if (!opt.dem_path.empty()) {
                    dem = read_raster_aligned_f32(opt.dem_path, meta, GRA_Bilinear);
                }

                // 3. Extract coherent connected component around reference region & build deformation exclusion mask
                std::vector<uint8_t> ref_cc = extract_reference_connected_component(
                    meta.width, meta.height, ramp_disp, coh, ref_mask, opt.min_coh);
                std::vector<uint8_t> deform_excl = build_deform_exclusion_mask(
                    opt, meta, smeta, state_bands, ref_mask, ref_cc, ramp_disp, coh);

                // 4. Joint robust ramp + DEM tropospheric correction on masked background
                ramp = remove_robust_ramp_and_dem(
                    meta.width, meta.height, ramp_disp, coh, dem, deform_excl, ref_mask, ref_cc,
                    opt.ramp_order, opt.min_coh, opt.ramp_mask_sigma, opt.ramp_mask_dilate_px);

                // 5. Pin corrected displacement to known stable reference region
                apply_reference_pinning(ramp_disp, coh, ref_mask, opt.min_coh, ref_pin);

                // 6. Incremental O(1) sufficient-statistics state update
                const bool is_first_any = (smeta.pair_count == 0 && smeta.redundant_pair_count == 0);
                if (is_first_any) {
                    smeta.origin_days = t0;
                    smeta.first_t0_days = t0;
                    smeta.last_t1_days = t1;
                } else {
                    smeta.first_t0_days = std::min(smeta.first_t0_days, t0);
                    smeta.last_t1_days = std::max(smeta.last_t1_days, t1);
                }

                update_state_incremental(
                    state_bands, npix, ramp_disp, coh, t0, t1, smeta.origin_days, opt.min_coh,
                    opt.redundant_pair, opt.pair_weight_scale);

                if (opt.redundant_pair) {
                    smeta.redundant_pair_count += 1;
                } else {
                    smeta.sum_dt_days += (t1 - t0);
                    smeta.pair_count += 1;
                }

                write_state_raster(opt.state_path, meta, smeta, state_bands);
            }
        } else if (!have_existing) {
            throw std::runtime_error("Cannot export without either --disp/--coh or existing --state");
        }

        const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
        const double calendar_span_days = std::max(0.0, smeta.last_t1_days - smeta.first_t0_days);
        const int total_pairs_any = smeta.pair_count + smeta.redundant_pair_count;
        const bool stack_mature = (calendar_span_days >= opt.min_alert_span_days) &&
                                  (total_pairs_any >= 2 || opt.min_alert_span_days <= 0.0);

        if (!opt.no_export) {
            fs::path out_dir(opt.out_dir);
            fs::create_directories(out_dir);

            DerivedFields df = derive_products(state_bands, npix, smeta, opt.min_coh);
            if (!opt.skip_cog) {
                write_cog_f32(out_dir / "velocity_mm_yr.tif", meta, df.vel, "LOS Velocity (mm/yr)");
                write_cog_f32(out_dir / "acceleration_mm_yr2.tif", meta, df.accel, "LOS Acceleration (mm/yr^2)");
                write_cog_f32(out_dir / "cumulative_disp_mm.tif", meta, df.cum_disp, "Cumulative LOS Displacement (mm)");
                write_cog_f32(out_dir / "temporal_coherence.tif", meta, df.temp_coh, "Mean Temporal Coherence");
                write_cog_f32(out_dir / "velocity_uncertainty_mm_yr.tif", meta, df.unc, "Velocity Std Error (mm/yr)");
                write_cog_f32(out_dir / "velocity_snr.tif", meta, df.snr, "Velocity Signal-to-Noise Ratio |v|/sigma_v");
                if (opt.export_ramp_disp && !ramp_disp.empty() && !closure.pair_rejected) {
                    write_cog_f32(out_dir / "ramp_removed_disp_mm.tif", meta, ramp_disp, "Latest Pair Corrected Displacement (mm)");
                }
                if (closure.triplets_evaluated > 0 && !closure_res_map.empty()) {
                    write_cog_f32(out_dir / "closure_residual_mm.tif", meta, closure_res_map, "Triplet Phase Closure RMS Residual (mm)");
                }
            }

            std::vector<CorridorMetric> metrics;
            std::vector<AlertPoint> alerts;
            size_t total_snr_gated = 0;
            process_corridors(opt, meta, smeta, df, stack_mature, metrics, alerts, total_snr_gated);
            write_corridor_outputs(
                out_dir, opt, meta, smeta, ramp, ref_pin, closure, df,
                stack_mature, calendar_span_days, total_snr_gated, metrics, alerts);
        }

        if (!closure.pair_rejected) {
            std::cout << "insar_stack updated: pairs=" << smeta.pair_count;
            if (smeta.redundant_pair_count > 0) {
                std::cout << " (+" << smeta.redundant_pair_count << " redundant)";
            }
            std::cout << " span_days=" << std::fixed << std::setprecision(1) << calendar_span_days;
            if (ramp.applied) {
                std::cout << " ramp_rms=" << std::setprecision(2) << ramp.rms_before_mm
                          << "->" << ramp.rms_after_mm << "mm";
            }
            if (ramp.dem_applied) {
                std::cout << " dem_slope=" << std::setprecision(2) << ramp.dem_slope_mm_per_km << "mm/km";
            }
            if (ref_pin.applied) {
                std::cout << " ref_pin=" << std::setprecision(2) << ref_pin.offset_mm
                          << "mm(" << ref_pin.ref_pixels_used << "px)";
            }
            if (closure.triplets_evaluated > 0) {
                std::cout << " closure_masked=" << closure.masked_pixels
                          << "(rms=" << std::setprecision(2) << closure.rms_mm << "mm)";
            }
            std::cout << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}

int run_cli(const std::vector<std::string>& arguments) {
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.emplace_back("insar_stack");
    storage.insert(storage.end(), arguments.begin(), arguments.end());

    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& value : storage) {
        argv.push_back(value.data());
    }
    return run_cli(static_cast<int>(argv.size()), argv.data());
}

}  // namespace insar::stack
