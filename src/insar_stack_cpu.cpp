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
#include "ogr_api.h"
#include "ogr_spatialref.h"
#include "ogrsf_frmts.h"

namespace fs = std::filesystem;

namespace {

constexpr double kDaysPerYear = 365.25;
constexpr double kNanF = std::numeric_limits<float>::quiet_NaN();
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
};

struct RampResult {
    int order = 1;
    bool applied = false;
    double rms_before_mm = 0.0;
    double rms_after_mm = 0.0;
    std::vector<double> coeffs;
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
    double coverage_ratio = 0.0;
    double mean_coherence = kNanD;
    double mean_velocity_mm_yr = kNanD;
    double min_velocity_mm_yr = kNanD;
    double max_velocity_mm_yr = kNanD;
    double p95_abs_velocity_mm_yr = kNanD;
    double mean_cumulative_disp_mm = kNanD;
    double max_abs_cumulative_disp_mm = kNanD;
    double mean_acceleration_mm_yr2 = kNanD;
    double max_abs_acceleration_mm_yr2 = kNanD;
    double latest_interval_disp_mm = kNanD;
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
    double dt_days = -1.0;
    double t0_days = kNanD;
    double t1_days = kNanD;
    int ramp_order = 1;
    float min_coh = 0.35f;
    double buffer_m = 100.0;
    float vel_alert_mm_yr = 15.0f;
    float accel_alert_mm_yr2 = 25.0f;
    float min_alert_coh = 0.55f;
    double max_missing_frac = 0.6;
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
        << "Sequential O(1) InSAR stacking, ramp removal, and vector corridor analytics.\n\n"
        << "Required:\n"
        << "  --state <stack_state.tif>     Persistent Float64 sufficient-statistics raster\n"
        << "  --disp <displacement_mm.tif>  Input unwrapped pair displacement GeoTIFF (mm)\n"
        << "  --coh <coherence.tif>         Input interferometric coherence GeoTIFF [0..1]\n"
        << "  --dt-days <double>            Acquisition baseline duration (days), OR\n"
        << "  --t0-days <d0> --t1-days <d1> Explicit epoch bounds relative to stack t=0 (days)\n"
        << "  --out-dir <dir>               Output directory for derived COGs & corridor JSONs\n\n"
        << "Optional:\n"
        << "  --ramp-order <0|1|2>          Coherence-weighted Huber IRLS ramp order (-1=off, default 1)\n"
        << "  --min-coh <float>             Minimum coherence for ramp fit & stack update (default 0.35)\n"
        << "  --corridors <path>            Vector file (GeoJSON/GPKG) of infrastructure corridors\n"
        << "  --buffer-m <double>           Metric corridor buffer radius in meters (default 100.0)\n"
        << "  --vel-alert-mm-yr <float>     |Velocity| alert threshold in mm/yr (default 15.0)\n"
        << "  --accel-alert-mm-yr2 <float>  |Acceleration| alert threshold in mm/yr^2 (default 25.0)\n"
        << "  --min-alert-coh <float>       Minimum temporal coherence for alerts (default 0.55)\n"
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
        } else if (arg == "--buffer-m") {
            opt.buffer_m = std::stod(need_val(i, "--buffer-m"));
        } else if (arg == "--vel-alert-mm-yr") {
            opt.vel_alert_mm_yr = std::stof(need_val(i, "--vel-alert-mm-yr"));
        } else if (arg == "--accel-alert-mm-yr2") {
            opt.accel_alert_mm_yr2 = std::stof(need_val(i, "--accel-alert-mm-yr2"));
        } else if (arg == "--min-alert-coh") {
            opt.min_alert_coh = std::stof(need_val(i, "--min-alert-coh"));
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

double eval_ramp_basis(int order, double u, double v, const std::vector<double>& c) {
    if (order == 0) {
        return c[0];
    }
    if (order == 1) {
        return c[0] + c[1] * u + c[2] * v;
    }
    return c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * u * v + c[5] * v * v;
}

RampResult remove_robust_ramp(
    int width,
    int height,
    std::vector<float>& disp,
    const std::vector<float>& coh,
    int order,
    float min_coh
) {
    RampResult res;
    res.order = order;
    if (order < 0) {
        return res;
    }
    const int n_params = (order == 0) ? 1 : (order == 1 ? 3 : 6);
    const size_t npix = static_cast<size_t>(width) * static_cast<size_t>(height);

    std::vector<size_t> valid_idx;
    valid_idx.reserve(npix / 2);
    double sum_sq_before = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        if (std::isfinite(disp[i]) && std::isfinite(coh[i]) && coh[i] >= min_coh) {
            valid_idx.push_back(i);
            sum_sq_before += static_cast<double>(disp[i]) * static_cast<double>(disp[i]);
        }
    }
    if (valid_idx.size() < static_cast<size_t>(std::max(12, n_params * 4))) {
        return res;
    }
    res.rms_before_mm = std::sqrt(sum_sq_before / static_cast<double>(valid_idx.size()));

    const double inv_half_w = (width > 1) ? 2.0 / static_cast<double>(width - 1) : 0.0;
    const double inv_half_h = (height > 1) ? 2.0 / static_cast<double>(height - 1) : 0.0;

    std::vector<double> coeffs(n_params, 0.0);
    double mad_sigma = 1.0;
    const size_t n_valid = valid_idx.size();
    const size_t sample_step = std::max<size_t>(1, n_valid / 200000);
    std::vector<double> abs_res_sample;
    abs_res_sample.reserve((n_valid + sample_step - 1) / sample_step);

    // 2 convex Huber IRLS passes followed by 4 redescending Tukey Biweight passes
    // so localized subsidence bowls do not bias the regional background ramp.
    for (int iter = 0; iter < 6; ++iter) {
        std::vector<double> ata(static_cast<size_t>(n_params) * n_params, 0.0);
        std::vector<double> aty(n_params, 0.0);
        const double huber_delta = 1.345 * mad_sigma;
        const double tukey_c = 4.685 * mad_sigma;

        #pragma omp parallel
        {
            std::vector<double> local_ata(static_cast<size_t>(n_params) * n_params, 0.0);
            std::vector<double> local_aty(n_params, 0.0);
            double phi[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            #pragma omp for schedule(static)
            for (size_t k = 0; k < n_valid; ++k) {
                size_t idx = valid_idx[k];
                int y = static_cast<int>(idx / static_cast<size_t>(width));
                int x = static_cast<int>(idx % static_cast<size_t>(width));
                double u = static_cast<double>(x) * inv_half_w - 1.0;
                double v = static_cast<double>(y) * inv_half_h - 1.0;

                double g = static_cast<double>(coh[idx]);
                double w = g * g * g;
                double val = static_cast<double>(disp[idx]);
                if (iter > 0) {
                    double r = std::abs(val - eval_ramp_basis(order, u, v, coeffs));
                    if (iter <= 2) {
                        if (r > huber_delta) {
                            w *= (huber_delta / r);
                        }
                    } else {
                        if (r >= tukey_c) {
                            continue;
                        }
                        double t = r / tukey_c;
                        double s = 1.0 - t * t;
                        w *= s * s;
                    }
                }
                phi[0] = 1.0;
                if (order >= 1) {
                    phi[1] = u;
                    phi[2] = v;
                }
                if (order == 2) {
                    phi[3] = u * u;
                    phi[4] = u * v;
                    phi[5] = v * v;
                }
                for (int r = 0; r < n_params; ++r) {
                    double wr = w * phi[r];
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
        if (!solve_linear_system(n_params, ata, aty)) {
            if (iter == 0) {
                return res;
            }
            break;
        }
        coeffs = aty;

        if (iter < 5) {
            abs_res_sample.clear();
            for (size_t k = 0; k < n_valid; k += sample_step) {
                size_t idx = valid_idx[k];
                int y = static_cast<int>(idx / static_cast<size_t>(width));
                int x = static_cast<int>(idx % static_cast<size_t>(width));
                double u = static_cast<double>(x) * inv_half_w - 1.0;
                double v = static_cast<double>(y) * inv_half_h - 1.0;
                double pred = eval_ramp_basis(order, u, v, coeffs);
                abs_res_sample.push_back(std::abs(static_cast<double>(disp[idx]) - pred));
            }
            size_t mid = abs_res_sample.size() / 2;
            std::nth_element(abs_res_sample.begin(), abs_res_sample.begin() + mid, abs_res_sample.end());
            mad_sigma = std::max(0.05, 1.4826 * abs_res_sample[mid]);
        }
    }

    double sum_sq_after = 0.0;
    #pragma omp parallel for reduction(+:sum_sq_after) schedule(static)
    for (size_t idx = 0; idx < npix; ++idx) {
        if (!std::isfinite(disp[idx])) continue;
        int y = static_cast<int>(idx / static_cast<size_t>(width));
        int x = static_cast<int>(idx % static_cast<size_t>(width));
        double u = static_cast<double>(x) * inv_half_w - 1.0;
        double v = static_cast<double>(y) * inv_half_h - 1.0;
        double corrected = static_cast<double>(disp[idx]) - eval_ramp_basis(order, u, v, coeffs);
        disp[idx] = static_cast<float>(corrected);
        if (std::isfinite(coh[idx]) && coh[idx] >= min_coh) {
            sum_sq_after += corrected * corrected;
        }
    }
    res.applied = true;
    res.coeffs = std::move(coeffs);
    res.rms_after_mm = std::sqrt(sum_sq_after / static_cast<double>(n_valid));
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
    float min_coh
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
        latest[i] = kNanD;
        double c = static_cast<double>(coh[i]);
        double d = static_cast<double>(disp[i]);
        if (!std::isfinite(c)) {
            continue;
        }
        if (std::isfinite(d)) {
            fp_cnt[i] += 1.0;
            sum_coh[i] += std::clamp(c, 0.0, 1.0);
        }
        if (!std::isfinite(d) || c < static_cast<double>(min_coh)) {
            continue;
        }
        double g = std::clamp(c, 0.0, 1.0);
        double w = std::max(1e-4, g * g);
        sum_w[i] += w;
        s11[i] += w * x11;
        s12[i] += w * x12;
        s22[i] += w * x22;
        b1[i] += w * x1 * d;
        b2[i] += w * x2 * d;
        syy[i] += w * d * d;
        cum_raw[i] += d;
        valid_dt[i] += tau_yr;
        valid_cnt[i] += 1.0;
        latest[i] = d;
    }
}

struct DerivedFields {
    std::vector<float> vel;
    std::vector<float> accel;
    std::vector<float> cum_disp;
    std::vector<float> temp_coh;
    std::vector<float> unc;
    std::vector<float> latest_disp;
    std::vector<double> vel_d;
    std::vector<double> accel_d;
    std::vector<double> cum_d;
    std::vector<double> coh_d;
    std::vector<double> unc_d;
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
    out.latest_disp.assign(npix, kNanF);

    out.vel_d.assign(npix, kNanD);
    out.accel_d.assign(npix, kNanD);
    out.cum_d.assign(npix, kNanD);
    out.coh_d.assign(npix, kNanD);
    out.unc_d.assign(npix, kNanD);
    out.latest_d.assign(npix, kNanD);
    out.valid_cnt.assign(npix, 0);

    const double total_span_yr = smeta.sum_dt_days / kDaysPerYear;
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
        // Require reasonable epoch coverage when multi-pass (>=2) so single erratic glint doesn't dominate
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

double meters_to_crs_units(const OGRSpatialReference& srs, double buffer_m, double center_lat_deg) {
    if (srs.IsProjected()) {
        double lin = srs.GetLinearUnits(nullptr);
        return (lin > 0.0) ? (buffer_m / lin) : buffer_m;
    }
    double cos_lat = std::max(0.2, std::cos(center_lat_deg * 3.14159265358979323846 / 180.0));
    return buffer_m / (111320.0 * cos_lat);
}

std::string classify_corridor_risk(
    int valid_pixels,
    int alert_pixel_count,
    double mean_vel,
    double p95_abs_vel,
    float vel_alert_mm_yr
) {
    if (valid_pixels <= 0 || !std::isfinite(p95_abs_vel)) {
        return "no_coverage";
    }
    if (alert_pixel_count > 0 &&
        (p95_abs_vel >= 2.0 * vel_alert_mm_yr || alert_pixel_count >= 10 || std::abs(mean_vel) >= vel_alert_mm_yr)) {
        return "critical";
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
    std::vector<CorridorMetric>& metrics_out,
    std::vector<AlertPoint>& alerts_out
) {
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
    const int min_alert_passes = (smeta.pair_count >= 2) ? 2 : 1;

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
            double sum_vel = 0.0, sum_coh = 0.0, sum_cum = 0.0, sum_acc = 0.0, sum_lat = 0.0;
            double min_v = std::numeric_limits<double>::infinity();
            double max_v = -std::numeric_limits<double>::infinity();
            double max_abs_cum = 0.0, max_abs_acc = 0.0;
            int acc_cnt = 0, lat_cnt = 0;

            for (int sy = 0; sy < ph; ++sy) {
                const int gy = y0 + sy;
                for (int sx = 0; sx < pw; ++sx) {
                    if (mask[static_cast<size_t>(sy) * pw + sx] == 0) continue;
                    const size_t gi = static_cast<size_t>(gy) * meta.width + (x0 + sx);
                    cm.pixel_count++;
                    if (!std::isfinite(df.vel_d[gi])) continue;

                    cm.valid_pixels++;
                    double v = df.vel_d[gi];
                    double av = std::abs(v);
                    double c = df.coh_d[gi];
                    double cd = df.cum_d[gi];
                    abs_vels.push_back(av);
                    sum_vel += v;
                    sum_coh += c;
                    sum_cum += cd;
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    max_abs_cum = std::max(max_abs_cum, std::abs(cd));

                    if (std::isfinite(df.accel_d[gi])) {
                        sum_acc += df.accel_d[gi];
                        max_abs_acc = std::max(max_abs_acc, std::abs(df.accel_d[gi]));
                        acc_cnt++;
                    }
                    if (std::isfinite(df.latest_d[gi])) {
                        sum_lat += df.latest_d[gi];
                        lat_cnt++;
                    }
                    bool high_coh = (c >= static_cast<double>(opt.min_alert_coh));
                    if (high_coh) {
                        cm.high_coh_pixels++;
                    }
                    bool is_alert = high_coh && df.valid_cnt[gi] >= min_alert_passes &&
                        (av >= static_cast<double>(opt.vel_alert_mm_yr) ||
                         (std::isfinite(df.accel_d[gi]) && std::abs(df.accel_d[gi]) >= static_cast<double>(opt.accel_alert_mm_yr2)));
                    if (is_alert) {
                        cm.alert_pixel_count++;
                        double score = av;
                        if (owner_corridor[gi] < 0 || score > owner_best_vel[gi]) {
                            owner_corridor[gi] = corridor_idx;
                            owner_best_vel[gi] = score;
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
                cm.p95_abs_velocity_mm_yr, opt.vel_alert_mm_yr);
            metrics_out.push_back(std::move(cm));
        }
    }

    // Collect candidate alert points strictly inside corridors, sorted & spatially NMS-clustered
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
    const DerivedFields& df,
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
                << "        \"pixel_count\": " << m.pixel_count << ",\n"
                << "        \"valid_pixels\": " << m.valid_pixels << ",\n"
                << "        \"high_coh_pixels\": " << m.high_coh_pixels << ",\n"
                << "        \"alert_pixel_count\": " << m.alert_pixel_count << ",\n"
                << "        \"coverage_ratio\": " << json_num(m.coverage_ratio, 4) << ",\n"
                << "        \"mean_coherence\": " << json_num(m.mean_coherence, 4) << ",\n"
                << "        \"mean_velocity_mm_yr\": " << json_num(m.mean_velocity_mm_yr, 3) << ",\n"
                << "        \"min_velocity_mm_yr\": " << json_num(m.min_velocity_mm_yr, 3) << ",\n"
                << "        \"max_velocity_mm_yr\": " << json_num(m.max_velocity_mm_yr, 3) << ",\n"
                << "        \"p95_abs_velocity_mm_yr\": " << json_num(m.p95_abs_velocity_mm_yr, 3) << ",\n"
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
               << "        \"velocity_mm_yr\": " << json_num(a.velocity_mm_yr, 3) << ",\n"
               << "        \"acceleration_mm_yr2\": " << json_num(a.acceleration_mm_yr2, 3) << ",\n"
               << "        \"cumulative_disp_mm\": " << json_num(a.cumulative_disp_mm, 3) << ",\n"
               << "        \"temporal_coherence\": " << json_num(a.temporal_coherence, 4) << ",\n"
               << "        \"uncertainty_mm_yr\": " << json_num(a.uncertainty_mm_yr, 3) << ",\n"
               << "        \"pixel_x\": " << a.px << ",\n"
               << "        \"pixel_y\": " << a.py << "\n"
               << "      }\n"
               << "    }" << (i + 1 < alerts.size() ? "," : "") << "\n";
        }
        af << "  ]\n}\n";
    }

    int crit_cnt = 0, warn_cnt = 0, watch_cnt = 0;
    for (const auto& m : metrics) {
        if (m.risk_level == "critical") crit_cnt++;
        else if (m.risk_level == "warning") warn_cnt++;
        else if (m.risk_level == "watch") watch_cnt++;
    }

    std::ofstream sf(out_dir / "stack_summary.json");
    sf << "{\n"
       << "  \"width\": " << meta.width << ",\n"
       << "  \"height\": " << meta.height << ",\n"
       << "  \"pair_count\": " << smeta.pair_count << ",\n"
       << "  \"first_t0_days\": " << json_num(smeta.first_t0_days, 3) << ",\n"
       << "  \"last_t1_days\": " << json_num(smeta.last_t1_days, 3) << ",\n"
       << "  \"total_span_days\": " << json_num(smeta.sum_dt_days, 3) << ",\n"
       << "  \"valid_stack_pixels\": " << df.valid_stack_pixels << ",\n"
       << "  \"mean_temporal_coherence\": " << json_num(df.mean_coh, 4) << ",\n"
       << "  \"mean_velocity_mm_yr\": " << json_num(df.mean_vel, 3) << ",\n"
       << "  \"p95_abs_velocity_mm_yr\": " << json_num(df.p95_abs_vel, 3) << ",\n"
       << "  \"ramp_removal\": {\n"
       << "    \"order\": " << ramp.order << ",\n"
       << "    \"applied\": " << (ramp.applied ? "true" : "false") << ",\n"
       << "    \"rms_before_mm\": " << json_num(ramp.rms_before_mm, 4) << ",\n"
       << "    \"rms_after_mm\": " << json_num(ramp.rms_after_mm, 4) << ",\n"
       << "    \"coeffs\": [";
    for (size_t i = 0; i < ramp.coeffs.size(); ++i) {
        sf << json_num(ramp.coeffs[i], 5) << (i + 1 < ramp.coeffs.size() ? ", " : "");
    }
    sf << "]\n  },\n"
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

int main(int argc, char** argv) {
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
        std::vector<float> ramp_disp;

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
            if (smeta.pair_count == 0) {
                smeta.origin_days = t0;
                smeta.first_t0_days = t0;
            }

            ramp = remove_robust_ramp(
                meta.width, meta.height, ramp_disp, coh, opt.ramp_order, opt.min_coh);
            update_state_incremental(
                state_bands, npix, ramp_disp, coh, t0, t1, smeta.origin_days, opt.min_coh);

            smeta.last_t1_days = t1;
            smeta.sum_dt_days += (t1 - t0);
            smeta.pair_count += 1;

            write_state_raster(opt.state_path, meta, smeta, state_bands);
        } else if (!have_existing) {
            throw std::runtime_error("Cannot export without either --disp/--coh or existing --state");
        }

        const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
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
                if (opt.export_ramp_disp && !ramp_disp.empty()) {
                    write_cog_f32(out_dir / "ramp_removed_disp_mm.tif", meta, ramp_disp, "Latest Pair Ramp-Removed Displacement (mm)");
                }
            }

            std::vector<CorridorMetric> metrics;
            std::vector<AlertPoint> alerts;
            process_corridors(opt, meta, smeta, df, metrics, alerts);
            write_corridor_outputs(out_dir, opt, meta, smeta, ramp, df, metrics, alerts);
        }

        std::cout << "insar_stack updated: pairs=" << smeta.pair_count
                  << " span_days=" << std::fixed << std::setprecision(1) << smeta.sum_dt_days;
        if (ramp.applied) {
            std::cout << " ramp_rms=" << std::setprecision(2) << ramp.rms_before_mm
                      << "->" << ramp.rms_after_mm << "mm";
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
