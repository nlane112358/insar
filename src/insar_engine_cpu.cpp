#include <cpl_conv.h>
#include <cpl_error.h>
#include <cpl_string.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using cfloat = std::complex<float>;
constexpr float PI_F     = 3.14159265358979323846f;
constexpr float TWO_PI_F = 6.28318530717958647692f;

inline float wrap_phase(float d) {
    d = std::fmod(d + PI_F, TWO_PI_F);
    if (d < 0.0f) d += TWO_PI_F;
    return d - PI_F;
}

inline int next_power_of_two(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

inline bool is_valid_complex(const cfloat& z) {
    float re = z.real();
    float im = z.imag();
    return std::isfinite(re) && std::isfinite(im) && (re * re + im * im > 0.0f);
}

// ============================================================================
// GDAL Raster I/O Helpers (OPERA CSLC-S1 HDF5 + GeoTIFF)
// ============================================================================

struct GeoMetadata {
    int width = 0;
    int height = 0;
    bool has_georef = false;
    double gt[6] = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
    std::string projection_wkt;
};

struct GdalDatasetDeleter {
    void operator()(GDALDataset* ds) const {
        if (ds) GDALClose(ds);
    }
};
using UniqueGdalDataset = std::unique_ptr<GDALDataset, GdalDatasetDeleter>;

bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    size_t offset = s.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[offset + i])) !=
            std::tolower(static_cast<unsigned char>(suffix[i]))) {
            return false;
        }
    }
    return true;
}

UniqueGdalDataset open_slc_dataset(const std::string& path, const std::string& subdataset) {
    bool is_already_prefixed =
        path.rfind("NETCDF:", 0) == 0 || path.rfind("HDF5:", 0) == 0;
    bool is_hdf5 = !is_already_prefixed &&
                   (ends_with_ci(path, ".h5") || ends_with_ci(path, ".hdf5"));

    std::vector<std::string> candidates;
    if (is_hdf5) {
        // NETCDF driver parses CF-1.8 UTM projection & coordinates in OPERA CSLC-S1
        std::string sub = (!subdataset.empty() && subdataset[0] == '/')
                              ? subdataset
                              : ("/" + subdataset);
        candidates.push_back("NETCDF:\"" + path + "\":" + sub);
        candidates.push_back("HDF5:\"" + path + "\":/" + sub);
    }
    candidates.push_back(path);

    for (const auto& uri : candidates) {
        CPLPushErrorHandler(CPLQuietErrorHandler);
        GDALDataset* raw = static_cast<GDALDataset*>(
            GDALOpenEx(uri.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY,
                       nullptr, nullptr, nullptr));
        CPLPopErrorHandler();

        if (raw && raw->GetRasterCount() >= 1) {
            std::printf("Opened GDAL dataset: %s (%dx%d)\n",
                        uri.c_str(), raw->GetRasterXSize(), raw->GetRasterYSize());
            return UniqueGdalDataset(raw);
        }
        if (raw) GDALClose(raw);
    }

    std::fprintf(stderr, "Error: GDAL could not open raster '%s' (subdataset '%s').\n",
                 path.c_str(), subdataset.c_str());
    return nullptr;
}

bool read_slc_gdal(
    const std::string& path,
    const std::string& subdataset,
    int xoff, int yoff, int req_w, int req_h,
    std::vector<cfloat>& out_slc,
    GeoMetadata& meta)
{
    auto ds = open_slc_dataset(path, subdataset);
    if (!ds) return false;

    int full_w = ds->GetRasterXSize();
    int full_h = ds->GetRasterYSize();

    if (xoff < 0 || yoff < 0 || xoff >= full_w || yoff >= full_h) {
        std::fprintf(stderr, "Error: Window offset (%d, %d) outside raster (%dx%d).\n",
                     xoff, yoff, full_w, full_h);
        return false;
    }

    int win_w = (req_w > 0) ? std::min(req_w, full_w - xoff) : (full_w - xoff);
    int win_h = (req_h > 0) ? std::min(req_h, full_h - yoff) : (full_h - yoff);

    meta.width = win_w;
    meta.height = win_h;
    if (ds->GetGeoTransform(meta.gt) == CE_None) {
        meta.has_georef = true;
        // Shift origin if reading a cropped AOI window
        double orig_x = meta.gt[0] + xoff * meta.gt[1] + yoff * meta.gt[2];
        double orig_y = meta.gt[3] + xoff * meta.gt[4] + yoff * meta.gt[5];
        meta.gt[0] = orig_x;
        meta.gt[3] = orig_y;
    }
    const char* wkt = ds->GetProjectionRef();
    if (wkt && wkt[0] != '\0') {
        meta.projection_wkt = wkt;
    }

    GDALRasterBand* band = ds->GetRasterBand(1);
    out_slc.resize(static_cast<size_t>(win_w) * win_h);

    CPLErr err = band->RasterIO(
        GF_Read, xoff, yoff, win_w, win_h,
        out_slc.data(), win_w, win_h,
        GDT_CFloat32, 0, 0);

    if (err != CE_None) {
        std::fprintf(stderr, "Error: GDAL RasterIO failed while reading '%s'.\n", path.c_str());
        return false;
    }
    return true;
}

// Writes a true Cloud-Optimized GeoTIFF (COG) with internal overviews via MEM -> CreateCopy
bool write_cog_f32(
    const std::string& path,
    const std::vector<float>& data,
    const std::vector<uint8_t>& valid_mask,
    const GeoMetadata& meta)
{
    GDALDriver* mem_drv = GetGDALDriverManager()->GetDriverByName("MEM");
    if (!mem_drv) {
        std::fprintf(stderr, "Error: GDAL MEM driver unavailable.\n");
        return false;
    }

    UniqueGdalDataset mem_ds(mem_drv->Create(
        "", meta.width, meta.height, 1, GDT_Float32, nullptr));
    if (!mem_ds) return false;

    if (meta.has_georef) {
        mem_ds->SetGeoTransform(const_cast<double*>(meta.gt));
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
    char** options = nullptr;
    if (out_drv) {
        options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
        options = CSLSetNameValue(options, "PREDICTOR", "YES");
        options = CSLSetNameValue(options, "BLOCKSIZE", "512");
        options = CSLSetNameValue(options, "NUM_THREADS", "ALL_CPUS");
        options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    } else {
        out_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!out_drv) {
            std::fprintf(stderr, "Error: Neither GDAL COG nor GTiff driver available.\n");
            return false;
        }
        options = CSLSetNameValue(options, "TILED", "YES");
        options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
        options = CSLSetNameValue(options, "PREDICTOR", "3");
        options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    }

    UniqueGdalDataset out_ds(out_drv->CreateCopy(
        path.c_str(), mem_ds.get(), FALSE, options, nullptr, nullptr));
    CSLDestroy(options);
    if (!out_ds) {
        std::fprintf(stderr, "Error: Could not create COG '%s'.\n", path.c_str());
        return false;
    }
    return true;
}

struct Wgs84BBox {
    bool valid = false;
    double min_lon = 0.0, min_lat = 0.0, max_lon = 0.0, max_lat = 0.0;
};

struct AnomalyPoint {
    double lon = 0.0, lat = 0.0;
    float disp_mm = 0.0f;
    float coherence = 0.0f;
};

void export_metadata_and_alerts(
    const std::string& out_dir,
    const GeoMetadata& meta,
    const std::vector<float>& disp_mm,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& valid_mask,
    float alert_threshold_mm,
    float min_alert_coh,
    double ifg_ms,
    double unwrap_ms,
    int iterations,
    size_t valid_pixels)
{
    // 1. Build CRS -> WGS84 (lon, lat) transformer
    OGRCoordinateTransformation* ct = nullptr;
    if (meta.has_georef && !meta.projection_wkt.empty()) {
        OGRSpatialReference src_srs, dst_srs;
        if (src_srs.SetFromUserInput(meta.projection_wkt.c_str()) == OGRERR_NONE) {
            src_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            dst_srs.SetWellKnownGeogCS("WGS84");
            dst_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            ct = OGRCreateCoordinateTransformation(&src_srs, &dst_srs);
        }
    }

    auto pixel_to_lonlat = [&](double px, double py, double& lon, double& lat) -> bool {
        if (!ct) return false;
        double x = meta.gt[0] + px * meta.gt[1] + py * meta.gt[2];
        double y = meta.gt[3] + px * meta.gt[4] + py * meta.gt[5];
        if (!ct->Transform(1, &x, &y)) return false;
        lon = x;
        lat = y;
        return std::isfinite(lon) && std::isfinite(lat);
    };

    Wgs84BBox bbox;
    if (ct) {
        double corners[4][2] = {
            {0.0, 0.0},
            {static_cast<double>(meta.width), 0.0},
            {static_cast<double>(meta.width), static_cast<double>(meta.height)},
            {0.0, static_cast<double>(meta.height)}
        };
        bbox.min_lon = 180.0; bbox.max_lon = -180.0;
        bbox.min_lat = 90.0;  bbox.max_lat = -90.0;
        bbox.valid = true;
        for (auto& c : corners) {
            double lon = 0.0, lat = 0.0;
            if (pixel_to_lonlat(c[0], c[1], lon, lat)) {
                bbox.min_lon = std::min(bbox.min_lon, lon);
                bbox.max_lon = std::max(bbox.max_lon, lon);
                bbox.min_lat = std::min(bbox.min_lat, lat);
                bbox.max_lat = std::max(bbox.max_lat, lat);
            } else {
                bbox.valid = false;
            }
        }
    }

    // 2. Scan for coarse-clustered high-coherence anomalies + min/max displacement
    float min_d = 0.0f, max_d = 0.0f;
    bool has_stats = false;
    for (size_t i = 0; i < disp_mm.size(); ++i) {
        if (!valid_mask[i] || coh[i] < min_alert_coh || !std::isfinite(disp_mm[i])) continue;
        if (!has_stats) {
            min_d = max_d = disp_mm[i];
            has_stats = true;
        } else {
            min_d = std::min(min_d, disp_mm[i]);
            max_d = std::max(max_d, disp_mm[i]);
        }
    }

    constexpr int CLUSTER = 32;
    std::vector<AnomalyPoint> anomalies;
    for (int by = 0; by < meta.height; by += CLUSTER) {
        int y_end = std::min(by + CLUSTER, meta.height);
        for (int bx = 0; bx < meta.width; bx += CLUSTER) {
            int x_end = std::min(bx + CLUSTER, meta.width);
            int best_x = -1, best_y = -1;
            float worst_abs = alert_threshold_mm;
            for (int y = by; y < y_end; ++y) {
                for (int x = bx; x < x_end; ++x) {
                    size_t idx = static_cast<size_t>(y) * meta.width + x;
                    if (!valid_mask[idx] || coh[idx] < min_alert_coh || !std::isfinite(disp_mm[idx])) continue;
                    float a = std::abs(disp_mm[idx]);
                    if (a >= worst_abs) {
                        worst_abs = a;
                        best_x = x;
                        best_y = y;
                    }
                }
            }
            if (best_x >= 0) {
                size_t idx = static_cast<size_t>(best_y) * meta.width + best_x;
                double lon = 0.0, lat = 0.0;
                if (pixel_to_lonlat(best_x + 0.5, best_y + 0.5, lon, lat)) {
                    anomalies.push_back({lon, lat, disp_mm[idx], coh[idx]});
                }
            }
        }
    }
    if (ct) OGRCoordinateTransformation::DestroyCT(ct);

    // 3. Write alerts.geojson
    {
        std::ofstream gj(out_dir + "/alerts.geojson");
        gj << "{\"type\":\"FeatureCollection\",\"features\":[";
        for (size_t i = 0; i < anomalies.size(); ++i) {
            const auto& a = anomalies[i];
            if (i > 0) gj << ",";
            gj << "{\"type\":\"Feature\",\"geometry\":{\"type\":\"Point\",\"coordinates\":["
               << std::setprecision(9) << a.lon << "," << a.lat
               << "]},\"properties\":{\"displacement_mm\":"
               << std::setprecision(6) << a.disp_mm << ",\"coherence\":" << a.coherence << "}}";
        }
        gj << "]}\n";
    }

    // 4. Write summary.json
    {
        std::ofstream sj(out_dir + "/summary.json");
        sj << "{\n"
           << "  \"width\": " << meta.width << ",\n"
           << "  \"height\": " << meta.height << ",\n"
           << "  \"valid_pixels\": " << valid_pixels << ",\n"
           << "  \"ifg_ms\": " << ifg_ms << ",\n"
           << "  \"unwrap_ms\": " << unwrap_ms << ",\n"
           << "  \"processing_time_ms\": " << (ifg_ms + unwrap_ms) << ",\n"
           << "  \"iterations\": " << iterations << ",\n"
           << "  \"min_displacement_mm\": " << min_d << ",\n"
           << "  \"max_displacement_mm\": " << max_d << ",\n"
           << "  \"anomalies_detected\": " << anomalies.size() << ",\n";
        if (bbox.valid) {
            sj << std::setprecision(9)
               << "  \"bbox_wgs84\": [" << bbox.min_lon << ", " << bbox.min_lat << ", "
               << bbox.max_lon << ", " << bbox.max_lat << "]\n";
        } else {
            sj << "  \"bbox_wgs84\": null\n";
        }
        sj << "}\n";
    }
}

// ============================================================================
// Precomputed 1D Radix-2 FFT Plan (Zero Allocations During Solve)
// ============================================================================

struct Fft1DPlan {
    int n = 0;
    std::vector<int> bit_rev;
    std::vector<cfloat> twiddles;

    void init(int size) {
        n = size;
        int log_n = 0;
        while ((1 << log_n) < n) ++log_n;

        bit_rev.resize(n);
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < log_n; ++b) {
                r = (r << 1) | ((i >> b) & 1);
            }
            bit_rev[i] = r;
        }

        twiddles.resize(n / 2);
        for (int k = 0; k < n / 2; ++k) {
            double angle = -2.0 * static_cast<double>(PI_F) * k / n;
            twiddles[k] = cfloat(static_cast<float>(std::cos(angle)),
                                 static_cast<float>(std::sin(angle)));
        }
    }

    void exec(cfloat* data, bool inverse) const {
        for (int i = 0; i < n; ++i) {
            int j = bit_rev[i];
            if (i < j) std::swap(data[i], data[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            int half = len >> 1;
            int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; ++j) {
                    cfloat w = twiddles[j * step];
                    if (inverse) w = std::conj(w);
                    cfloat u = data[i + j];
                    cfloat v = data[i + j + half] * w;
                    data[i + j]        = u + v;
                    data[i + j + half] = u - v;
                }
            }
        }
    }
};

void transpose_blocked(const cfloat* src, cfloat* dst, int rows, int cols) {
    constexpr int TILE = 32;
    #pragma omp parallel for schedule(static)
    for (int r0 = 0; r0 < rows; r0 += TILE) {
        int r_end = std::min(r0 + TILE, rows);
        for (int c0 = 0; c0 < cols; c0 += TILE) {
            int c_end = std::min(c0 + TILE, cols);
            for (int r = r0; r < r_end; ++r) {
                for (int c = c0; c < c_end; ++c) {
                    dst[static_cast<size_t>(c) * rows + r] =
                        src[static_cast<size_t>(r) * cols + c];
                }
            }
        }
    }
}

} // namespace

// ============================================================================
// CPU InSAR Engine (Arbitrary Dimensions + NaN-Aware Multilook & Unwrap)
// ============================================================================

class CpuInsarEngine {
public:
    CpuInsarEngine(int out_width, int out_height)
        : W_(out_width), H_(out_height),
          PW_(next_power_of_two(out_width)),
          PH_(next_power_of_two(out_height)),
          PN_(static_cast<size_t>(PW_) * PH_),
          MW_(2 * PW_), MH_(2 * PH_)
    {
        wx_.resize(PN_);
        wy_.resize(PN_);
        phi_pad_.resize(PN_);
        r_.resize(PN_);
        z_.resize(PN_);
        p_.resize(PN_);
        Ap_.resize(PN_);

        buf_a_.resize(static_cast<size_t>(MH_) * MW_);
        buf_b_.resize(static_cast<size_t>(MW_) * MH_);

        fft_w_.init(MW_);
        fft_h_.init(MH_);

        cos_kx_.resize(MW_);
        for (int kx = 0; kx < MW_; ++kx) {
            cos_kx_[kx] = std::cos(PI_F * static_cast<float>(kx) / static_cast<float>(PW_));
        }
        cos_ky_.resize(MH_);
        for (int ky = 0; ky < MH_; ++ky) {
            cos_ky_[ky] = std::cos(PI_F * static_cast<float>(ky) / static_cast<float>(PH_));
        }
    }

    int padded_width() const { return PW_; }
    int padded_height() const { return PH_; }

    struct RunStats {
        double ifg_ms = 0.0;
        double unwrap_ms = 0.0;
        int iterations = 0;
        double rel_residual = 0.0;
        size_t valid_pixels = 0;
    };

    RunStats process_pair(
        const std::vector<cfloat>& slc1,
        const std::vector<cfloat>& slc2,
        int in_w, int in_h,
        int stride_x, int stride_y,
        int look_rx, int look_ry,
        std::vector<float>& wrapped,
        std::vector<float>& coherence,
        std::vector<float>& unwrapped,
        std::vector<uint8_t>& valid_mask,
        int max_iters = 40,
        float tol = 1e-4f)
    {
        using Clock = std::chrono::steady_clock;
        RunStats stats;

        const size_t out_n = static_cast<size_t>(W_) * H_;
        wrapped.assign(out_n, 0.0f);
        coherence.assign(out_n, 0.0f);
        unwrapped.assign(out_n, 0.0f);
        valid_mask.assign(out_n, 0);

        // --- Stage 1: NaN-Safe Coherent Multilooking + Coherence ---
        auto t0 = Clock::now();
        stats.valid_pixels = compute_interferogram_and_coherence(
            slc1, slc2, in_w, in_h, stride_x, stride_y, look_rx, look_ry,
            wrapped, coherence, valid_mask);
        auto t1 = Clock::now();

        // --- Stage 2: Masked Coherence-Weighted PCG on Padded Power-of-2 Grid ---
        std::fill(phi_pad_.begin(), phi_pad_.end(), 0.0f);
        build_weights_and_rhs(wrapped, coherence, valid_mask, 1e-5f);

        double b_norm = std::sqrt(dot_product(r_, r_));
        if (b_norm > 1e-12) {
            apply_poisson_preconditioner(r_, z_);
            std::copy(z_.begin(), z_.end(), p_.begin());
            double rz_old = dot_product(r_, z_);

            for (int iter = 1; iter <= max_iters; ++iter) {
                apply_weighted_laplacian(p_, Ap_);
                double pAp = dot_product(p_, Ap_);
                if (pAp <= 1e-18) break;

                float alpha = static_cast<float>(rz_old / pAp);
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(PN_); ++i) {
                    phi_pad_[i] += alpha * p_[i];
                    r_[i]       -= alpha * Ap_[i];
                }

                double r_norm = std::sqrt(dot_product(r_, r_));
                stats.iterations = iter;
                stats.rel_residual = r_norm / b_norm;
                if (stats.rel_residual < tol) break;

                apply_poisson_preconditioner(r_, z_);
                double rz_new = dot_product(r_, z_);
                float beta = static_cast<float>(rz_new / (rz_old + 1e-30));

                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(PN_); ++i) {
                    p_[i] = z_[i] + beta * p_[i];
                }
                rz_old = rz_new;
            }
        }

        // Crop padded solver solution back to true W_ x H_ burst grid
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H_; ++y) {
            const float* src_row = &phi_pad_[static_cast<size_t>(y) * PW_];
            float* dst_row       = &unwrapped[static_cast<size_t>(y) * W_];
            std::copy(src_row, src_row + W_, dst_row);
        }

        auto t2 = Clock::now();
        stats.ifg_ms    = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.unwrap_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        return stats;
    }

private:
    size_t compute_interferogram_and_coherence(
        const std::vector<cfloat>& slc1,
        const std::vector<cfloat>& slc2,
        int in_w, int in_h,
        int stride_x, int stride_y,
        int look_rx, int look_ry,
        std::vector<float>& wrapped,
        std::vector<float>& coh,
        std::vector<uint8_t>& valid_mask) const
    {
        const int rx = std::max(look_rx, stride_x / 2);
        const int ry = std::max(look_ry, stride_y / 2);
        const int min_samples = std::max(1, ((2 * rx + 1) * (2 * ry + 1)) / 4);

        size_t total_valid = 0;

        #pragma omp parallel for reduction(+:total_valid) schedule(static)
        for (int oy = 0; oy < H_; ++oy) {
            int cy = oy * stride_y + stride_y / 2;
            int y0 = std::max(0, cy - ry);
            int y1 = std::min(in_h - 1, cy + ry);

            for (int ox = 0; ox < W_; ++ox) {
                int cx = ox * stride_x + stride_x / 2;
                int x0 = std::max(0, cx - rx);
                int x1 = std::min(in_w - 1, cx + rx);

                float sum_re = 0.0f, sum_im = 0.0f;
                float pow1 = 0.0f, pow2 = 0.0f;
                int count = 0;

                for (int j = y0; j <= y1; ++j) {
                    size_t row = static_cast<size_t>(j) * in_w;
                    for (int i = x0; i <= x1; ++i) {
                        const cfloat a = slc1[row + i];
                        const cfloat b = slc2[row + i];
                        if (!is_valid_complex(a) || !is_valid_complex(b)) continue;

                        sum_re += a.real() * b.real() + a.imag() * b.imag();
                        sum_im += a.imag() * b.real() - a.real() * b.imag();
                        pow1   += a.real() * a.real() + a.imag() * a.imag();
                        pow2   += b.real() * b.real() + b.imag() * b.imag();
                        ++count;
                    }
                }

                size_t idx = static_cast<size_t>(oy) * W_ + ox;
                if (count >= min_samples && pow1 > 0.0f && pow2 > 0.0f) {
                    wrapped[idx]    = std::atan2(sum_im, sum_re);
                    float denom     = std::sqrt(pow1 * pow2) + 1e-12f;
                    coh[idx]        = std::clamp(std::hypot(sum_re, sum_im) / denom, 0.0f, 1.0f);
                    valid_mask[idx] = 1;
                    ++total_valid;
                } else {
                    wrapped[idx]    = 0.0f;
                    coh[idx]        = 0.0f;
                    valid_mask[idx] = 0;
                }
            }
        }
        return total_valid;
    }

    void build_weights_and_rhs(
        const std::vector<float>& psi,
        const std::vector<float>& coh,
        const std::vector<uint8_t>& valid_mask,
        float min_weight)
    {
        auto is_valid = [&](int x, int y) -> bool {
            return (x >= 0 && x < W_ && y >= 0 && y < H_) &&
                   (valid_mask[static_cast<size_t>(y) * W_ + x] != 0);
        };
        auto get_psi = [&](int x, int y) -> float {
            return psi[static_cast<size_t>(y) * W_ + x];
        };
        auto get_coh = [&](int x, int y) -> float {
            return coh[static_cast<size_t>(y) * W_ + x];
        };

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < PH_; ++y) {
            for (int x = 0; x < PW_; ++x) {
                size_t pidx = static_cast<size_t>(y) * PW_ + x;
                bool v_here = is_valid(x, y);

                // Edge to right neighbor (x -> x+1)
                float w_right = 0.0f, dx_right = 0.0f;
                if (x + 1 < PW_) {
                    if (v_here && is_valid(x + 1, y)) {
                        float c = std::min(get_coh(x, y), get_coh(x + 1, y));
                        w_right  = std::max(min_weight, c * c * c);
                        dx_right = wrap_phase(get_psi(x + 1, y) - get_psi(x, y));
                    } else {
                        w_right  = min_weight;
                        dx_right = 0.0f;
                    }
                }
                wx_[pidx] = w_right;

                // Edge to bottom neighbor (y -> y+1)
                float w_down = 0.0f, dy_down = 0.0f;
                if (y + 1 < PH_) {
                    if (v_here && is_valid(x, y + 1)) {
                        float c = std::min(get_coh(x, y), get_coh(x, y + 1));
                        w_down  = std::max(min_weight, c * c * c);
                        dy_down = wrap_phase(get_psi(x, y + 1) - get_psi(x, y));
                    } else {
                        w_down  = min_weight;
                        dy_down = 0.0f;
                    }
                }
                wy_[pidx] = w_down;

                // Edge from left neighbor (x-1 -> x)
                float w_left = 0.0f, dx_left = 0.0f;
                if (x > 0) {
                    if (is_valid(x - 1, y) && v_here) {
                        float c = std::min(get_coh(x - 1, y), get_coh(x, y));
                        w_left  = std::max(min_weight, c * c * c);
                        dx_left = wrap_phase(get_psi(x, y) - get_psi(x - 1, y));
                    } else {
                        w_left  = min_weight;
                        dx_left = 0.0f;
                    }
                }

                // Edge from top neighbor (y-1 -> y)
                float w_up = 0.0f, dy_up = 0.0f;
                if (y > 0) {
                    if (is_valid(x, y - 1) && v_here) {
                        float c = std::min(get_coh(x, y - 1), get_coh(x, y));
                        w_up  = std::max(min_weight, c * c * c);
                        dy_up = wrap_phase(get_psi(x, y) - get_psi(x, y - 1));
                    } else {
                        w_up  = min_weight;
                        dy_up = 0.0f;
                    }
                }

                r_[pidx] = (-w_right * dx_right + w_left * dx_left) +
                           (-w_down  * dy_down  + w_up   * dy_up);
            }
        }
    }

    void apply_weighted_laplacian(const std::vector<float>& p, std::vector<float>& Ap) const {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < PH_; ++y) {
            for (int x = 0; x < PW_; ++x) {
                size_t idx = static_cast<size_t>(y) * PW_ + x;
                float center = p[idx];
                float acc = 0.0f;

                if (x + 1 < PW_) acc += wx_[idx]       * (center - p[idx + 1]);
                if (x > 0)       acc += wx_[idx - 1]   * (center - p[idx - 1]);
                if (y + 1 < PH_) acc += wy_[idx]       * (center - p[idx + PW_]);
                if (y > 0)       acc += wy_[idx - PW_] * (center - p[idx - PW_]);

                Ap[idx] = acc;
            }
        }
    }

    void apply_poisson_preconditioner(const std::vector<float>& in, std::vector<float>& out) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < PH_; ++y) {
            size_t row_top = static_cast<size_t>(y) * MW_;
            size_t row_bot = static_cast<size_t>(MH_ - 1 - y) * MW_;
            for (int x = 0; x < PW_; ++x) {
                cfloat v(in[static_cast<size_t>(y) * PW_ + x], 0.0f);
                buf_a_[row_top + x]             = v;
                buf_a_[row_top + (MW_ - 1 - x)] = v;
                buf_a_[row_bot + x]             = v;
                buf_a_[row_bot + (MW_ - 1 - x)] = v;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < MH_; ++y) {
            fft_w_.exec(&buf_a_[static_cast<size_t>(y) * MW_], false);
        }

        transpose_blocked(buf_a_.data(), buf_b_.data(), MH_, MW_);

        const float inv_area = 1.0f / (static_cast<float>(MW_) * static_cast<float>(MH_));
        #pragma omp parallel for schedule(static)
        for (int kx = 0; kx < MW_; ++kx) {
            cfloat* col = &buf_b_[static_cast<size_t>(kx) * MH_];
            fft_h_.exec(col, false);

            float cx = cos_kx_[kx];
            for (int ky = 0; ky < MH_; ++ky) {
                if (kx == 0 && ky == 0) {
                    col[0] = cfloat(0.0f, 0.0f);
                } else {
                    float eig = 4.0f - 2.0f * cx - 2.0f * cos_ky_[ky];
                    col[ky] *= (inv_area / eig);
                }
            }

            fft_h_.exec(col, true);
        }

        transpose_blocked(buf_b_.data(), buf_a_.data(), MW_, MH_);

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < PH_; ++y) {
            cfloat* row = &buf_a_[static_cast<size_t>(y) * MW_];
            fft_w_.exec(row, true);
            size_t out_row = static_cast<size_t>(y) * PW_;
            for (int x = 0; x < PW_; ++x) {
                out[out_row + x] = row[x].real();
            }
        }
    }

    double dot_product(const std::vector<float>& a, const std::vector<float>& b) const {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(PN_); ++i) {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return sum;
    }

    int W_, H_;
    int PW_, PH_;
    size_t PN_;
    int MW_, MH_;

    std::vector<float> wx_, wy_, phi_pad_, r_, z_, p_, Ap_;
    std::vector<cfloat> buf_a_, buf_b_;
    std::vector<float> cos_kx_, cos_ky_;
    Fft1DPlan fft_w_, fft_h_;
};

// ============================================================================
// Synthetic Fallback Generator (Now Supports Arbitrary Sizes + Skewed NaN Border)
// ============================================================================

void generate_synthetic_scene(
    int W, int H, float wavelength_mm,
    std::vector<cfloat>& slc1,
    std::vector<cfloat>& slc2,
    std::vector<float>& true_phase_rad,
    std::vector<float>& true_coh,
    GeoMetadata& meta)
{
    size_t N = static_cast<size_t>(W) * H;
    slc1.resize(N);
    slc2.resize(N);
    true_phase_rad.resize(N);
    true_coh.resize(N);

    meta.width = W;
    meta.height = H;
    meta.has_georef = true;
    // Arbitrary UTM Zone 14N geotransform (Mexico City area) for QGIS sanity testing
    meta.gt[0] = 480000.0; meta.gt[1] = 10.0; meta.gt[2] = 0.0;
    meta.gt[3] = 2155000.0; meta.gt[4] = 0.0; meta.gt[5] = -10.0;
    meta.projection_wkt = "EPSG:32614";

    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const float mm_to_rad = (4.0f * PI_F) / wavelength_mm;
    const float nan_f = std::numeric_limits<float>::quiet_NaN();

    for (int y = 0; y < H; ++y) {
        float yn = (static_cast<float>(y) / H) - 0.5f;
        for (int x = 0; x < W; ++x) {
            float xn = (static_cast<float>(x) / W) - 0.5f;
            size_t idx = static_cast<size_t>(y) * W + x;

            // Simulate a skewed satellite burst border with NaNs
            float skewed_y = yn + 0.15f * xn;
            if (std::abs(skewed_y) > 0.42f || std::abs(xn) > 0.46f) {
                slc1[idx] = cfloat(nan_f, nan_f);
                slc2[idx] = cfloat(nan_f, nan_f);
                true_phase_rad[idx] = nan_f;
                true_coh[idx] = 0.0f;
                continue;
            }

            float r1 = (xn + 0.12f) * (xn + 0.12f) / 0.018f + (yn - 0.05f) * (yn - 0.05f) / 0.014f;
            float r2 = (xn - 0.20f) * (xn - 0.20f) / 0.008f + (yn + 0.18f) * (yn + 0.18f) / 0.010f;
            float disp_mm = -95.0f * std::exp(-r1) - 40.0f * std::exp(-r2) + 12.0f * xn - 8.0f * yn;
            float phi = disp_mm * mm_to_rad;
            true_phase_rad[idx] = phi;

            float r_lake = (xn - 0.15f) * (xn - 0.15f) / 0.012f + (yn + 0.15f) * (yn + 0.15f) / 0.012f;
            float rho = (r_lake < 1.0f) ? 0.15f : 0.90f;
            true_coh[idx] = rho;

            float a_re = gauss(rng), a_im = gauss(rng);
            float n_re = gauss(rng), n_im = gauss(rng);

            float c = std::cos(phi);
            float s = std::sin(phi);
            float rot_re = a_re * c + a_im * s;
            float rot_im = a_im * c - a_re * s;
            float noise_scale = std::sqrt(1.0f - rho * rho);

            slc1[idx] = cfloat(a_re, a_im);
            slc2[idx] = cfloat(rho * rot_re + noise_scale * n_re,
                               rho * rot_im + noise_scale * n_im);
        }
    }
}

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --slc1 <path>         Reference OPERA CSLC-S1 .h5 or complex GeoTIFF\n"
        "  --slc2 <path>         Secondary OPERA CSLC-S1 .h5 or complex GeoTIFF\n"
        "  --subdataset <path>   HDF5 dataset path (default: /data/VV)\n"
        "  --stride-x <int>      Horizontal multilook stride (default: 6 for OPERA, 1 for synthetic)\n"
        "  --stride-y <int>      Vertical multilook stride (default: 3 for OPERA, 1 for synthetic)\n"
        "  --look-rx <int>       Coherence half-window in X (default: 3)\n"
        "  --look-ry <int>       Coherence half-window in Y (default: 3)\n"
        "  --window x y w h      Optional pixel crop window on input raster\n"
        "  --width <int>         Synthetic width when --slc1/--slc2 omitted (default: 900)\n"
        "  --height <int>        Synthetic height when --slc1/--slc2 omitted (default: 700)\n"
        "  --iters <int>         Max PCG iterations (default: 40)\n"
        "  --wavelength-mm <f>   Radar wavelength in mm (default: 55.4658 for Sentinel-1 C-band)\n"
        "  --alert-mm <f>        Displacement alert threshold in mm (default: 25.0)\n"
        "  --alert-coh <f>       Minimum coherence for alerts/stats (default: 0.65)\n"
        "  --out-dir <dir>       Output directory for COGs & JSON (default: ./out)\n",
        prog);
}

int main(int argc, char** argv) {
    GDALAllRegister();

    std::string slc1_path, slc2_path;
    std::string subdataset = "/data/VV";
    std::string out_dir = "out";
    int synth_w = 900, synth_h = 700; // Deliberately non-power-of-2 default
    int stride_x = -1, stride_y = -1;
    int look_rx = 3, look_ry = 3;
    int win_x = 0, win_y = 0, win_w = 0, win_h = 0;
    int max_iters = 40;
    float wavelength_mm = 55.4658f; // Sentinel-1 C-band (use 238.4 for NISAR L-band)
    float alert_threshold_mm = 25.0f;
    float min_alert_coh = 0.65f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--slc1" && i + 1 < argc)          slc1_path = argv[++i];
        else if (arg == "--slc2" && i + 1 < argc)          slc2_path = argv[++i];
        else if (arg == "--subdataset" && i + 1 < argc)    subdataset = argv[++i];
        else if (arg == "--out-dir" && i + 1 < argc)       out_dir = argv[++i];
        else if (arg == "--stride-x" && i + 1 < argc)      stride_x = std::stoi(argv[++i]);
        else if (arg == "--stride-y" && i + 1 < argc)      stride_y = std::stoi(argv[++i]);
        else if (arg == "--look-rx" && i + 1 < argc)       look_rx = std::stoi(argv[++i]);
        else if (arg == "--look-ry" && i + 1 < argc)       look_ry = std::stoi(argv[++i]);
        else if (arg == "--width" && i + 1 < argc)         synth_w = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc)        synth_h = std::stoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc)         max_iters = std::stoi(argv[++i]);
        else if (arg == "--wavelength-mm" && i + 1 < argc) wavelength_mm = std::stof(argv[++i]);
        else if (arg == "--alert-mm" && i + 1 < argc)      alert_threshold_mm = std::stof(argv[++i]);
        else if (arg == "--alert-coh" && i + 1 < argc)     min_alert_coh = std::stof(argv[++i]);
        else if (arg == "--window" && i + 4 < argc) {
            win_x = std::stoi(argv[++i]);
            win_y = std::stoi(argv[++i]);
            win_w = std::stoi(argv[++i]);
            win_h = std::stoi(argv[++i]);
        }
    }

    const bool synthetic = slc1_path.empty() || slc2_path.empty();
    if (stride_x <= 0) stride_x = synthetic ? 1 : 6; // OPERA 5m -> 30m
    if (stride_y <= 0) stride_y = synthetic ? 1 : 3; // OPERA 10m -> 30m

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    std::vector<cfloat> slc1, slc2;
    std::vector<float> true_phi, true_coh;
    GeoMetadata in_meta;

    if (synthetic) {
        std::printf("Running synthetic mode (%dx%d, skewed NaN border) on %d OpenMP threads\n",
                    synth_w, synth_h, threads);
        generate_synthetic_scene(synth_w, synth_h, wavelength_mm,
                                 slc1, slc2, true_phi, true_coh, in_meta);
    } else {
        GeoMetadata meta2;
        if (!read_slc_gdal(slc1_path, subdataset, win_x, win_y, win_w, win_h, slc1, in_meta) ||
            !read_slc_gdal(slc2_path, subdataset, win_x, win_y, win_w, win_h, slc2, meta2)) {
            return EXIT_FAILURE;
        }
        if (in_meta.width != meta2.width || in_meta.height != meta2.height) {
            std::fprintf(stderr, "Error: SLC dimensions mismatch (%dx%d vs %dx%d). "
                                 "Verify both files share the same OPERA Burst ID.\n",
                         in_meta.width, in_meta.height, meta2.width, meta2.height);
            return EXIT_FAILURE;
        }
        std::printf("Loaded real SLC pair (%dx%d) on %d OpenMP threads\n",
                    in_meta.width, in_meta.height, threads);
    }

    int out_w = std::max(1, in_meta.width / stride_x);
    int out_h = std::max(1, in_meta.height / stride_y);

    GeoMetadata out_meta = in_meta;
    out_meta.width  = out_w;
    out_meta.height = out_h;
    if (out_meta.has_georef) {
        out_meta.gt[1] *= stride_x;
        out_meta.gt[2] *= stride_y;
        out_meta.gt[4] *= stride_x;
        out_meta.gt[5] *= stride_y;
    }

    CpuInsarEngine engine(out_w, out_h);
    std::printf("Multilooked output: %dx%d (stride %dx%d) -> Padded FFT solver grid: %dx%d\n",
                out_w, out_h, stride_x, stride_y,
                engine.padded_width(), engine.padded_height());

    std::vector<float> wrapped, coh, unwrapped;
    std::vector<uint8_t> valid_mask;

    auto stats = engine.process_pair(
        slc1, slc2, in_meta.width, in_meta.height,
        stride_x, stride_y, look_rx, look_ry,
        wrapped, coh, unwrapped, valid_mask,
        max_iters, 1e-4f);

    // Reference unwrapped phase to median of high-coherence valid pixels, convert to LOS mm
    const float rad_to_mm = wavelength_mm / (4.0f * PI_F);
    std::vector<float> ref_samples;
    ref_samples.reserve(stats.valid_pixels);
    for (size_t i = 0; i < unwrapped.size(); ++i) {
        if (valid_mask[i] && coh[i] > 0.65f) {
            ref_samples.push_back(unwrapped[i]);
        }
    }
    if (ref_samples.empty()) {
        for (size_t i = 0; i < unwrapped.size(); ++i) {
            if (valid_mask[i]) ref_samples.push_back(unwrapped[i]);
        }
    }

    float median_ref = 0.0f;
    if (!ref_samples.empty()) {
        size_t mid = ref_samples.size() / 2;
        std::nth_element(ref_samples.begin(), ref_samples.begin() + mid, ref_samples.end());
        median_ref = ref_samples[mid];
    }

    std::vector<float> disp_mm(unwrapped.size(), 0.0f);
    for (size_t i = 0; i < unwrapped.size(); ++i) {
        if (valid_mask[i]) {
            disp_mm[i] = (unwrapped[i] - median_ref) * rad_to_mm;
        }
    }

    std::printf("------------------------------------------------------------\n");
    std::printf("Valid pixels inside burst:               %zu / %zu (%.1f%%)\n",
                stats.valid_pixels, unwrapped.size(),
                100.0 * static_cast<double>(stats.valid_pixels) / unwrapped.size());
    std::printf("Stage 1 (Multilook + Coherence):         %8.2f ms\n", stats.ifg_ms);
    std::printf("Stage 2 (Masked PCG Unwrap, %2d iters):   %8.2f ms (res=%.2e)\n",
                stats.iterations, stats.unwrap_ms, stats.rel_residual);
    std::printf("Total CPU Compute Time:                  %8.2f ms\n",
                stats.ifg_ms + stats.unwrap_ms);

    if (synthetic && stride_x == 1 && stride_y == 1) {
        double mean_diff = 0.0;
        size_t valid_count = 0;
        for (size_t i = 0; i < unwrapped.size(); ++i) {
            if (valid_mask[i] && true_coh[i] > 0.5f) {
                mean_diff += (unwrapped[i] - true_phi[i]);
                ++valid_count;
            }
        }
        mean_diff /= static_cast<double>(valid_count);

        double sq_err_mm = 0.0;
        for (size_t i = 0; i < unwrapped.size(); ++i) {
            if (valid_mask[i] && true_coh[i] > 0.5f) {
                double err_mm = ((unwrapped[i] - true_phi[i]) - mean_diff) * rad_to_mm;
                sq_err_mm += err_mm * err_mm;
            }
        }
        double rmse_mm = std::sqrt(sq_err_mm / static_cast<double>(valid_count));
        std::printf("Coherent-Area LOS Accuracy (C-band):     %8.2f mm RMSE\n", rmse_mm);
    }
    std::printf("------------------------------------------------------------\n");

    std::filesystem::create_directories(out_dir);
    bool ok1 = write_cog_f32(out_dir + "/wrapped_phase.tif", wrapped, valid_mask, out_meta);
    bool ok2 = write_cog_f32(out_dir + "/coherence.tif", coh, valid_mask, out_meta);
    bool ok3 = write_cog_f32(out_dir + "/displacement_mm.tif", disp_mm, valid_mask, out_meta);

    export_metadata_and_alerts(
        out_dir, out_meta, disp_mm, coh, valid_mask,
        alert_threshold_mm, min_alert_coh,
        stats.ifg_ms, stats.unwrap_ms, stats.iterations, stats.valid_pixels);

    if (ok1 && ok2 && ok3) {
        std::printf("Exported COGs, alerts.geojson, and summary.json to '%s/'\n", out_dir.c_str());
    } else {
        std::fprintf(stderr, "Error: Failed writing one or more output GeoTIFFs.\n");
        return EXIT_FAILURE;
    }

    return 0;
}
