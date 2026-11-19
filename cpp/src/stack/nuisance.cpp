#include "insar/stack/nuisance.hpp"

#include "insar/io/gdal_utils.hpp"
#include "insar/stack/stack_io.hpp"

#include <ogr_spatialref.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace insar::stack {

namespace {

using insar::io::UniqueOgrTransform;

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

} // namespace

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
    const StackOptions& opt,
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
        UniqueOgrTransform ct(OGRCreateCoordinateTransformation(&wgs84, &dst_srs));
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
        const double crs_radius = insar::io::meters_to_crs_units(dst_srs, opt.ref_radius_m, opt.ref_lat);
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

std::vector<uint8_t> build_deform_exclusion_mask(
    const StackOptions& opt,
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

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        if (ref_mask[i]) {
            excl[i] = 0;
        }
    }
    return excl;
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

    std::vector<size_t> fit_idx = unmasked_idx;
    if (fit_idx.size() < std::max<size_t>(30, all_coh_idx.size() / 10)) {
        fit_idx = all_coh_idx;
        excluded_prior_cnt = 0;
    }

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

    if (!run_irls(fit_idx, coeffs, mad_sigma, 0, 2)) {
        return res;
    }

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

    run_irls(stable_idx, coeffs, mad_sigma, 2, 6);

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

} // namespace insar::stack
