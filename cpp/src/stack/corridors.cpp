#include "insar/stack/corridors.hpp"

#include "insar/io/gdal_utils.hpp"

#include <cpl_conv.h>
#include <gdal_alg.h>
#include <gdal_priv.h>
#include <ogr_api.h>
#include <ogr_spatialref.h>
#include <ogrsf_frmts.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace insar::stack {

namespace {

using insar::io::CslStringList;
using insar::io::UniqueGdalDataset;
using insar::io::UniqueOgrFeature;
using insar::io::UniqueOgrGeom;
using insar::io::UniqueOgrTransform;

std::string get_feature_str(OGRFeature* feat, const char* field, const std::string& fallback = "") {
    int idx = feat->GetFieldIndex(field);
    if (idx < 0 || !feat->IsFieldSetAndNotNull(idx)) {
        return fallback;
    }
    const char* val = feat->GetFieldAsString(idx);
    return (val != nullptr && *val != '\0') ? std::string(val) : fallback;
}

} // namespace

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

std::string json_num(double v, int prec) {
    if (!std::isfinite(v)) {
        return "null";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
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
    const StackOptions& opt,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const DerivedFields& df,
    bool stack_mature,
    std::vector<CorridorMetric>& metrics_out,
    std::vector<AlertPoint>& alerts_out,
    size_t& total_snr_gated_out
) {
    insar::io::ensure_gdal_registered();
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
    UniqueOgrTransform to_wgs84(OGRCreateCoordinateTransformation(&dst_srs, &wgs84_srs));
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
    const double buf_units = (opt.buffer_m > 0.0) ? insar::io::meters_to_crs_units(dst_srs, opt.buffer_m, cy) : 0.0;

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

        UniqueOgrTransform to_dst(OGRCreateCoordinateTransformation(&src_srs, &dst_srs));
        UniqueOgrTransform src_to_wgs(OGRCreateCoordinateTransformation(&src_srs, &wgs84_srs));
        if (!to_dst) {
            throw std::runtime_error("Failed to create vector->raster coordinate transform");
        }

        OGRFeature* raw_feat = nullptr;
        while ((raw_feat = layer->GetNextFeature()) != nullptr) {
            UniqueOgrFeature feat(raw_feat);
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
            CslStringList r_opts;
            r_opts.set("ALL_TOUCHED", "TRUE");
            CPLErr r_err = GDALRasterizeGeometries(
                GDALDataset::ToHandle(sub_ds.get()), 1, bands,
                1, geoms, nullptr, nullptr, burn, r_opts.raw(), nullptr, nullptr);
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
    const StackOptions& opt,
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

} // namespace insar::stack
