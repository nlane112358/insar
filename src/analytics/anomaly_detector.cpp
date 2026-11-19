#include "insar/analytics/anomaly_detector.hpp"

#include "insar/io/gdal_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace insar {

AnomalyDetector::AnomalyDetector(
    float alert_threshold_mm,
    float min_coherence,
    int cluster_size)
    : alert_threshold_mm_(alert_threshold_mm),
      min_coherence_(min_coherence),
      cluster_size_(cluster_size)
{
    if (cluster_size_ <= 0) {
        throw std::invalid_argument("AnomalyDetector cluster_size must be positive");
    }
}

AnomalyReport AnomalyDetector::analyze(const InsarProducts& products) const {
    AnomalyReport report;
    const GeoMetadata& meta = products.metadata;
    const auto& disp_mm = products.displacement_mm;
    const auto& coh = products.coherence;
    const auto& valid_mask = products.valid_mask;

    io::GeoCoordinateTransformer transformer(meta);
    report.bbox = transformer.compute_wgs84_bbox();

    bool has_stats = false;
    for (size_t i = 0; i < disp_mm.size(); ++i) {
        if (!valid_mask[i] || coh[i] < min_coherence_ || !std::isfinite(disp_mm[i])) {
            continue;
        }
        if (!has_stats) {
            report.min_displacement_mm = disp_mm[i];
            report.max_displacement_mm = disp_mm[i];
            has_stats = true;
        } else {
            report.min_displacement_mm = std::min(report.min_displacement_mm, disp_mm[i]);
            report.max_displacement_mm = std::max(report.max_displacement_mm, disp_mm[i]);
        }
    }

    for (int by = 0; by < meta.height; by += cluster_size_) {
        const int y_end = std::min(by + cluster_size_, meta.height);
        for (int bx = 0; bx < meta.width; bx += cluster_size_) {
            const int x_end = std::min(bx + cluster_size_, meta.width);
            int best_x = -1;
            int best_y = -1;
            float worst_abs = alert_threshold_mm_;

            for (int y = by; y < y_end; ++y) {
                for (int x = bx; x < x_end; ++x) {
                    const size_t idx = static_cast<size_t>(y) * meta.width + x;
                    if (!valid_mask[idx] || coh[idx] < min_coherence_ || !std::isfinite(disp_mm[idx])) {
                        continue;
                    }
                    const float a = std::abs(disp_mm[idx]);
                    if (a >= worst_abs) {
                        worst_abs = a;
                        best_x = x;
                        best_y = y;
                    }
                }
            }

            if (best_x >= 0) {
                const size_t idx = static_cast<size_t>(best_y) * meta.width + best_x;
                double lon = 0.0;
                double lat = 0.0;
                if (transformer.pixel_to_lonlat(best_x + 0.5, best_y + 0.5, lon, lat)) {
                    report.anomalies.push_back({lon, lat, disp_mm[idx], coh[idx]});
                }
            }
        }
    }

    return report;
}

} // namespace insar
