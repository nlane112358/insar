#pragma once

#include "insar/analytics/anomaly_detector.hpp"
#include "insar/core/types.hpp"
#include "insar/io/cog_writer.hpp"

#include <memory>
#include <string>

namespace insar::io {

class ProductExporter {
public:
    ProductExporter(float alert_threshold_mm, float min_alert_coh);
    ProductExporter(
        float alert_threshold_mm,
        float min_alert_coh,
        std::unique_ptr<IRasterWriter> raster_writer);

    bool export_all(const std::string& out_dir, const InsarProducts& products) const;

private:
    bool write_alerts_geojson(const std::string& path, const AnomalyReport& report) const;
    bool write_summary_json(
        const std::string& path,
        const InsarProducts& products,
        const AnomalyReport& report) const;

    AnomalyDetector detector_;
    std::unique_ptr<IRasterWriter> raster_writer_;
};

} // namespace insar::io
