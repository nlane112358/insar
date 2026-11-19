#include "insar/io/product_exporter.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace insar::io {

ProductExporter::ProductExporter(float alert_threshold_mm, float min_alert_coh)
    : ProductExporter(
          alert_threshold_mm,
          min_alert_coh,
          std::make_unique<GdalCogWriter>())
{
}

ProductExporter::ProductExporter(
    float alert_threshold_mm,
    float min_alert_coh,
    std::unique_ptr<IRasterWriter> raster_writer)
    : detector_(alert_threshold_mm, min_alert_coh),
      raster_writer_(std::move(raster_writer))
{
    if (!raster_writer_) {
        throw std::invalid_argument("ProductExporter requires a non-null raster writer");
    }
}

bool ProductExporter::export_all(const std::string& out_dir, const InsarProducts& products) const {
    std::filesystem::create_directories(out_dir);

    const bool ok1 = raster_writer_->write(
        out_dir + "/wrapped_phase.tif", products.wrapped_phase, products.valid_mask, products.metadata);
    const bool ok2 = raster_writer_->write(
        out_dir + "/coherence.tif", products.coherence, products.valid_mask, products.metadata);
    const bool ok3 = raster_writer_->write(
        out_dir + "/displacement_mm.tif", products.displacement_mm, products.valid_mask, products.metadata);

    const AnomalyReport report = detector_.analyze(products);
    const bool ok_gj = write_alerts_geojson(out_dir + "/alerts.geojson", report);
    const bool ok_sj = write_summary_json(out_dir + "/summary.json", products, report);

    return ok1 && ok2 && ok3 && ok_gj && ok_sj;
}

bool ProductExporter::write_alerts_geojson(
    const std::string& path,
    const AnomalyReport& report) const
{
    std::ofstream gj(path);
    if (!gj) {
        return false;
    }

    gj << "{\"type\":\"FeatureCollection\",\"features\":[";
    for (size_t i = 0; i < report.anomalies.size(); ++i) {
        const auto& a = report.anomalies[i];
        if (i > 0) {
            gj << ",";
        }
        gj << "{\"type\":\"Feature\",\"geometry\":{\"type\":\"Point\",\"coordinates\":["
           << std::setprecision(9) << a.lon << "," << a.lat
           << "]},\"properties\":{\"displacement_mm\":"
           << std::setprecision(6) << a.disp_mm << ",\"coherence\":" << a.coherence << "}}";
    }
    gj << "]}\n";
    return static_cast<bool>(gj);
}

bool ProductExporter::write_summary_json(
    const std::string& path,
    const InsarProducts& products,
    const AnomalyReport& report) const
{
    std::ofstream sj(path);
    if (!sj) {
        return false;
    }

    const GeoMetadata& meta = products.metadata;
    const ProcessingStats& stats = products.stats;

    sj << "{\n"
       << "  \"width\": " << meta.width << ",\n"
       << "  \"height\": " << meta.height << ",\n"
       << "  \"valid_pixels\": " << stats.valid_pixels << ",\n"
       << "  \"ifg_ms\": " << stats.ifg_ms << ",\n"
       << "  \"unwrap_ms\": " << stats.unwrap_ms << ",\n"
       << "  \"processing_time_ms\": " << (stats.ifg_ms + stats.unwrap_ms) << ",\n"
       << "  \"iterations\": " << stats.iterations << ",\n"
       << "  \"min_displacement_mm\": " << report.min_displacement_mm << ",\n"
       << "  \"max_displacement_mm\": " << report.max_displacement_mm << ",\n"
       << "  \"anomalies_detected\": " << report.anomalies.size() << ",\n";

    if (report.bbox.valid) {
        sj << std::setprecision(9)
           << "  \"bbox_wgs84\": [" << report.bbox.min_lon << ", " << report.bbox.min_lat << ", "
           << report.bbox.max_lon << ", " << report.bbox.max_lat << "]\n";
    } else {
        sj << "  \"bbox_wgs84\": null\n";
    }
    sj << "}\n";
    return static_cast<bool>(sj);
}

} // namespace insar::io
