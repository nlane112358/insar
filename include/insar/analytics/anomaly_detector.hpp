#pragma once

#include "insar/core/types.hpp"

#include <vector>

namespace insar {

struct AnomalyPoint {
    double lon = 0.0;
    double lat = 0.0;
    float disp_mm = 0.0f;
    float coherence = 0.0f;
};

struct AnomalyReport {
    Wgs84BBox bbox;
    float min_displacement_mm = 0.0f;
    float max_displacement_mm = 0.0f;
    std::vector<AnomalyPoint> anomalies;
};

class AnomalyDetector {
public:
    AnomalyDetector(
        float alert_threshold_mm,
        float min_coherence,
        int cluster_size = 32);

    AnomalyReport analyze(const InsarProducts& products) const;

private:
    float alert_threshold_mm_ = 25.0f;
    float min_coherence_ = 0.65f;
    int cluster_size_ = 32;
};

} // namespace insar
