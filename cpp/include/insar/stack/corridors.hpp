#pragma once

#include "insar/stack/stack_types.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace insar::stack {

std::string json_escape(const std::string& input);
std::string json_num(double v, int prec = 4);

std::string classify_corridor_risk(
    int valid_pixels,
    int alert_pixel_count,
    double mean_vel,
    double p95_abs_vel,
    float vel_alert_mm_yr,
    bool stack_mature,
    bool& maturity_gated_out);

void process_corridors(
    const StackOptions& opt,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const DerivedFields& df,
    bool stack_mature,
    std::vector<CorridorMetric>& metrics_out,
    std::vector<AlertPoint>& alerts_out,
    size_t& total_snr_gated_out);

void write_corridor_outputs(
    const std::filesystem::path& out_dir,
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
    const std::vector<AlertPoint>& alerts);

} // namespace insar::stack
