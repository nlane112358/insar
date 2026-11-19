#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace insar::stack {

inline constexpr double kDaysPerYear = 365.25;
inline constexpr float kNanF = std::numeric_limits<float>::quiet_NaN();
inline constexpr double kNanD = std::numeric_limits<double>::quiet_NaN();

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

struct StackOptions {
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

} // namespace insar::stack
