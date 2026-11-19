#include "insar/stack/corridors.hpp"
#include "insar/stack/normal_equations.hpp"
#include "insar/stack/nuisance.hpp"
#include "insar/stack/stack_types.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check_impl(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::fprintf(stderr, "Stack test assertion failed: (%s) at %s:%d\n", expr, file, line);
        std::exit(EXIT_FAILURE);
    }
}

#define STACK_CHECK(cond) check_impl((cond), #cond, __FILE__, __LINE__)

void test_incremental_normal_equations_and_acceleration() {
    const int w = 16;
    const int h = 16;
    const size_t npix = static_cast<size_t>(w) * h;

    std::vector<double> state(static_cast<size_t>(insar::stack::kNumStateBands) * npix, 0.0);
    std::fill(
        state.data() + static_cast<size_t>(insar::stack::kBandLatestDisp) * npix,
        state.data() + static_cast<size_t>(insar::stack::kNumStateBands) * npix,
        insar::stack::kNanD);

    // Quadratic cumulative trajectory d(t) = v0 * t_yr + 0.5 * a * t_yr^2
    const double v0 = -40.0;
    const double accel = -20.0;
    const double step_days = 36.525; // 0.1 yr per pass
    std::vector<float> disp(npix, 0.0f);
    std::vector<float> coh(npix, 0.9f);

    for (int k = 0; k < 5; ++k) {
        const double t0 = k * step_days;
        const double t1 = (k + 1) * step_days;
        const double t0_yr = t0 / insar::stack::kDaysPerYear;
        const double t1_yr = t1 / insar::stack::kDaysPerYear;
        const double inc = (v0 * t1_yr + 0.5 * accel * t1_yr * t1_yr) -
                           (v0 * t0_yr + 0.5 * accel * t0_yr * t0_yr);
        std::fill(disp.begin(), disp.end(), static_cast<float>(inc));
        insar::stack::update_state_incremental(
            state, npix, disp, coh, t0, t1, 0.0, 0.35f, false, 1.0);
    }

    insar::stack::StackMetadata smeta;
    smeta.origin_days = 0.0;
    smeta.first_t0_days = 0.0;
    smeta.last_t1_days = 5.0 * step_days;
    smeta.sum_dt_days = 5.0 * step_days;
    smeta.pair_count = 5;

    const auto df = insar::stack::derive_products(state, npix, smeta, 0.35f);
    STACK_CHECK(df.valid_stack_pixels == npix);
    STACK_CHECK(std::isfinite(df.accel_d[0]));
    STACK_CHECK(std::abs(df.accel_d[0] - accel) < 0.2);
    STACK_CHECK(df.snr_d[0] > 10.0);
}

void test_nuisance_ramp_dem_and_connected_component_isolation() {
    const int w = 64;
    const int h = 64;
    const size_t npix = static_cast<size_t>(w) * h;

    std::vector<float> disp(npix, 0.0f);
    std::vector<float> coh(npix, 0.92f);
    std::vector<float> dem(npix, 0.0f);
    std::vector<uint8_t> ref_mask(npix, 0);
    std::vector<uint8_t> deform_excl(npix, 0);

    // Reference patch on left side
    for (int y = 26; y <= 36; ++y) {
        for (int x = 4; x <= 12; ++x) {
            ref_mask[static_cast<size_t>(y) * w + x] = 1;
        }
    }

    // Disconnected coherence barrier at x = 50 (mountain/water gap)
    for (int y = 0; y < h; ++y) {
        coh[static_cast<size_t>(y) * w + 50] = 0.1f;
        coh[static_cast<size_t>(y) * w + 51] = 0.1f;
    }

    const double inv_hw = 2.0 / (w - 1);
    const double inv_hh = 2.0 / (h - 1);
    for (int y = 0; y < h; ++y) {
        const double v = y * inv_hh - 1.0;
        for (int x = 0; x < w; ++x) {
            const double u = x * inv_hw - 1.0;
            const size_t idx = static_cast<size_t>(y) * w + x;
            const double hill_dx = (x - 20.0) / 14.0;
            const double hill_dy = (y - 18.0) / 14.0;
            const double elev_m = 500.0 + 900.0 * std::exp(-(hill_dx * hill_dx + hill_dy * hill_dy))
                                + 250.0 * std::sin(3.5 * u) * std::cos(2.5 * v);
            dem[idx] = static_cast<float>(elev_m);
            const double tropo = 12.0 * (elev_m - 1000.0) / 1000.0;
            const double ramp = 6.0 * u - 4.0 * v;
            const double dx = x - 30.0;
            const double dy = y - 32.0;
            const double bowl = -35.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * 4.5 * 4.5));
            const double island_jump = (x >= 52) ? 28.0 : 0.0;
            disp[idx] = static_cast<float>(ramp + tropo + bowl + island_jump);
        }
    }

    const auto cc = insar::stack::extract_reference_connected_component(
        w, h, disp, coh, ref_mask, 0.35f);
    STACK_CHECK(!cc.empty());
    STACK_CHECK(cc[30 * w + 8] == 1);
    STACK_CHECK(cc[30 * w + 56] == 0);

    const auto ramp_res = insar::stack::remove_robust_ramp_and_dem(
        w, h, disp, coh, dem, deform_excl, ref_mask, cc, 1, 0.35f, 2.5f, 2);
    STACK_CHECK(ramp_res.applied);
    STACK_CHECK(ramp_res.dem_applied);
    if (std::abs(ramp_res.dem_slope_mm_per_km - 12.0) >= 1.5) {
        std::fprintf(stderr, "dem_slope=%.4f dem_std=%.2f corr_before=%.4f corr_after=%.4f\n",
                     ramp_res.dem_slope_mm_per_km, ramp_res.dem_std_m,
                     ramp_res.dem_corr_before, ramp_res.dem_corr_after);
    }
    STACK_CHECK(std::abs(ramp_res.dem_slope_mm_per_km - 12.0) < 1.5);
    // Bowl center should preserve ~ -35 mm subsidence relative to clean background
    STACK_CHECK(std::abs(disp[32 * w + 30] - (-35.0f)) < 2.0f);
    STACK_CHECK(std::abs(disp[10 * w + 10]) < 1.5f);
}

void test_corridor_maturity_gating() {
    bool gated = false;
    const std::string immature = insar::stack::classify_corridor_risk(
        200, 15, -45.0, 80.0, 15.0f, false, gated);
    STACK_CHECK(immature == "warning");
    STACK_CHECK(gated);

    const std::string mature = insar::stack::classify_corridor_risk(
        200, 15, -45.0, 80.0, 15.0f, true, gated);
    STACK_CHECK(mature == "critical");
    STACK_CHECK(!gated);
}

} // namespace

int main() {
    test_incremental_normal_equations_and_acceleration();
    test_nuisance_ramp_dem_and_connected_component_isolation();
    test_corridor_maturity_gating();
    std::printf("All insar C++ stack unit tests passed.\n");
    return 0;
}
