#include "insar/stack/normal_equations.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace insar::stack {

void update_state_incremental(
    std::vector<double>& state,
    size_t npix,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    double t0_days,
    double t1_days,
    double origin_days,
    float min_coh,
    bool redundant_pair,
    double weight_scale
) {
    const double tau_yr = (t1_days - t0_days) / kDaysPerYear;
    const double m_yr = (0.5 * (t0_days + t1_days) - origin_days) / kDaysPerYear;
    const double x1 = tau_yr;
    const double x2 = m_yr * tau_yr;
    const double x11 = x1 * x1;
    const double x12 = x1 * x2;
    const double x22 = x2 * x2;

    double* sum_w = state.data() + static_cast<size_t>(kBandSumW) * npix;
    double* s11 = state.data() + static_cast<size_t>(kBandS11) * npix;
    double* s12 = state.data() + static_cast<size_t>(kBandS12) * npix;
    double* s22 = state.data() + static_cast<size_t>(kBandS22) * npix;
    double* b1 = state.data() + static_cast<size_t>(kBandB1) * npix;
    double* b2 = state.data() + static_cast<size_t>(kBandB2) * npix;
    double* syy = state.data() + static_cast<size_t>(kBandSyy) * npix;
    double* cum_raw = state.data() + static_cast<size_t>(kBandCumDispRaw) * npix;
    double* valid_dt = state.data() + static_cast<size_t>(kBandValidDtYr) * npix;
    double* sum_coh = state.data() + static_cast<size_t>(kBandSumCoh) * npix;
    double* valid_cnt = state.data() + static_cast<size_t>(kBandValidCount) * npix;
    double* fp_cnt = state.data() + static_cast<size_t>(kBandFootprintCount) * npix;
    double* latest = state.data() + static_cast<size_t>(kBandLatestDisp) * npix;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        if (!redundant_pair) {
            latest[i] = kNanD;
        }
        const double c = static_cast<double>(coh[i]);
        const double d = static_cast<double>(disp[i]);
        if (!std::isfinite(c)) {
            continue;
        }
        const double g = std::clamp(c, 0.0, 1.0);
        if (std::isfinite(d)) {
            if (!redundant_pair || fp_cnt[i] == 0.0) {
                fp_cnt[i] += 1.0;
                sum_coh[i] += g;
            }
        }
        if (!std::isfinite(d) || c < static_cast<double>(min_coh)) {
            continue;
        }
        const double w = std::max(1e-4, g * g * weight_scale);
        sum_w[i] += w;
        s11[i] += w * x11;
        s12[i] += w * x12;
        s22[i] += w * x22;
        b1[i] += w * x1 * d;
        b2[i] += w * x2 * d;
        syy[i] += w * d * d;
        valid_cnt[i] += 1.0;
        if (!redundant_pair) {
            cum_raw[i] += d;
            valid_dt[i] += tau_yr;
            latest[i] = d;
        }
    }
}

DerivedFields derive_products(
    const std::vector<double>& state,
    size_t npix,
    const StackMetadata& smeta,
    float min_coh
) {
    DerivedFields out;
    out.vel.assign(npix, kNanF);
    out.accel.assign(npix, kNanF);
    out.cum_disp.assign(npix, kNanF);
    out.temp_coh.assign(npix, kNanF);
    out.unc.assign(npix, kNanF);
    out.snr.assign(npix, kNanF);
    out.latest_disp.assign(npix, kNanF);

    out.vel_d.assign(npix, kNanD);
    out.accel_d.assign(npix, kNanD);
    out.cum_d.assign(npix, kNanD);
    out.coh_d.assign(npix, kNanD);
    out.unc_d.assign(npix, kNanD);
    out.snr_d.assign(npix, kNanD);
    out.latest_d.assign(npix, kNanD);
    out.valid_cnt.assign(npix, 0);

    const double span_days = std::max(smeta.sum_dt_days, smeta.last_t1_days - smeta.first_t0_days);
    const double total_span_yr = span_days / kDaysPerYear;
    const double* s11 = state.data() + static_cast<size_t>(kBandS11) * npix;
    const double* s12 = state.data() + static_cast<size_t>(kBandS12) * npix;
    const double* s22 = state.data() + static_cast<size_t>(kBandS22) * npix;
    const double* b1 = state.data() + static_cast<size_t>(kBandB1) * npix;
    const double* b2 = state.data() + static_cast<size_t>(kBandB2) * npix;
    const double* syy = state.data() + static_cast<size_t>(kBandSyy) * npix;
    const double* cum_raw = state.data() + static_cast<size_t>(kBandCumDispRaw) * npix;
    const double* valid_dt = state.data() + static_cast<size_t>(kBandValidDtYr) * npix;
    const double* sum_coh = state.data() + static_cast<size_t>(kBandSumCoh) * npix;
    const double* valid_cnt = state.data() + static_cast<size_t>(kBandValidCount) * npix;
    const double* fp_cnt = state.data() + static_cast<size_t>(kBandFootprintCount) * npix;
    const double* latest = state.data() + static_cast<size_t>(kBandLatestDisp) * npix;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        int fc = static_cast<int>(std::llround(fp_cnt[i]));
        if (fc <= 0) {
            continue;
        }
        double tcoh = sum_coh[i] / static_cast<double>(fc);
        out.coh_d[i] = tcoh;
        out.temp_coh[i] = static_cast<float>(tcoh);

        int vc = static_cast<int>(std::llround(valid_cnt[i]));
        out.valid_cnt[i] = vc;
        if (std::isfinite(latest[i])) {
            out.latest_d[i] = latest[i];
            out.latest_disp[i] = static_cast<float>(latest[i]);
        }

        if (vc < 1 || s11[i] <= 1e-16 || tcoh < static_cast<double>(min_coh) * 0.5) {
            continue;
        }
        if (smeta.pair_count >= 3 && vc < std::max(2, static_cast<int>(std::ceil(0.35 * smeta.pair_count)))) {
            continue;
        }
        double v_lin = b1[i] / s11[i];
        double cdisp = cum_raw[i];
        if (total_span_yr > valid_dt[i] + 1e-9) {
            cdisp += (total_span_yr - valid_dt[i]) * v_lin;
        }

        out.vel_d[i] = v_lin;
        out.vel[i] = static_cast<float>(v_lin);
        out.cum_d[i] = cdisp;
        out.cum_disp[i] = static_cast<float>(cdisp);

        if (vc >= 2) {
            double wssr = std::max(0.0, syy[i] - v_lin * b1[i]);
            double se_v = std::sqrt(wssr / (static_cast<double>(vc - 1) * s11[i]));
            out.unc_d[i] = se_v;
            out.unc[i] = static_cast<float>(se_v);
            double snr_val = (se_v > 1e-9)
                ? (std::abs(v_lin) / se_v)
                : ((std::abs(v_lin) > 1e-6) ? 1e6 : 0.0);
            out.snr_d[i] = snr_val;
            out.snr[i] = static_cast<float>(std::min(snr_val, 1e6));
        }

        if (vc >= 3 && s11[i] > 0.0 && s22[i] > 0.0) {
            double det = s11[i] * s22[i] - s12[i] * s12[i];
            double norm_det = det / (s11[i] * s22[i]);
            if (norm_det > 1e-6) {
                double a_hat = (s11[i] * b2[i] - s12[i] * b1[i]) / det;
                out.accel_d[i] = a_hat;
                out.accel[i] = static_cast<float>(a_hat);
            }
        }
    }

    std::vector<double> abs_vels;
    abs_vels.reserve(npix / 4);
    double sum_v = 0.0;
    double sum_c = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        if (std::isfinite(out.vel_d[i])) {
            out.valid_stack_pixels++;
            sum_v += out.vel_d[i];
            sum_c += out.coh_d[i];
            abs_vels.push_back(std::abs(out.vel_d[i]));
        }
    }
    if (!abs_vels.empty()) {
        out.mean_vel = sum_v / static_cast<double>(abs_vels.size());
        out.mean_coh = sum_c / static_cast<double>(abs_vels.size());
        size_t p95_idx = static_cast<size_t>(std::floor(0.95 * static_cast<double>(abs_vels.size() - 1)));
        std::nth_element(abs_vels.begin(), abs_vels.begin() + p95_idx, abs_vels.end());
        out.p95_abs_vel = abs_vels[p95_idx];
    }
    return out;
}

} // namespace insar::stack
