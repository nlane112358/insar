#include "insar/stack/closure.hpp"

#include "insar/stack/stack_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace insar::stack {

ClosureResult evaluate_triplet_closure(
    const StackOptions& opt,
    const RasterMeta& meta,
    const std::vector<float>& disp,
    const std::vector<float>& coh,
    const std::vector<uint8_t>& ref_mask,
    std::vector<uint8_t>& bad_mask_out,
    std::vector<float>& residual_map_out
) {
    ClosureResult res;
    res.thresh_mm = opt.closure_thresh_mm;
    res.max_bad_frac = opt.closure_max_bad_frac;
    res.mode = opt.closure_mode;

    const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
    bad_mask_out.assign(npix, 0);
    residual_map_out.assign(npix, kNanF);

    if (opt.closure_triplets.empty()) {
        return res;
    }

    std::vector<uint16_t> eval_cnt(npix, 0);
    std::vector<uint16_t> fail_cnt(npix, 0);
    std::vector<double> sum_sq_res(npix, 0.0);
    std::vector<float> tmp_r(npix, kNanF);
    const bool has_ref = std::any_of(ref_mask.begin(), ref_mask.end(), [](uint8_t v) { return v != 0; });

    for (const auto& trip : opt.closure_triplets) {
        RasterMeta meta_a, meta_b;
        std::vector<float> da = read_single_band_f32(trip.disp_a_path, meta_a);
        std::vector<float> db = read_single_band_f32(trip.disp_b_path, meta_b);
        assert_compatible_grid(meta, meta_a, "closure triplet A");
        assert_compatible_grid(meta, meta_b, "closure triplet B");

        const bool is_add_sub = (trip.mode == "add_sub");
        std::fill(tmp_r.begin(), tmp_r.end(), kNanF);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; ++i) {
            if (std::isfinite(disp[i]) && std::isfinite(da[i]) && std::isfinite(db[i])) {
                tmp_r[i] = is_add_sub ? (disp[i] + da[i] - db[i]) : (da[i] + db[i] - disp[i]);
            }
        }

        // Remove constant datum offset per triplet using coherent reference (or scene) median
        std::vector<float> offset_samples;
        offset_samples.reserve(npix / 8);
        if (has_ref) {
            for (size_t i = 0; i < npix; ++i) {
                if (ref_mask[i] && std::isfinite(tmp_r[i]) && std::isfinite(coh[i]) && coh[i] >= opt.min_coh) {
                    offset_samples.push_back(tmp_r[i]);
                }
            }
        }
        if (offset_samples.size() < 10) {
            offset_samples.clear();
            const size_t step = std::max<size_t>(1, npix / 250000);
            for (size_t i = 0; i < npix; i += step) {
                if (std::isfinite(tmp_r[i]) && std::isfinite(coh[i]) && coh[i] >= opt.min_coh) {
                    offset_samples.push_back(tmp_r[i]);
                }
            }
        }
        double median_offset = 0.0;
        if (!offset_samples.empty()) {
            const size_t mid = offset_samples.size() / 2;
            std::nth_element(offset_samples.begin(), offset_samples.begin() + mid, offset_samples.end());
            median_offset = static_cast<double>(offset_samples[mid]);
        }

        res.triplets_evaluated++;
        const double thresh = static_cast<double>(opt.closure_thresh_mm);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; ++i) {
            if (!std::isfinite(tmp_r[i])) continue;
            const double rc = static_cast<double>(tmp_r[i]) - median_offset;
            eval_cnt[i]++;
            sum_sq_res[i] += rc * rc;
            if (std::abs(rc) > thresh) {
                fail_cnt[i]++;
            }
        }
    }

    const bool strict_mode = (opt.closure_mode == "strict");
    size_t eval_coh_cnt = 0;
    size_t bad_coh_cnt = 0;
    size_t masked_total = 0;
    double total_sq = 0.0;

    #pragma omp parallel for reduction(+:eval_coh_cnt, bad_coh_cnt, masked_total, total_sq) schedule(static)
    for (size_t i = 0; i < npix; ++i) {
        if (eval_cnt[i] == 0) continue;
        const double rms_pix = std::sqrt(sum_sq_res[i] / static_cast<double>(eval_cnt[i]));
        residual_map_out[i] = static_cast<float>(rms_pix);
        const bool is_bad = strict_mode
            ? (fail_cnt[i] > 0)
            : (static_cast<int>(fail_cnt[i]) * 2 > static_cast<int>(eval_cnt[i]));
        if (is_bad) {
            bad_mask_out[i] = 1;
            masked_total++;
        }
        if (std::isfinite(coh[i]) && coh[i] >= opt.min_coh) {
            eval_coh_cnt++;
            total_sq += rms_pix * rms_pix;
            if (is_bad) {
                bad_coh_cnt++;
            }
        }
    }

    res.evaluated_coherent_pixels = eval_coh_cnt;
    res.masked_pixels = masked_total;
    if (eval_coh_cnt > 0) {
        res.bad_fraction = static_cast<double>(bad_coh_cnt) / static_cast<double>(eval_coh_cnt);
        res.rms_mm = std::sqrt(total_sq / static_cast<double>(eval_coh_cnt));
        res.pair_rejected = (res.bad_fraction > static_cast<double>(opt.closure_max_bad_frac));
    }
    return res;
}

} // namespace insar::stack
