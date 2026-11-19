#include "insar/engine/cpu_insar_engine.hpp"

#include "insar/engine/phase_calibrator.hpp"

#include <chrono>
#include <stdexcept>

namespace insar {

CpuInsarEngine::CpuInsarEngine(int out_width, int out_height)
    : out_w_(out_width),
      out_h_(out_height),
      unwrapper_(out_width, out_height)
{
    if (out_w_ <= 0 || out_h_ <= 0) {
        throw std::invalid_argument("CpuInsarEngine output dimensions must be positive");
    }
}

InsarProducts CpuInsarEngine::process(const SlcPair& pair, const ProcessingConfig& config) {
    using Clock = std::chrono::steady_clock;

    pair.validate();

    InsarProducts products;
    products.metadata = pair.reference.metadata().multilooked(
        out_w_, out_h_, config.stride_x, config.stride_y);

    const size_t out_n = static_cast<size_t>(out_w_) * out_h_;
    products.wrapped_phase.assign(out_n, 0.0f);
    products.coherence.assign(out_n, 0.0f);
    products.unwrapped_phase.assign(out_n, 0.0f);
    products.valid_mask.assign(out_n, 0);

    const auto t0 = Clock::now();
    products.stats.valid_pixels = ifg_estimator_.compute(
        pair,
        out_w_,
        out_h_,
        config.stride_x,
        config.stride_y,
        config.look_rx,
        config.look_ry,
        products.wrapped_phase,
        products.coherence,
        products.valid_mask);
    const auto t1 = Clock::now();

    const UnwrapResult unwrap_res = unwrapper_.unwrap(
        products.wrapped_phase,
        products.coherence,
        products.valid_mask,
        products.unwrapped_phase,
        config.max_iters,
        config.tolerance,
        config.min_edge_weight);
    const auto t2 = Clock::now();

    products.stats.iterations = unwrap_res.iterations;
    products.stats.rel_residual = unwrap_res.rel_residual;
    products.stats.ifg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    products.stats.unwrap_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    products.displacement_mm = PhaseCalibrator::to_displacement_mm(
        products.unwrapped_phase,
        products.coherence,
        products.valid_mask,
        config.wavelength_mm,
        config.min_ref_coherence);

    return products;
}

} // namespace insar
