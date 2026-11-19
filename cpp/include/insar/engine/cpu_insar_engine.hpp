#pragma once

#include "insar/engine/insar_engine.hpp"
#include "insar/engine/interferogram_estimator.hpp"
#include "insar/engine/phase_unwrapper.hpp"

namespace insar {

class CpuInsarEngine final : public IInsarEngine {
public:
    CpuInsarEngine(int out_width, int out_height);

    InsarProducts process(const SlcPair& pair, const ProcessingConfig& config) override;

    int out_width() const noexcept { return out_w_; }
    int out_height() const noexcept { return out_h_; }
    int padded_width() const noexcept { return unwrapper_.padded_width(); }
    int padded_height() const noexcept { return unwrapper_.padded_height(); }

private:
    int out_w_ = 0;
    int out_h_ = 0;
    InterferogramEstimator ifg_estimator_;
    PcgPhaseUnwrapper unwrapper_;
};

} // namespace insar
