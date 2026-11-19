#pragma once

#include "insar/io/slc_source.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace insar {

class SyntheticSlcSource final : public ISlcSource {
public:
    SyntheticSlcSource(
        int width,
        int height,
        float wavelength_mm,
        uint32_t seed = 42);

    SlcPair load() override;
    std::string describe(int threads) const override;
    std::optional<SyntheticTruth> ground_truth() const override { return truth_; }

private:
    int width_ = 0;
    int height_ = 0;
    float wavelength_mm_ = 55.4658f;
    uint32_t seed_ = 42;
    std::optional<SyntheticTruth> truth_;
};

} // namespace insar
