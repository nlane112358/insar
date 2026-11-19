#pragma once

#include "insar/core/types.hpp"

#include <optional>
#include <string>

namespace insar {

class ISlcSource {
public:
    virtual ~ISlcSource() = default;

    virtual SlcPair load() = 0;
    virtual std::string describe(int threads) const = 0;
    virtual std::optional<SyntheticTruth> ground_truth() const { return std::nullopt; }
};

} // namespace insar
