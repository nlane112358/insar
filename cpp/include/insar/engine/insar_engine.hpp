#pragma once

#include "insar/core/types.hpp"

namespace insar {

class IInsarEngine {
public:
    virtual ~IInsarEngine() = default;
    virtual InsarProducts process(const SlcPair& pair, const ProcessingConfig& config) = 0;
};

} // namespace insar
