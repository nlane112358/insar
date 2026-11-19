#pragma once

#include "insar/core/types.hpp"

#include <string>

namespace insar {

class ISlcSource {
public:
    virtual ~ISlcSource() = default;

    virtual SlcPair load() = 0;
    virtual std::string describe(int threads) const = 0;
};

} // namespace insar
