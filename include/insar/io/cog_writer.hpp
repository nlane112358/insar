#pragma once

#include "insar/core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace insar::io {

class IRasterWriter {
public:
    virtual ~IRasterWriter() = default;

    virtual bool write(
        const std::string& path,
        const std::vector<float>& data,
        const std::vector<uint8_t>& valid_mask,
        const GeoMetadata& meta) const = 0;
};

class GdalCogWriter final : public IRasterWriter {
public:
    bool write(
        const std::string& path,
        const std::vector<float>& data,
        const std::vector<uint8_t>& valid_mask,
        const GeoMetadata& meta) const override;
};

} // namespace insar::io
