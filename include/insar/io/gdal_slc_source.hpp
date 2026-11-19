#pragma once

#include "insar/io/slc_source.hpp"

#include <string>

namespace insar {

class GdalSlcSource final : public ISlcSource {
public:
    GdalSlcSource(
        std::string slc1_path,
        std::string slc2_path,
        std::string subdataset,
        CropWindow window);

    SlcPair load() override;
    std::string describe(int threads) const override;

private:
    SlcRaster read_single(const std::string& path) const;

    std::string slc1_path_;
    std::string slc2_path_;
    std::string subdataset_;
    CropWindow window_;
    int loaded_width_ = 0;
    int loaded_height_ = 0;
};

} // namespace insar
