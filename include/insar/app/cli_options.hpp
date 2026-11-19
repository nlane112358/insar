#pragma once

#include "insar/core/types.hpp"

#include <string>

namespace insar::app {

struct CliOptions {
    bool show_help = false;
    std::string slc1_path;
    std::string slc2_path;
    std::string subdataset = "/data/VV";
    std::string out_dir = "out";
    int synth_width = 900;
    int synth_height = 700;
    int stride_x = -1;
    int stride_y = -1;
    int look_rx = 3;
    int look_ry = 3;
    CropWindow window;
    int max_iters = 40;
    float wavelength_mm = 55.4658f;
    float alert_threshold_mm = 25.0f;
    float min_alert_coh = 0.65f;

    bool is_synthetic() const noexcept;
    int resolved_stride_x() const noexcept;
    int resolved_stride_y() const noexcept;
    ProcessingConfig to_processing_config() const noexcept;
};

class CliParser {
public:
    static CliOptions parse(int argc, char** argv);
    static void print_usage(const char* program_name);
};

} // namespace insar::app
