#include "insar/app/cli_options.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace insar::app {

bool CliOptions::is_synthetic() const noexcept {
    return slc1_path.empty() || slc2_path.empty();
}

int CliOptions::resolved_stride_x() const noexcept {
    if (stride_x > 0) {
        return stride_x;
    }
    return is_synthetic() ? 1 : 6; // OPERA 5m -> 30m
}

int CliOptions::resolved_stride_y() const noexcept {
    if (stride_y > 0) {
        return stride_y;
    }
    return is_synthetic() ? 1 : 3; // OPERA 10m -> 30m
}

ProcessingConfig CliOptions::to_processing_config() const noexcept {
    ProcessingConfig cfg;
    cfg.stride_x = resolved_stride_x();
    cfg.stride_y = resolved_stride_y();
    cfg.look_rx = look_rx;
    cfg.look_ry = look_ry;
    cfg.max_iters = max_iters;
    cfg.wavelength_mm = wavelength_mm;
    return cfg;
}

CliOptions CliParser::parse(int argc, char** argv) {
    CliOptions opts;

    auto require_args = [&](int index, int count, const std::string& flag) {
        if (index + count >= argc) {
            throw std::invalid_argument("Missing value for option: " + flag);
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.show_help = true;
            return opts;
        } else if (arg == "--slc1") {
            require_args(i, 1, arg);
            opts.slc1_path = argv[++i];
        } else if (arg == "--slc2") {
            require_args(i, 1, arg);
            opts.slc2_path = argv[++i];
        } else if (arg == "--subdataset") {
            require_args(i, 1, arg);
            opts.subdataset = argv[++i];
        } else if (arg == "--out-dir") {
            require_args(i, 1, arg);
            opts.out_dir = argv[++i];
        } else if (arg == "--stride-x") {
            require_args(i, 1, arg);
            opts.stride_x = std::stoi(argv[++i]);
        } else if (arg == "--stride-y") {
            require_args(i, 1, arg);
            opts.stride_y = std::stoi(argv[++i]);
        } else if (arg == "--look-rx") {
            require_args(i, 1, arg);
            opts.look_rx = std::stoi(argv[++i]);
        } else if (arg == "--look-ry") {
            require_args(i, 1, arg);
            opts.look_ry = std::stoi(argv[++i]);
        } else if (arg == "--width") {
            require_args(i, 1, arg);
            opts.synth_width = std::stoi(argv[++i]);
        } else if (arg == "--height") {
            require_args(i, 1, arg);
            opts.synth_height = std::stoi(argv[++i]);
        } else if (arg == "--iters") {
            require_args(i, 1, arg);
            opts.max_iters = std::stoi(argv[++i]);
        } else if (arg == "--wavelength-mm") {
            require_args(i, 1, arg);
            opts.wavelength_mm = std::stof(argv[++i]);
        } else if (arg == "--alert-mm") {
            require_args(i, 1, arg);
            opts.alert_threshold_mm = std::stof(argv[++i]);
        } else if (arg == "--alert-coh") {
            require_args(i, 1, arg);
            opts.min_alert_coh = std::stof(argv[++i]);
        } else if (arg == "--window") {
            require_args(i, 4, arg);
            opts.window.x = std::stoi(argv[++i]);
            opts.window.y = std::stoi(argv[++i]);
            opts.window.width = std::stoi(argv[++i]);
            opts.window.height = std::stoi(argv[++i]);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (opts.synth_width <= 0 || opts.synth_height <= 0) {
        throw std::invalid_argument("--width and --height must be positive");
    }
    if (opts.look_rx < 0 || opts.look_ry < 0) {
        throw std::invalid_argument("--look-rx and --look-ry must be non-negative");
    }
    if (opts.max_iters < 0) {
        throw std::invalid_argument("--iters must be non-negative");
    }
    if (opts.wavelength_mm <= 0.0f) {
        throw std::invalid_argument("--wavelength-mm must be positive");
    }

    return opts;
}

void CliParser::print_usage(const char* program_name) {
    std::printf(
        "Usage: %s [options]\n"
        "  --slc1 <path>         Reference OPERA CSLC-S1 .h5 or complex GeoTIFF\n"
        "  --slc2 <path>         Secondary OPERA CSLC-S1 .h5 or complex GeoTIFF\n"
        "  --subdataset <path>   HDF5 dataset path (default: /data/VV)\n"
        "  --stride-x <int>      Horizontal multilook stride (default: 6 for OPERA, 1 for synthetic)\n"
        "  --stride-y <int>      Vertical multilook stride (default: 3 for OPERA, 1 for synthetic)\n"
        "  --look-rx <int>       Coherence half-window in X (default: 3)\n"
        "  --look-ry <int>       Coherence half-window in Y (default: 3)\n"
        "  --window x y w h      Optional pixel crop window on input raster\n"
        "  --width <int>         Synthetic width when --slc1/--slc2 omitted (default: 900)\n"
        "  --height <int>        Synthetic height when --slc1/--slc2 omitted (default: 700)\n"
        "  --iters <int>         Max PCG iterations (default: 40)\n"
        "  --wavelength-mm <f>   Radar wavelength in mm (default: 55.4658 for Sentinel-1 C-band)\n"
        "  --alert-mm <f>        Displacement alert threshold in mm (default: 25.0)\n"
        "  --alert-coh <f>       Minimum coherence for alerts/stats (default: 0.65)\n"
        "  --out-dir <dir>       Output directory for COGs & JSON (default: ./out)\n",
        program_name);
}

} // namespace insar::app
