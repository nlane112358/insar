#include "insar/app/insar_application.hpp"

#include "insar/engine/cpu_insar_engine.hpp"
#include "insar/io/gdal_slc_source.hpp"
#include "insar/io/product_exporter.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace insar::app {

InsarApplication::InsarApplication(CliOptions options)
    : options_(std::move(options))
{
}

std::unique_ptr<ISlcSource> InsarApplication::create_slc_source() const {
    return std::make_unique<GdalSlcSource>(
        options_.slc1_path,
        options_.slc2_path,
        options_.subdataset,
        options_.window);
}

int InsarApplication::run() const {
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    const ProcessingConfig config = options_.to_processing_config();
    auto source = create_slc_source();

    const SlcPair pair = source->load();
    std::printf("%s\n", source->describe(threads).c_str());

    const int out_w = std::max(1, pair.width() / config.stride_x);
    const int out_h = std::max(1, pair.height() / config.stride_y);

    CpuInsarEngine engine(out_w, out_h);
    std::printf("Multilooked output: %dx%d (stride %dx%d) -> Padded FFT solver grid: %dx%d\n",
                out_w, out_h, config.stride_x, config.stride_y,
                engine.padded_width(), engine.padded_height());

    const InsarProducts products = engine.process(pair, config);
    const ProcessingStats& stats = products.stats;
    const size_t out_n = products.unwrapped_phase.size();

    std::printf("------------------------------------------------------------\n");
    std::printf("Valid pixels inside burst:               %zu / %zu (%.1f%%)\n",
                stats.valid_pixels, out_n,
                100.0 * static_cast<double>(stats.valid_pixels) / out_n);
    std::printf("Stage 1 (Multilook + Coherence):         %8.2f ms\n", stats.ifg_ms);
    std::printf("Stage 2 (Masked PCG Unwrap, %2d iters):   %8.2f ms (res=%.2e)\n",
                stats.iterations, stats.unwrap_ms, stats.rel_residual);
    std::printf("Total CPU Compute Time:                  %8.2f ms\n",
                stats.ifg_ms + stats.unwrap_ms);
    std::printf("------------------------------------------------------------\n");

    const io::ProductExporter exporter(options_.alert_threshold_mm, options_.min_alert_coh);
    if (!exporter.export_all(options_.out_dir, products)) {
        std::fprintf(stderr, "Error: Failed writing one or more output GeoTIFFs.\n");
        return EXIT_FAILURE;
    }

    std::printf("Exported COGs, alerts.geojson, and summary.json to '%s/'\n",
                options_.out_dir.c_str());
    return 0;
}

} // namespace insar::app
