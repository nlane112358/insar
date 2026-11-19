#include "insar/analytics/anomaly_detector.hpp"
#include "insar/app/cli_options.hpp"
#include "insar/core/types.hpp"
#include "insar/engine/cpu_insar_engine.hpp"
#include "insar/engine/phase_calibrator.hpp"
#include "insar/engine/phase_unwrapper.hpp"
#include "insar/engine/preconditioner.hpp"
#include "insar/engine/weighted_laplacian.hpp"
#include "insar/io/cog_writer.hpp"
#include "insar/io/gdal_slc_source.hpp"
#include "insar/io/gdal_utils.hpp"
#include "insar/io/product_exporter.hpp"
#include "insar/io/synthetic_slc_source.hpp"
#include "insar/math/fft.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_impl(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::fprintf(stderr, "Test assertion failed: (%s) at %s:%d\n", expr, file, line);
        std::exit(EXIT_FAILURE);
    }
}

#define TEST_CHECK(cond) check_impl((cond), #cond, __FILE__, __LINE__)

void test_fft_round_trip() {
    insar::math::Fft1DPlan plan(16);
    std::vector<insar::Complex> data(16);
    for (int i = 0; i < 16; ++i) {
        data[i] = insar::Complex(std::sin(0.3f * i), std::cos(0.7f * i));
    }
    const auto original = data;

    plan.execute(data.data(), false);
    plan.execute(data.data(), true);

    for (int i = 0; i < 16; ++i) {
        const insar::Complex reconstructed = data[i] / 16.0f;
        TEST_CHECK(std::abs(reconstructed - original[i]) < 1e-4f);
    }

    bool threw = false;
    try {
        insar::math::Fft1DPlan bad(12);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    TEST_CHECK(threw);
}

void test_poisson_preconditioner_inverts_laplacian() {
    const int pw = 16;
    const int ph = 16;
    insar::NeumannPoissonPreconditioner precond(pw, ph);

    std::vector<float> rhs(pw * ph, 0.0f);
    std::vector<float> sol(pw * ph, 0.0f);
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            rhs[y * pw + x] = std::cos(insar::math::kPi * (x + 0.5f) / pw) *
                              std::cos(insar::math::kPi * (y + 0.5f) / ph);
        }
    }

    precond.apply(rhs, sol);

    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            const size_t idx = static_cast<size_t>(y) * pw + x;
            const float c = sol[idx];
            float lap = 0.0f;
            if (x + 1 < pw) lap += (c - sol[idx + 1]);
            if (x > 0)      lap += (c - sol[idx - 1]);
            if (y + 1 < ph) lap += (c - sol[idx + pw]);
            if (y > 0)      lap += (c - sol[idx - pw]);
            TEST_CHECK(std::abs(lap - rhs[idx]) < 1e-4f);
        }
    }
}

void test_weighted_laplacian_symmetry() {
    const int w = 8;
    const int h = 8;
    insar::WeightedLaplacianSystem lap(w, h);

    std::vector<float> psi(w * h, 0.0f);
    std::vector<float> coh(w * h, 0.8f);
    std::vector<uint8_t> mask(w * h, 1);
    for (int i = 0; i < w * h; ++i) {
        psi[i] = 0.1f * i;
        coh[i] = 0.2f + 0.7f * ((i % 7) / 6.0f);
    }

    std::vector<float> rhs(w * h, 0.0f);
    lap.build(w, h, psi, coh, mask, 1e-5f, rhs);

    std::vector<float> p(w * h), q(w * h), Ap(w * h), Aq(w * h);
    for (int i = 0; i < w * h; ++i) {
        p[i] = std::sin(0.5f * i);
        q[i] = std::cos(0.3f * i);
    }
    lap.apply(p, Ap);
    lap.apply(q, Aq);

    double dot_pAq = 0.0;
    double dot_qAp = 0.0;
    for (int i = 0; i < w * h; ++i) {
        dot_pAq += static_cast<double>(p[i]) * Aq[i];
        dot_qAp += static_cast<double>(q[i]) * Ap[i];
    }
    TEST_CHECK(std::abs(dot_pAq - dot_qAp) < 1e-4);
}

void test_geo_metadata_transforms() {
    insar::GeoMetadata meta;
    meta.width = 200;
    meta.height = 100;
    meta.has_georef = true;
    meta.geotransform = {480000.0, 10.0, 0.0, 2155000.0, 0.0, -10.0};
    meta.projection_wkt = "EPSG:32614";

    const auto cropped = meta.cropped(5, 10, 100, 80);
    TEST_CHECK(cropped.width == 100 && cropped.height == 80);
    TEST_CHECK(std::abs(cropped.geotransform[0] - 480050.0) < 1e-9);
    TEST_CHECK(std::abs(cropped.geotransform[3] - 2154900.0) < 1e-9);

    const auto ml = cropped.multilooked(50, 40, 2, 2);
    TEST_CHECK(ml.width == 50 && ml.height == 40);
    TEST_CHECK(std::abs(ml.geotransform[1] - 20.0) < 1e-9);
    TEST_CHECK(std::abs(ml.geotransform[5] + 20.0) < 1e-9);
}

class IdentityPreconditioner final : public insar::IPreconditioner {
public:
    void apply(const std::vector<float>& in, std::vector<float>& out) override {
        out = in;
    }
};

void test_pcg_unwrapper_polymorphic_preconditioner() {
    const int w = 16;
    const int h = 16;
    std::vector<float> wrapped(w * h, 0.0f);
    std::vector<float> coh(w * h, 0.95f);
    std::vector<uint8_t> mask(w * h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float true_phi = 0.35f * x + 0.25f * y;
            wrapped[y * w + x] = insar::math::wrap_phase(true_phi);
        }
    }

    insar::PcgPhaseUnwrapper unwrapper(w, h, std::make_unique<IdentityPreconditioner>());
    std::vector<float> unwrapped;
    const auto res = unwrapper.unwrap(wrapped, coh, mask, unwrapped, 100, 1e-5f, 1e-5f);
    TEST_CHECK(res.iterations > 0);
    TEST_CHECK(res.rel_residual < 1e-3);

    const float dx = unwrapped[1] - unwrapped[0];
    const float dy = unwrapped[w] - unwrapped[0];
    TEST_CHECK(std::abs(dx - 0.35f) < 1e-3f);
    TEST_CHECK(std::abs(dy - 0.25f) < 1e-3f);
}

void test_synthetic_engine_end_to_end() {
    insar::SyntheticSlcSource source(450, 350, 55.4658f);
    const insar::SlcPair pair = source.load();
    const auto truth = source.ground_truth();
    TEST_CHECK(truth.has_value());

    insar::ProcessingConfig cfg;
    cfg.stride_x = 1;
    cfg.stride_y = 1;
    cfg.look_rx = 3;
    cfg.look_ry = 3;
    cfg.max_iters = 40;

    insar::CpuInsarEngine engine(450, 350);
    TEST_CHECK(engine.padded_width() == 512);
    TEST_CHECK(engine.padded_height() == 512);

    const insar::InsarProducts products = engine.process(pair, cfg);
    TEST_CHECK(products.metadata.width == 450);
    TEST_CHECK(products.metadata.height == 350);
    TEST_CHECK(products.stats.valid_pixels > 100000);
    TEST_CHECK(products.displacement_mm.size() == 450u * 350u);

    const double rmse_mm = insar::PhaseCalibrator::compute_coherent_rmse_mm(
        products.unwrapped_phase, products.valid_mask, *truth, cfg.wavelength_mm);
    TEST_CHECK(rmse_mm < 1.0);
}

class RecordingRasterWriter final : public insar::io::IRasterWriter {
public:
    mutable std::vector<std::string> written_paths;

    bool write(
        const std::string& path,
        const std::vector<float>& /*data*/,
        const std::vector<uint8_t>& /*valid_mask*/,
        const insar::GeoMetadata& /*meta*/) const override
    {
        written_paths.push_back(path);
        return true;
    }
};

void test_anomaly_detector_and_exporter() {
    insar::InsarProducts products;
    products.metadata.width = 64;
    products.metadata.height = 64;
    products.metadata.has_georef = true;
    products.metadata.geotransform = {480000.0, 10.0, 0.0, 2155000.0, 0.0, -10.0};
    products.metadata.projection_wkt = "EPSG:32614";
    products.wrapped_phase.assign(64 * 64, 0.0f);
    products.coherence.assign(64 * 64, 0.9f);
    products.unwrapped_phase.assign(64 * 64, 0.0f);
    products.displacement_mm.assign(64 * 64, -5.0f);
    products.valid_mask.assign(64 * 64, 1);
    products.stats.valid_pixels = 64 * 64;
    products.stats.ifg_ms = 1.5;
    products.stats.unwrap_ms = 2.5;
    products.stats.iterations = 4;

    products.displacement_mm[10 * 64 + 12] = -42.0f;
    products.displacement_mm[45 * 64 + 50] = 35.0f;

    auto writer = std::make_unique<RecordingRasterWriter>();
    const auto* writer_ptr = writer.get();
    insar::io::ProductExporter exporter(25.0f, 0.65f, std::move(writer));

    const auto tmp_dir = std::filesystem::temp_directory_path() / "insar_cpp_test_out";
    std::filesystem::remove_all(tmp_dir);
    TEST_CHECK(exporter.export_all(tmp_dir.string(), products));
    TEST_CHECK(writer_ptr->written_paths.size() == 3);
    TEST_CHECK(std::filesystem::exists(tmp_dir / "alerts.geojson"));
    TEST_CHECK(std::filesystem::exists(tmp_dir / "summary.json"));

    std::ifstream summary_in(tmp_dir / "summary.json");
    const std::string summary_text(
        (std::istreambuf_iterator<char>(summary_in)),
        std::istreambuf_iterator<char>());
    TEST_CHECK(summary_text.find("\"anomalies_detected\": 2") != std::string::npos);
    TEST_CHECK(summary_text.find("\"min_displacement_mm\": -42") != std::string::npos);
    TEST_CHECK(summary_text.find("\"max_displacement_mm\": 35") != std::string::npos);

    std::filesystem::remove_all(tmp_dir);
}

void test_gdal_cog_writer_and_slc_source() {
    insar::io::ensure_gdal_registered();
    const auto tmp_dir = std::filesystem::temp_directory_path() / "insar_cpp_gdal_test";
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);

    insar::GeoMetadata meta;
    meta.width = 32;
    meta.height = 32;
    meta.has_georef = true;
    meta.geotransform = {480000.0, 10.0, 0.0, 2155000.0, 0.0, -10.0};
    meta.projection_wkt = "EPSG:32614";

    std::vector<float> data(32 * 32, 1.25f);
    std::vector<uint8_t> mask(32 * 32, 1);
    mask[0] = 0;

    const std::string tif_path = (tmp_dir / "test_cog.tif").string();
    insar::io::GdalCogWriter writer;
    TEST_CHECK(writer.write(tif_path, data, mask, meta));
    TEST_CHECK(std::filesystem::exists(tif_path));

    // Also test GdalSlcSource with a complex GTiff pair
    GDALDriver* gtiff_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    TEST_CHECK(gtiff_drv != nullptr);
    const std::string c1_path = (tmp_dir / "slc1.tif").string();
    const std::string c2_path = (tmp_dir / "slc2.tif").string();
    std::vector<insar::Complex> slc_data(32 * 32, insar::Complex(1.0f, 0.5f));
    for (const auto& p : {c1_path, c2_path}) {
        insar::io::UniqueGdalDataset ds(
            gtiff_drv->Create(p.c_str(), 32, 32, 1, GDT_CFloat32, nullptr));
        TEST_CHECK(ds != nullptr);
        auto gt = meta.geotransform;
        ds->SetGeoTransform(gt.data());
        TEST_CHECK(ds->GetRasterBand(1)->RasterIO(
            GF_Write, 0, 0, 32, 32, slc_data.data(), 32, 32, GDT_CFloat32, 0, 0) == CE_None);
    }

    insar::GdalSlcSource source(c1_path, c2_path, "/data/VV", insar::CropWindow{2, 4, 16, 12});
    const insar::SlcPair pair = source.load();
    TEST_CHECK(pair.width() == 16);
    TEST_CHECK(pair.height() == 12);
    TEST_CHECK(std::abs(pair.reference.metadata().geotransform[0] - 480020.0) < 1e-9);
    TEST_CHECK(std::abs(pair.reference.metadata().geotransform[3] - 2154960.0) < 1e-9);

    std::filesystem::remove_all(tmp_dir);
}

void test_cli_parser() {
    const char* argv[] = {
        "insar_proto", "--width", "256", "--height", "128", "--window", "1", "2", "3", "4"
    };
    const auto opts = insar::app::CliParser::parse(10, const_cast<char**>(argv));
    TEST_CHECK(opts.synth_width == 256);
    TEST_CHECK(opts.synth_height == 128);
    TEST_CHECK(opts.window.x == 1 && opts.window.y == 2 && opts.window.width == 3 && opts.window.height == 4);
    TEST_CHECK(opts.is_synthetic());
    TEST_CHECK(opts.resolved_stride_x() == 1);
    TEST_CHECK(opts.resolved_stride_y() == 1);

    const char* bad_argv[] = {"insar_proto", "--unknown-flag"};
    bool threw = false;
    try {
        (void)insar::app::CliParser::parse(2, const_cast<char**>(bad_argv));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    TEST_CHECK(threw);
}

} // namespace

int main() {
    test_fft_round_trip();
    test_poisson_preconditioner_inverts_laplacian();
    test_weighted_laplacian_symmetry();
    test_geo_metadata_transforms();
    test_pcg_unwrapper_polymorphic_preconditioner();
    test_synthetic_engine_end_to_end();
    test_anomaly_detector_and_exporter();
    test_gdal_cog_writer_and_slc_source();
    test_cli_parser();

    std::printf("All insar C++ unit tests passed.\n");
    return 0;
}
