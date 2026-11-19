#include "insar/stack/stack_application.hpp"

#include "insar/io/gdal_utils.hpp"
#include "insar/stack/closure.hpp"
#include "insar/stack/corridors.hpp"
#include "insar/stack/normal_equations.hpp"
#include "insar/stack/nuisance.hpp"
#include "insar/stack/stack_io.hpp"
#include "insar/stack/stack_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using namespace insar::stack;

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Sequential O(1) InSAR stacking, DEM tropospheric correction, triplet closure, and corridor analytics.\n\n"
        << "Required:\n"
        << "  --state <stack_state.tif>     Persistent Float64 sufficient-statistics raster\n"
        << "  --disp <displacement_mm.tif>  Input unwrapped pair displacement GeoTIFF (mm)\n"
        << "  --coh <coherence.tif>         Input interferometric coherence GeoTIFF [0..1]\n"
        << "  --dt-days <double>            Acquisition baseline duration (days), OR\n"
        << "  --t0-days <d0> --t1-days <d1> Explicit epoch bounds relative to stack t=0 (days)\n"
        << "  --out-dir <dir>               Output directory for derived COGs & corridor JSONs\n\n"
        << "Reference & Ramp / Tropospheric Correction:\n"
        << "  --ramp-order <-1|0|1|2>       Robust Huber/Tukey IRLS ramp order (-1=off, default 1)\n"
        << "  --min-coh <float>             Minimum coherence for ramp fit & stack update (default 0.35)\n"
        << "  --dem <dem.tif>               DEM GeoTIFF (local/SRTM/Copernicus) for elevation-correlated tropo removal\n"
        << "  --ref-lonlat <lon> <lat>      Pin zero datum to known stable WGS84 reference region\n"
        << "  --ref-radius-m <double>       Radius in meters for --ref-lonlat (default 1000.0)\n"
        << "  --ref-px <x> <y> <r>          Pin zero datum to pixel disk (x, y, radius_px)\n"
        << "  --ref-region <path>           Vector polygon (GeoJSON/GPKG) of known stable reference region\n"
        << "  --deform-mask <path>          Vector polygon or raster mask of high-deformation zones to exclude from ramp fit\n"
        << "  --prior-vel <vel.tif>         Optional prior velocity raster for masking |v| >= --ramp-deform-vel-mm-yr\n"
        << "  --ramp-deform-vel-mm-yr <f>   Velocity threshold (mm/yr) to auto-mask deforming pixels before ramp fit (default 25.0)\n"
        << "  --ramp-mask-sigma <float>     Single-pair residual MAD multiplier for masking deformation before final fit (default 2.5)\n"
        << "  --ramp-mask-dilate-px <int>   Dilation radius (px) around masked deformation zones (default 2)\n\n"
        << "Triplet Phase Closure & Redundant Pairs:\n"
        << "  --closure-triplet <a> <b> <m> Check triplet closure (m='add_sub': d+da-db, m='sum_sub': da+db-d); repeatable\n"
        << "  --closure-thresh-mm <float>   Per-pixel |closure| threshold in mm to mask unwrapping errors (default 10.0)\n"
        << "  --closure-max-bad-frac <f>    Reject entire pair if bad coherent pixel fraction exceeds this (default 0.35)\n"
        << "  --closure-mode <m>            'consensus' (majority fail, default) or 'strict' (any triplet fail)\n"
        << "  --redundant-pair              Update WLS velocity/accel normal eqs without double-counting cumulative span\n"
        << "  --pair-weight-scale <double>  Weight multiplier for pair normal equations (default 1.0)\n\n"
        << "Corridors & Alert Maturity / SNR Gating:\n"
        << "  --corridors <path>            Vector file (GeoJSON/GPKG) of infrastructure corridors\n"
        << "  --buffer-m <double>           Metric corridor buffer radius in meters (default 100.0)\n"
        << "  --vel-alert-mm-yr <float>     |Velocity| alert threshold in mm/yr (default 15.0)\n"
        << "  --accel-alert-mm-yr2 <float>  |Acceleration| alert threshold in mm/yr^2 (default 25.0)\n"
        << "  --min-alert-coh <float>       Minimum temporal coherence for alerts (default 0.55)\n"
        << "  --min-alert-snr <float>       Minimum velocity SNR |v|/sigma_v for alerts (default 2.5)\n"
        << "  --min-alert-span-days <d>     Minimum stack span (days) before issuing critical alerts (default 60.0)\n"
        << "  --min-alert-passes <int>      Minimum valid stacked pairs at a pixel for alerts (0=auto)\n"
        << "  --no-export                   Update state raster only without exporting outputs\n"
        << "  --skip-cog                    Export corridor JSONs/summary without encoding COGs\n"
        << "  --help                        Show this help text\n";
}

StackOptions parse_args(int argc, char** argv) {
    StackOptions opt;
    auto need_val = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("Missing value for ") + flag);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--state") {
            opt.state_path = need_val(i, "--state");
        } else if (arg == "--disp") {
            opt.disp_path = need_val(i, "--disp");
        } else if (arg == "--coh") {
            opt.coh_path = need_val(i, "--coh");
        } else if (arg == "--out-dir") {
            opt.out_dir = need_val(i, "--out-dir");
        } else if (arg == "--corridors") {
            opt.corridors_path = need_val(i, "--corridors");
        } else if (arg == "--dem") {
            opt.dem_path = need_val(i, "--dem");
        } else if (arg == "--deform-mask") {
            opt.deform_mask_path = need_val(i, "--deform-mask");
        } else if (arg == "--prior-vel") {
            opt.prior_vel_path = need_val(i, "--prior-vel");
        } else if (arg == "--ref-region") {
            opt.ref_region_path = need_val(i, "--ref-region");
        } else if (arg == "--ref-lonlat") {
            opt.has_ref_lonlat = true;
            opt.ref_lon = std::stod(need_val(i, "--ref-lonlat lon"));
            opt.ref_lat = std::stod(need_val(i, "--ref-lonlat lat"));
        } else if (arg == "--ref-radius-m") {
            opt.ref_radius_m = std::stod(need_val(i, "--ref-radius-m"));
        } else if (arg == "--ref-px") {
            opt.has_ref_px = true;
            opt.ref_px = std::stoi(need_val(i, "--ref-px x"));
            opt.ref_py = std::stoi(need_val(i, "--ref-px y"));
            opt.ref_radius_px = std::stoi(need_val(i, "--ref-px r"));
        } else if (arg == "--dt-days") {
            opt.dt_days = std::stod(need_val(i, "--dt-days"));
        } else if (arg == "--t0-days") {
            opt.t0_days = std::stod(need_val(i, "--t0-days"));
        } else if (arg == "--t1-days") {
            opt.t1_days = std::stod(need_val(i, "--t1-days"));
        } else if (arg == "--ramp-order") {
            opt.ramp_order = std::stoi(need_val(i, "--ramp-order"));
        } else if (arg == "--min-coh") {
            opt.min_coh = std::stof(need_val(i, "--min-coh"));
        } else if (arg == "--ramp-deform-vel-mm-yr") {
            opt.ramp_deform_vel_mm_yr = std::stof(need_val(i, "--ramp-deform-vel-mm-yr"));
        } else if (arg == "--ramp-mask-sigma") {
            opt.ramp_mask_sigma = std::stof(need_val(i, "--ramp-mask-sigma"));
        } else if (arg == "--ramp-mask-dilate-px") {
            opt.ramp_mask_dilate_px = std::stoi(need_val(i, "--ramp-mask-dilate-px"));
        } else if (arg == "--closure-triplet") {
            ClosureTripletSpec spec;
            spec.disp_a_path = need_val(i, "--closure-triplet disp_a");
            spec.disp_b_path = need_val(i, "--closure-triplet disp_b");
            spec.mode = need_val(i, "--closure-triplet mode");
            if (spec.mode != "add_sub" && spec.mode != "sum_sub") {
                throw std::runtime_error("--closure-triplet mode must be 'add_sub' or 'sum_sub'");
            }
            opt.closure_triplets.push_back(std::move(spec));
        } else if (arg == "--closure-thresh-mm") {
            opt.closure_thresh_mm = std::stof(need_val(i, "--closure-thresh-mm"));
        } else if (arg == "--closure-max-bad-frac") {
            opt.closure_max_bad_frac = std::stof(need_val(i, "--closure-max-bad-frac"));
        } else if (arg == "--closure-mode") {
            opt.closure_mode = need_val(i, "--closure-mode");
            if (opt.closure_mode != "consensus" && opt.closure_mode != "strict") {
                throw std::runtime_error("--closure-mode must be 'consensus' or 'strict'");
            }
        } else if (arg == "--redundant-pair") {
            opt.redundant_pair = true;
        } else if (arg == "--pair-weight-scale") {
            opt.pair_weight_scale = std::stod(need_val(i, "--pair-weight-scale"));
        } else if (arg == "--buffer-m") {
            opt.buffer_m = std::stod(need_val(i, "--buffer-m"));
        } else if (arg == "--vel-alert-mm-yr") {
            opt.vel_alert_mm_yr = std::stof(need_val(i, "--vel-alert-mm-yr"));
        } else if (arg == "--accel-alert-mm-yr2") {
            opt.accel_alert_mm_yr2 = std::stof(need_val(i, "--accel-alert-mm-yr2"));
        } else if (arg == "--min-alert-coh") {
            opt.min_alert_coh = std::stof(need_val(i, "--min-alert-coh"));
        } else if (arg == "--min-alert-snr") {
            opt.min_alert_snr = std::stof(need_val(i, "--min-alert-snr"));
        } else if (arg == "--min-alert-span-days") {
            opt.min_alert_span_days = std::stod(need_val(i, "--min-alert-span-days"));
        } else if (arg == "--min-alert-passes") {
            opt.min_alert_passes = std::stoi(need_val(i, "--min-alert-passes"));
        } else if (arg == "--no-export") {
            opt.no_export = true;
        } else if (arg == "--skip-cog") {
            opt.skip_cog = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opt.state_path.empty()) {
        throw std::runtime_error("--state is required");
    }
    if (!opt.no_export && opt.out_dir.empty()) {
        throw std::runtime_error("--out-dir is required unless --no-export is set");
    }
    bool have_disp = !opt.disp_path.empty();
    bool have_coh = !opt.coh_path.empty();
    if (have_disp != have_coh) {
        throw std::runtime_error("--disp and --coh must both be provided");
    }
    if (have_disp) {
        if (std::isfinite(opt.t0_days) && std::isfinite(opt.t1_days)) {
            if (opt.t1_days <= opt.t0_days) {
                throw std::runtime_error("--t1-days must exceed --t0-days");
            }
            opt.dt_days = opt.t1_days - opt.t0_days;
        } else if (!(opt.dt_days > 0.0)) {
            throw std::runtime_error("Provide --dt-days > 0 or valid --t0-days and --t1-days");
        }
    }
    if (opt.ramp_order < -1 || opt.ramp_order > 2) {
        throw std::runtime_error("--ramp-order must be -1, 0, 1, or 2");
    }
    if (opt.has_ref_lonlat && opt.ref_radius_m <= 0.0) {
        throw std::runtime_error("--ref-radius-m must be positive");
    }
    if (opt.has_ref_px && opt.ref_radius_px <= 0) {
        throw std::runtime_error("--ref-px radius must be positive");
    }
    if (opt.pair_weight_scale <= 0.0) {
        throw std::runtime_error("--pair-weight-scale must be positive");
    }
    if (opt.closure_thresh_mm <= 0.0f) {
        throw std::runtime_error("--closure-thresh-mm must be positive");
    }
    return opt;
}

} // namespace

namespace insar::stack {

int run_cli(int argc, char** argv) {
    try {
        StackOptions opt = parse_args(argc, argv);
        insar::io::ensure_gdal_registered();

        RasterMeta meta;
        StackMetadata smeta;
        std::vector<double> state_bands;
        const bool have_existing = fs::exists(opt.state_path);
        if (have_existing) {
            read_state_raster(opt.state_path, meta, smeta, state_bands);
        }

        RampResult ramp;
        ramp.order = opt.ramp_order;
        ReferencePinResult ref_pin;
        ClosureResult closure;
        std::vector<float> ramp_disp;
        std::vector<float> closure_res_map;

        if (!opt.disp_path.empty()) {
            RasterMeta disp_meta, coh_meta;
            ramp_disp = read_single_band_f32(opt.disp_path, disp_meta);
            std::vector<float> coh = read_single_band_f32(opt.coh_path, coh_meta);
            assert_compatible_grid(disp_meta, coh_meta, "coherence");

            const size_t npix = static_cast<size_t>(disp_meta.width) * static_cast<size_t>(disp_meta.height);
            if (!have_existing) {
                meta = disp_meta;
                state_bands.assign(static_cast<size_t>(kNumStateBands) * npix, 0.0);
                std::fill(
                    state_bands.data() + static_cast<size_t>(kBandLatestDisp) * npix,
                    state_bands.data() + static_cast<size_t>(kNumStateBands) * npix,
                    kNanD);
            } else {
                assert_compatible_grid(meta, disp_meta, "displacement");
            }

            double t0 = std::isfinite(opt.t0_days)
                ? opt.t0_days
                : (have_existing ? smeta.last_t1_days : 0.0);
            double t1 = std::isfinite(opt.t1_days) ? opt.t1_days : (t0 + opt.dt_days);

            std::vector<uint8_t> ref_mask = build_reference_mask(opt, meta, ref_pin);

            // 1. Triplet phase closure screening BEFORE ramp fitting or stack state update
            std::vector<uint8_t> closure_bad;
            closure = evaluate_triplet_closure(opt, meta, ramp_disp, coh, ref_mask, closure_bad, closure_res_map);

            if (closure.pair_rejected) {
                smeta.rejected_pair_count += 1;
                if (have_existing) {
                    write_state_raster(opt.state_path, meta, smeta, state_bands);
                }
                std::cout << "insar_stack REJECTED pair (closure bad_frac="
                          << std::fixed << std::setprecision(3) << closure.bad_fraction
                          << " > " << closure.max_bad_frac
                          << ", rms=" << std::setprecision(2) << closure.rms_mm << "mm)\n";
            } else {
                if (closure.masked_pixels > 0) {
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < npix; ++i) {
                        if (closure_bad[i]) {
                            ramp_disp[i] = kNanF;
                        }
                    }
                }

                // 2. Load optional DEM aligned to stack grid
                std::vector<float> dem;
                if (!opt.dem_path.empty()) {
                    dem = read_raster_aligned_f32(opt.dem_path, meta, GRA_Bilinear);
                }

                // 3. Extract coherent connected component around reference region & build deformation exclusion mask
                std::vector<uint8_t> ref_cc = extract_reference_connected_component(
                    meta.width, meta.height, ramp_disp, coh, ref_mask, opt.min_coh);
                std::vector<uint8_t> deform_excl = build_deform_exclusion_mask(
                    opt, meta, smeta, state_bands, ref_mask, ref_cc, ramp_disp, coh);

                // 4. Joint robust ramp + DEM tropospheric correction on masked background
                ramp = remove_robust_ramp_and_dem(
                    meta.width, meta.height, ramp_disp, coh, dem, deform_excl, ref_mask, ref_cc,
                    opt.ramp_order, opt.min_coh, opt.ramp_mask_sigma, opt.ramp_mask_dilate_px);

                // 5. Pin corrected displacement to known stable reference region
                apply_reference_pinning(ramp_disp, coh, ref_mask, opt.min_coh, ref_pin);

                // 6. Incremental O(1) sufficient-statistics state update
                const bool is_first_any = (smeta.pair_count == 0 && smeta.redundant_pair_count == 0);
                if (is_first_any) {
                    smeta.origin_days = t0;
                    smeta.first_t0_days = t0;
                    smeta.last_t1_days = t1;
                } else {
                    smeta.first_t0_days = std::min(smeta.first_t0_days, t0);
                    smeta.last_t1_days = std::max(smeta.last_t1_days, t1);
                }

                update_state_incremental(
                    state_bands, npix, ramp_disp, coh, t0, t1, smeta.origin_days, opt.min_coh,
                    opt.redundant_pair, opt.pair_weight_scale);

                if (opt.redundant_pair) {
                    smeta.redundant_pair_count += 1;
                } else {
                    smeta.sum_dt_days += (t1 - t0);
                    smeta.pair_count += 1;
                }

                write_state_raster(opt.state_path, meta, smeta, state_bands);
            }
        } else if (!have_existing) {
            throw std::runtime_error("Cannot export without either --disp/--coh or existing --state");
        }

        const size_t npix = static_cast<size_t>(meta.width) * static_cast<size_t>(meta.height);
        const double calendar_span_days = std::max(0.0, smeta.last_t1_days - smeta.first_t0_days);
        const int total_pairs_any = smeta.pair_count + smeta.redundant_pair_count;
        const bool stack_mature = (calendar_span_days >= opt.min_alert_span_days) &&
                                  (total_pairs_any >= 2 || opt.min_alert_span_days <= 0.0);

        if (!opt.no_export) {
            fs::path out_dir(opt.out_dir);
            fs::create_directories(out_dir);

            DerivedFields df = derive_products(state_bands, npix, smeta, opt.min_coh);
            if (!opt.skip_cog) {
                write_cog_f32(out_dir / "velocity_mm_yr.tif", meta, df.vel, "LOS Velocity (mm/yr)");
                write_cog_f32(out_dir / "acceleration_mm_yr2.tif", meta, df.accel, "LOS Acceleration (mm/yr^2)");
                write_cog_f32(out_dir / "cumulative_disp_mm.tif", meta, df.cum_disp, "Cumulative LOS Displacement (mm)");
                write_cog_f32(out_dir / "temporal_coherence.tif", meta, df.temp_coh, "Mean Temporal Coherence");
                write_cog_f32(out_dir / "velocity_uncertainty_mm_yr.tif", meta, df.unc, "Velocity Std Error (mm/yr)");
                write_cog_f32(out_dir / "velocity_snr.tif", meta, df.snr, "Velocity Signal-to-Noise Ratio |v|/sigma_v");
                if (opt.export_ramp_disp && !ramp_disp.empty() && !closure.pair_rejected) {
                    write_cog_f32(out_dir / "ramp_removed_disp_mm.tif", meta, ramp_disp, "Latest Pair Corrected Displacement (mm)");
                }
                if (closure.triplets_evaluated > 0 && !closure_res_map.empty()) {
                    write_cog_f32(out_dir / "closure_residual_mm.tif", meta, closure_res_map, "Triplet Phase Closure RMS Residual (mm)");
                }
            }

            std::vector<CorridorMetric> metrics;
            std::vector<AlertPoint> alerts;
            size_t total_snr_gated = 0;
            process_corridors(opt, meta, smeta, df, stack_mature, metrics, alerts, total_snr_gated);
            write_corridor_outputs(
                out_dir, opt, meta, smeta, ramp, ref_pin, closure, df,
                stack_mature, calendar_span_days, total_snr_gated, metrics, alerts);
        }

        if (!closure.pair_rejected) {
            std::cout << "insar_stack updated: pairs=" << smeta.pair_count;
            if (smeta.redundant_pair_count > 0) {
                std::cout << " (+" << smeta.redundant_pair_count << " redundant)";
            }
            std::cout << " span_days=" << std::fixed << std::setprecision(1) << calendar_span_days;
            if (ramp.applied) {
                std::cout << " ramp_rms=" << std::setprecision(2) << ramp.rms_before_mm
                          << "->" << ramp.rms_after_mm << "mm";
            }
            if (ramp.dem_applied) {
                std::cout << " dem_slope=" << std::setprecision(2) << ramp.dem_slope_mm_per_km << "mm/km";
            }
            if (ref_pin.applied) {
                std::cout << " ref_pin=" << std::setprecision(2) << ref_pin.offset_mm
                          << "mm(" << ref_pin.ref_pixels_used << "px)";
            }
            if (closure.triplets_evaluated > 0) {
                std::cout << " closure_masked=" << closure.masked_pixels
                          << "(rms=" << std::setprecision(2) << closure.rms_mm << "mm)";
            }
            std::cout << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}

int run_cli(const std::vector<std::string>& arguments) {
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.emplace_back("insar_stack");
    storage.insert(storage.end(), arguments.begin(), arguments.end());

    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& value : storage) {
        argv.push_back(value.data());
    }
    return run_cli(static_cast<int>(argv.size()), argv.data());
}

} // namespace insar::stack
