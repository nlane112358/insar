#pragma once

#include "insar/stack/stack_types.hpp"

#include <gdal_alg.h>
#include <gdalwarper.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace insar::stack {

std::vector<float> read_single_band_f32(const std::string& path, RasterMeta& meta_out);

bool grids_match(const RasterMeta& a, const RasterMeta& b);

void assert_compatible_grid(const RasterMeta& a, const RasterMeta& b, const char* label);

std::vector<float> read_raster_aligned_f32(
    const std::string& path,
    const RasterMeta& target_meta,
    GDALResampleAlg resample_alg = GRA_Bilinear);

void write_cog_f32(
    const std::filesystem::path& out_path,
    const RasterMeta& meta,
    const std::vector<float>& data,
    const char* description);

void read_state_raster(
    const std::filesystem::path& state_path,
    RasterMeta& meta,
    StackMetadata& smeta,
    std::vector<double>& state_bands);

void write_state_raster(
    const std::filesystem::path& state_path,
    const RasterMeta& meta,
    const StackMetadata& smeta,
    const std::vector<double>& state_bands);

std::vector<uint8_t> rasterize_vector_mask(const std::string& vector_path, const RasterMeta& meta);

std::vector<uint8_t> load_mask_any(const std::string& path, const RasterMeta& meta);

} // namespace insar::stack
