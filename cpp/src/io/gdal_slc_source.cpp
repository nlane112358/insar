#include "insar/io/gdal_slc_source.hpp"

#include "insar/io/gdal_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace insar {

namespace {

bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    const size_t offset = s.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[offset + i])) !=
            std::tolower(static_cast<unsigned char>(suffix[i]))) {
            return false;
        }
    }
    return true;
}

io::UniqueGdalDataset open_slc_dataset(const std::string& path, const std::string& subdataset) {
    const bool is_already_prefixed =
        path.rfind("NETCDF:", 0) == 0 || path.rfind("HDF5:", 0) == 0;
    const bool is_hdf5 = !is_already_prefixed &&
                         (ends_with_ci(path, ".h5") || ends_with_ci(path, ".hdf5"));

    std::vector<std::string> candidates;
    if (is_hdf5) {
        // NETCDF driver parses CF-1.8 UTM projection & coordinates in OPERA CSLC-S1
        const std::string sub = (!subdataset.empty() && subdataset[0] == '/')
                                    ? subdataset
                                    : ("/" + subdataset);
        candidates.push_back("NETCDF:\"" + path + "\":" + sub);
        candidates.push_back("HDF5:\"" + path + "\":/" + sub);
    }
    candidates.push_back(path);

    for (const auto& uri : candidates) {
        io::UniqueGdalDataset ds;
        {
            io::CplQuietErrorScope quiet;
            ds.reset(static_cast<GDALDataset*>(
                GDALOpenEx(uri.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY,
                           nullptr, nullptr, nullptr)));
        }

        if (ds && ds->GetRasterCount() >= 1) {
            std::printf("Opened GDAL dataset: %s (%dx%d)\n",
                        uri.c_str(), ds->GetRasterXSize(), ds->GetRasterYSize());
            return ds;
        }
    }

    std::ostringstream oss;
    oss << "GDAL could not open raster '" << path << "' (subdataset '" << subdataset << "').";
    throw std::runtime_error(oss.str());
}

} // namespace

GdalSlcSource::GdalSlcSource(
    std::string slc1_path,
    std::string slc2_path,
    std::string subdataset,
    CropWindow window)
    : slc1_path_(std::move(slc1_path)),
      slc2_path_(std::move(slc2_path)),
      subdataset_(std::move(subdataset)),
      window_(window)
{
}

SlcRaster GdalSlcSource::read_single(const std::string& path) const {
    io::ensure_gdal_registered();
    auto ds = open_slc_dataset(path, subdataset_);

    const int full_w = ds->GetRasterXSize();
    const int full_h = ds->GetRasterYSize();

    if (window_.x < 0 || window_.y < 0 || window_.x >= full_w || window_.y >= full_h) {
        std::ostringstream oss;
        oss << "Window offset (" << window_.x << ", " << window_.y
            << ") outside raster (" << full_w << "x" << full_h << ").";
        throw std::runtime_error(oss.str());
    }

    const int win_w = (window_.width > 0)
        ? std::min(window_.width, full_w - window_.x)
        : (full_w - window_.x);
    const int win_h = (window_.height > 0)
        ? std::min(window_.height, full_h - window_.y)
        : (full_h - window_.y);

    GeoMetadata meta;
    meta.width = win_w;
    meta.height = win_h;
    if (ds->GetGeoTransform(meta.geotransform.data()) == CE_None) {
        meta.has_georef = true;
        meta = meta.cropped(window_.x, window_.y, win_w, win_h);
    }

    const char* wkt = ds->GetProjectionRef();
    if (wkt && wkt[0] != '\0') {
        meta.projection_wkt = wkt;
    }

    GDALRasterBand* band = ds->GetRasterBand(1);
    std::vector<Complex> pixels(static_cast<size_t>(win_w) * win_h);

    const CPLErr err = band->RasterIO(
        GF_Read, window_.x, window_.y, win_w, win_h,
        pixels.data(), win_w, win_h,
        GDT_CFloat32, 0, 0);

    if (err != CE_None) {
        std::ostringstream oss;
        oss << "GDAL RasterIO failed while reading '" << path << "'.";
        throw std::runtime_error(oss.str());
    }

    return SlcRaster(std::move(meta), std::move(pixels));
}

SlcPair GdalSlcSource::load() {
    SlcRaster ref = read_single(slc1_path_);
    SlcRaster sec = read_single(slc2_path_);
    SlcPair pair{std::move(ref), std::move(sec)};
    pair.validate();
    loaded_width_ = pair.width();
    loaded_height_ = pair.height();
    return pair;
}

std::string GdalSlcSource::describe(int threads) const {
    std::ostringstream oss;
    oss << "Loaded real SLC pair (" << loaded_width_ << "x" << loaded_height_
        << ") on " << threads << " OpenMP threads";
    return oss.str();
}

} // namespace insar
