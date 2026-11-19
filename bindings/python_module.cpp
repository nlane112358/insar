#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "insar/app/command_runner.hpp"
#include "insar/engine/interferogram_estimator.hpp"
#include "insar/stack/stack_application.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> own_numpy_array(std::vector<T>&& values, size_t height, size_t width) {
    auto* owner = new std::vector<T>(std::move(values));
    nb::capsule lifetime(owner, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        owner->data(), {height, width}, lifetime);
}

NB_MODULE(_insar_core, m) {
    m.doc() = "In-process Python access to libinsar pair and stack engines, including DLPack arrays";

    using ComplexArray = nb::ndarray<
        const insar::Complex,
        nb::ndim<2>,
        nb::c_contig,
        nb::device::cpu>;

    m.def(
        "estimate_interferogram",
        [](ComplexArray reference,
           ComplexArray secondary,
           int stride_x,
           int stride_y,
           int look_rx,
           int look_ry) {
            if (reference.shape(0) != secondary.shape(0) || reference.shape(1) != secondary.shape(1)) {
                throw std::invalid_argument("Reference and secondary SLC arrays must have identical shapes");
            }
            if (stride_x <= 0 || stride_y <= 0) {
                throw std::invalid_argument("stride_x and stride_y must be positive");
            }
            const int in_height = static_cast<int>(reference.shape(0));
            const int in_width = static_cast<int>(reference.shape(1));
            const int out_width = std::max(1, in_width / stride_x);
            const int out_height = std::max(1, in_height / stride_y);
            const size_t output_size = static_cast<size_t>(out_width) * static_cast<size_t>(out_height);
            std::vector<float> wrapped(output_size, 0.0f);
            std::vector<float> coherence(output_size, 0.0f);
            std::vector<uint8_t> valid_mask(output_size, 0);
            insar::InterferogramEstimator estimator;
            size_t valid_count = 0;
            {
                nb::gil_scoped_release release;
                valid_count = estimator.compute(
                    reference.data(),
                    secondary.data(),
                    in_width,
                    in_height,
                    out_width,
                    out_height,
                    stride_x,
                    stride_y,
                    look_rx,
                    look_ry,
                    wrapped,
                    coherence,
                    valid_mask);
            }
            return nb::make_tuple(
                own_numpy_array(std::move(wrapped), out_height, out_width),
                own_numpy_array(std::move(coherence), out_height, out_width),
                own_numpy_array(std::move(valid_mask), out_height, out_width),
                valid_count);
        },
        nb::arg("reference"),
        nb::arg("secondary"),
        nb::arg("stride_x") = 1,
        nb::arg("stride_y") = 1,
        nb::arg("look_rx") = 3,
        nb::arg("look_ry") = 3,
        "Estimate wrapped phase, coherence, and a valid-pixel mask directly from CPU NumPy/DLPack SLC arrays.");

    m.def(
        "run_pairwise",
        [](const std::vector<std::string>& arguments) {
            nb::gil_scoped_release release;
            return insar::app::run_pairwise_cli(arguments);
        },
        nb::arg("arguments"),
        "Run pairwise processing with the same options accepted by insar_proto.");

    m.def(
        "run_stack",
        [](const std::vector<std::string>& arguments) {
            nb::gil_scoped_release release;
            return insar::stack::run_cli(arguments);
        },
        nb::arg("arguments"),
        "Run stack processing with the same options accepted by insar_stack.");
}
