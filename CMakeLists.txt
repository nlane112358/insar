cmake_minimum_required(VERSION 3.18)
project(insar_cpu_proto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(INSAR_BUILD_TESTS "Build C++ unit tests" ON)

include(CheckCXXCompilerFlag)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    foreach(_omp_prefix IN ITEMS /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS "${_omp_prefix}/include/omp.h")
            list(APPEND CMAKE_PREFIX_PATH "${_omp_prefix}")
            break()
        endif()
    endforeach()
endif()

find_package(OpenMP)
find_package(GDAL REQUIRED)

add_library(insar_core STATIC
    src/core/types.cpp
    src/math/fft.cpp
    src/engine/preconditioner.cpp
    src/engine/interferogram_estimator.cpp
    src/engine/weighted_laplacian.cpp
    src/engine/phase_unwrapper.cpp
    src/engine/phase_calibrator.cpp
    src/engine/cpu_insar_engine.cpp
    src/io/gdal_utils.cpp
    src/io/gdal_slc_source.cpp
    src/io/cog_writer.cpp
    src/io/product_exporter.cpp
    src/analytics/anomaly_detector.cpp
    src/app/cli_options.cpp
    src/app/insar_application.cpp
)

target_include_directories(insar_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(insar_core PUBLIC cxx_std_20)
target_link_libraries(insar_core PUBLIC GDAL::GDAL)

if(OpenMP_CXX_FOUND)
    target_link_libraries(insar_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(insar_core PUBLIC /O2 /arch:AVX2)
else()
    target_compile_options(insar_core PUBLIC -O3 -fno-math-errno -Wall -Wextra)
    if(APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)")
        check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
        if(HAS_MCPU_NATIVE)
            target_compile_options(insar_core PUBLIC -mcpu=native)
        else()
            check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
            if(HAS_MARCH_NATIVE)
                target_compile_options(insar_core PUBLIC -march=native)
            endif()
        endif()
    else()
        check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            target_compile_options(insar_core PUBLIC -march=native)
        else()
            check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
            if(HAS_MCPU_NATIVE)
                target_compile_options(insar_core PUBLIC -mcpu=native)
            endif()
        endif()
    endif()
endif()

add_executable(insar_proto src/main.cpp)
target_link_libraries(insar_proto PRIVATE insar_core)

add_executable(insar_stack src/insar_stack_cpu.cpp)
target_link_libraries(insar_stack PRIVATE insar_core)

if(INSAR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_insar_engine.cpp")
    enable_testing()
    add_executable(insar_cpp_tests tests/cpp/test_insar_engine.cpp)
    target_link_libraries(insar_cpp_tests PRIVATE insar_core)
    add_test(NAME insar_cpp_tests COMMAND insar_cpp_tests)
endif()
