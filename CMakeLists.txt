cmake_minimum_required(VERSION 3.18)
project(insar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(INSAR_BUILD_TESTS "Build C++ unit tests" ON)
option(INSAR_BUILD_PYTHON "Build the nanobind Python extension" OFF)
option(INSAR_NATIVE_ARCH "Optimize for the build host CPU" ON)

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

add_library(libinsar SHARED
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
    src/app/command_runner.cpp
    src/app/insar_application.cpp
    src/stack/stack_application.cpp
)

add_library(insar_core ALIAS libinsar)
set_target_properties(libinsar PROPERTIES
    OUTPUT_NAME insar
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(libinsar
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(libinsar PUBLIC cxx_std_20)
target_link_libraries(libinsar PUBLIC GDAL::GDAL)

if(OpenMP_CXX_FOUND)
    target_link_libraries(libinsar PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(libinsar PUBLIC /O2 /arch:AVX2)
else()
    target_compile_options(libinsar PUBLIC -O3 -fno-math-errno -Wall -Wextra)
    if(INSAR_NATIVE_ARCH AND (APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)"))
        check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
        if(HAS_MCPU_NATIVE)
            target_compile_options(libinsar PUBLIC -mcpu=native)
        else()
            check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
            if(HAS_MARCH_NATIVE)
                target_compile_options(libinsar PUBLIC -march=native)
            endif()
        endif()
    elseif(INSAR_NATIVE_ARCH)
        check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
                target_compile_options(libinsar PUBLIC -march=native)
        else()
            check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
            if(HAS_MCPU_NATIVE)
                target_compile_options(libinsar PUBLIC -mcpu=native)
            endif()
        endif()
    endif()
endif()

add_executable(insar_proto src/main.cpp)
target_link_libraries(insar_proto PRIVATE libinsar)

add_executable(insar_stack src/stack_main.cpp)
target_link_libraries(insar_stack PRIVATE libinsar)

if(INSAR_BUILD_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE nanobind_query_status
    )
    if(NOT nanobind_query_status EQUAL 0)
        message(FATAL_ERROR "Unable to locate nanobind. Install it with: ${Python_EXECUTABLE} -m pip install nanobind")
    endif()
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_insar_core bindings/python_module.cpp)
    target_link_libraries(_insar_core PRIVATE libinsar)
    set_target_properties(_insar_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
    install(TARGETS _insar_core LIBRARY DESTINATION .)
endif()

if(INSAR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_insar_engine.cpp")
    enable_testing()
    add_executable(insar_cpp_tests tests/cpp/test_insar_engine.cpp)
    target_link_libraries(insar_cpp_tests PRIVATE libinsar)
    add_test(NAME insar_cpp_tests COMMAND insar_cpp_tests)
endif()

include(GNUInstallDirs)
install(TARGETS libinsar insar_proto insar_stack
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
