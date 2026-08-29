cmake_minimum_required(VERSION 3.24)

project(
    FuriaOptimizer
    VERSION 0.1.0
    DESCRIPTION "C++ nonlinear constrained optimization library"
    LANGUAGES CXX
)

# =====================================================
# Options
# =====================================================

option(FURIAOPTIMIZER_BUILD_TESTS "Build unit tests" OFF)
option(FURIAOPTIMIZER_BUILD_EXAMPLES "Build examples" ON)
option(FURIAOPTIMIZER_BUILD_BENCHMARKS "Build benchmarks" OFF)

# =====================================================
# Global settings
# =====================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================================================
# Dependencies
# =====================================================

find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# =====================================================
# Library
# =====================================================

add_library(furiaopt
    src/solvers/unconstrained_solver.cpp
    src/solvers/lp_solver.cpp
    src/solvers/qp_solver.cpp
    src/solvers/constrained_solver.cpp
    src/direction_strategy.cpp
    src/barrier_ipm.cpp
)

add_library(FuriaOptimizer::furiaopt ALIAS furiaopt)

target_include_directories(furiaopt
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(furiaopt
    PUBLIC
        Eigen3::Eigen
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

target_compile_features(furiaopt PUBLIC cxx_std_20)

target_compile_options(furiaopt
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# =====================================================
# Examples
# =====================================================

if(FURIAOPTIMIZER_BUILD_EXAMPLES)
    #Executable for the Rosenbrock NLP example
    add_executable(rosenbrock_example examples/rosenbrock.cpp)
    target_link_libraries(rosenbrock_example PRIVATE furiaopt spdlog::spdlog)

    #Executable for the Trajectory Smoothing QP example
    add_executable(qp_trajectory_example examples/qp_2d_trajectory_smoothing.cpp)
    target_link_libraries(qp_trajectory_example PRIVATE furiaopt spdlog::spdlog)
endif()

# =====================================================
# Benchmarks
# =====================================================

if(FURIAOPTIMIZER_BUILD_BENCHMARKS)

    add_executable(furiaopt_benchmarks
        #benchmarks/example_benchmark.cpp
    )

endif()

# =====================================================
# Tests
# =====================================================

if(FURIAOPTIMIZER_BUILD_TESTS)

    enable_testing()

    find_package(Catch2 CONFIG REQUIRED)

    add_executable(furiaopt_tests
        tests/test_furiaopt.cpp
    )

    target_link_libraries(furiaopt_tests
        PRIVATE
            FuriaOptimizer::furiaopt
            Catch2::Catch2WithMain
    )

    include(Catch)

    catch_discover_tests(furiaopt_tests)

endif()