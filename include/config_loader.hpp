#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>

#include <nlohmann/json.hpp>

#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "solver_config.hpp"

namespace furiaopt
{

using json = nlohmann::json;

//------------------------------------------
// Enum parsing helpers
//------------------------------------------

inline DirectionMethod parse_direction_method(
    const std::string& s)
{
    if (s == "GradientDescent")
        return DirectionMethod::GradientDescent;

    if (s == "BFGS")
        return DirectionMethod::BFGS;

    if (s == "ExactNewton")
        return DirectionMethod::ExactNewton;

    throw std::runtime_error(
        "Unknown direction_method: " + s);
}

inline GlobalizationMethod parse_globalization_method(
    const std::string& s)
{
    if (s == "LineSearch")
        return GlobalizationMethod::LineSearch;

    if (s == "TrustRegion")
        return GlobalizationMethod::TrustRegion;

    throw std::runtime_error(
        "Unknown globalization_method: " + s);
}

inline std::shared_ptr<spdlog::logger> make_file_logger(
    const std::string& logs_dir, const std::string& log_file_name)
{
    if (logs_dir.empty()) {
        return std::make_shared<spdlog::logger>("null",
            std::make_shared<spdlog::sinks::null_sink_mt>());
    }
    auto logger = spdlog::basic_logger_mt<spdlog::synchronous_factory>(
        log_file_name, logs_dir + "/" + log_file_name, true);
    logger->set_level(spdlog::level::info);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    return logger;
}

inline std::shared_ptr<spdlog::logger> make_config_logger(
    const json& j, const std::string& logs_dir)
{
    return make_file_logger(logs_dir, j.value("log_file_name", std::string("solver_log.log")));
}

//------------------------------------------
// Load config file
//------------------------------------------

inline UnconstrainedSolverOptions load_solver_options(
    const std::string& filepath,
    const std::string& logs_dir = "")
{
    std::ifstream file(filepath);

    if (!file)
    {
        throw std::runtime_error(
            "Cannot open config file: " + filepath);
    }

    json j;
    file >> j;

    UnconstrainedSolverOptions opt;

    opt.max_iter =
        j.value("max_iter", 100);

    opt.gradient_tolerance =
        j.value("gradient_tolerance", 1e-8);

    opt.step_tolerance =
        j.value("step_tolerance", 1e-12);

    opt.function_tolerance =
        j.value("function_tolerance", 1e-12);

    opt.direction_method =
        parse_direction_method(
            j.value("direction_method",
                    "GradientDescent"));

    opt.globalization_method =
        parse_globalization_method(
            j.value("globalization_method",
                    "LineSearch"));

    opt.logger = make_config_logger(j, logs_dir);

    return opt;
}

inline IPMSolverOptions load_ipm_solver_options(
    const std::string& filepath,
    const std::string& logs_dir = "")
{
    std::ifstream file(filepath);

    if (!file)
    {
        throw std::runtime_error(
            "Cannot open config file: " + filepath);
    }

    json j;
    file >> j;

    IPMSolverOptions opt;

    opt.tau_initial =
        j.value("qp_tau_initial", 1.0);

    opt.tau_factor =
        j.value("qp_tau_factor", 0.2);

    opt.max_outer =
        j.value("qp_max_outer", 40);

    opt.max_inner =
        j.value("qp_max_inner", 50);

    opt.ipm_tol =
        j.value("qp_ipm_tol", 1e-8);

    opt.logger = make_config_logger(j, logs_dir);

    return opt;
}

inline ConstrainedSolverOptions load_constrained_solver_options(
    const std::string& filepath,
    const std::string& logs_dir = "")
{
    std::ifstream file(filepath);

    if (!file)
    {
        throw std::runtime_error(
            "Cannot open config file: " + filepath);
    }

    json j;
    file >> j;

    ConstrainedSolverOptions opt;

    // Load UnconstrainedSolverOptions part
    opt.max_iter =
        j.value("max_iter", 100);

    opt.gradient_tolerance =
        j.value("gradient_tolerance", 1e-8);

    opt.step_tolerance =
        j.value("step_tolerance", 1e-12);

    opt.function_tolerance =
        j.value("function_tolerance", 1e-12);

    opt.constraint_tolerance =
        j.value("constraint_tolerance", 1e-6);

    opt.direction_method =
        parse_direction_method(
            j.value("direction_method",
                    "GradientDescent"));

    opt.globalization_method =
        parse_globalization_method(
            j.value("globalization_method",
                    "LineSearch"));

    // Load IPMSolverOptions part (QP_subproblem_options)
    opt.QP_subproblem_options.tau_initial =
        j.value("qp_tau_initial", 1.0);

    opt.QP_subproblem_options.tau_factor =
        j.value("qp_tau_factor", 0.2);

    opt.QP_subproblem_options.max_outer =
        j.value("qp_max_outer", 40);

    opt.QP_subproblem_options.max_inner =
        j.value("qp_max_inner", 50);

    opt.QP_subproblem_options.ipm_tol =
        j.value("qp_ipm_tol", 1e-8);

    opt.logger = make_config_logger(j, logs_dir);

    return opt;
}

} // namespace furiaopt
