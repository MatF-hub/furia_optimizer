#pragma once

#include "solver_config.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace furiaopt::details {

    void solve_ipm_problem(const IPMProblem& problem, 
                           const IPMSolverOptions& options, 
                           const std::shared_ptr<spdlog::logger>& logger,
                           const char* who,
                           Result& result);

}