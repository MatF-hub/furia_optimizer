#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
namespace furiaopt{


class LPSolver{

    CostFunc cost_func_;

    std::reference_wrapper<const IPMSolverOptions> options_;
    std::reference_wrapper<const LPProblem> problem_;
    std::shared_ptr<spdlog::logger> logger_;

public:

    //Constructor that takes the solver options and the problem structure
    LPSolver(const IPMSolverOptions& options, const LPProblem& problem);

    //Solve call
    Result solve();

private:
    void general_LP_solver(Result& result);
};

}