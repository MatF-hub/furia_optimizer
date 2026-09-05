#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"

namespace furiaopt{

// Phase-1 initializer: finds a point strictly feasible for
// A*x + b = 0, C*x + d >= 0, for use as x0 by LPSolver/QPSolver when
// none is provided. Throws if the feasible set is empty or has no interior.
Eigen::VectorXd computeFeasiblePoint(
    const Eigen::VectorXd& c,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& d,
    const IPMSolverOptions& options);

}
