#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
#include "direction_strategy.hpp"
namespace furiaopt{


class ConstrainedSolver{

    //The same solver can be used for both NLP and LS problems.
    //At this scope use a type-erasure approach to decouple the solver from the specific problem type.
    CostFunc cost_func_;
    GradientFunc gradient_func_;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> get_approximate_hessian_func_;
    std::function<void(const Eigen::VectorXd&)> update_previous_gradient_func_;

    //Constrained
    EqualityConstraintFunc equality_constraint_func_;
    GradientEqualityConstraintFunc gradient_equality_constraint_func_;
    InequalityConstraintFunc inequality_constraint_func_;
    GradientInequalityConstraintFunc gradient_inequality_constraint_func_;

    Eigen::VectorXd x0_;

    std::reference_wrapper<const ConstrainedSolverOptions> options_;
    std::shared_ptr<spdlog::logger> logger_;

public:

    //Constructor that takes the solver options and the problem structure
    ConstrainedSolver(const ConstrainedSolverOptions& options, const NLPProblem& problem);
    ConstrainedSolver(const ConstrainedSolverOptions& options, const LSProblem& problem);

    //Solve call
    Result solve();
};

}
