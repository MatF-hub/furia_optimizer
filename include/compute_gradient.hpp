#pragma once

#include <Eigen/Dense> // For Eigen::VectorXd and Eigen::MatrixXd
#include <optional>
#include "solver_config.hpp"

namespace furiaopt
{

    inline Eigen::VectorXd compute_gradient(const NLPProblem& problem, const Eigen::VectorXd& x)
    {
        if (problem.hasGradient()){
            return problem.gradient_func.value()(x);
        }
        else
        {
            //Finite difference approximation path for the gradient
        }
        throw std::runtime_error("Unable to compute gradient: no analytic gradient provided and finite difference approximation not implemented");
    }

    inline Eigen::VectorXd compute_gradient(const LSProblem& problem, const Eigen::VectorXd& x)
    {
        if (problem.hasGradientResidualFunc()){
            Eigen::MatrixXd Grad_F = problem.gradient_residual_func.value()(x);
            return 2 * Grad_F * problem.residual_func(x);
        }
        else
        {
            //Finite difference approximation path for the gradient
        }
        throw std::runtime_error("Unable to compute gradient: no analytic gradient provided and finite difference approximation not implemented");
    }
    
}