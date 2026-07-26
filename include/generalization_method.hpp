#pragma once
#include "types.hpp"
#include "solver_config.hpp"

#include <Eigen/Dense>

namespace furiaopt {

inline double compute_step_length(const GlobalizationMethod& globalization_method, 
                                  const CostFunc& f, const double& dir_deriv, 
                                  const Eigen::VectorXd& x, 
                                  const Eigen::VectorXd& direction)
{
    if (globalization_method == GlobalizationMethod::LineSearch)
    {
        // -----------------------------------------
        // Backtracking Line Search (Armijo)
        // -----------------------------------------
        const double beta   = 0.5;     // shrink factor (0,1)
        const double c1     = 1e-4;    // Armijo constant
        const int max_ls    = 25;

        double alpha = 1.0;

        // Current cost
        const double fx = f(x);

        // Safety check: direction should be descent
        if (dir_deriv >= 0.0)
        {
            return 0.0;
        }

        for (int i = 0; i < max_ls; ++i)
        {
            Eigen::VectorXd x_trial = x + alpha * direction;

            double f_trial = f(x_trial);

            // Armijo condition:
            // f(x + alpha p) <= f(x) + c1 alpha g^T p
            if (f_trial <= fx + c1 * alpha * dir_deriv)
            {
                return alpha;
            }

            alpha *= beta;
        }

        // fallback if line search fails
        return alpha;
    }
    else if (globalization_method == GlobalizationMethod::TrustRegion)
    {
        // Prototype placeholder
        throw std::runtime_error("TrustRegion not yet implemented");
    }
    else
    {
        throw std::invalid_argument("Unsupported globalization method");
    }
};

inline double compute_step_length(const GlobalizationMethod& globalization_method, 
                                  const CostFunc& f, 
                                  const EqualityConstraintFunc& g, 
                                  const InequalityConstraintFunc& h, 
                                  const GradientFunc& grad_f, 
                                  const GradientInequalityConstraintFunc& grad_h,
                                  const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& direction, 
                                  const Eigen::VectorXd& sigma, 
                                  const Eigen::VectorXd& tau)
{
    // Create l1-norm merit function 
    CostFunc l1_norm_merit = [&](const Eigen::VectorXd& x) -> double {
        Eigen::VectorXd abs_grad = g(x).cwiseAbs();
        Eigen::VectorXd result = (-h(x).array()).max(0.0);
        return f(x) + sigma.dot(abs_grad) + tau.dot(result);
    };

    Eigen::VectorXd abs_grad = g(x).cwiseAbs();
    double Directional_derivative_l1_norm_merit = grad_f(x).dot(direction) - sigma.dot(abs_grad);
    Eigen::VectorXd ineq_constraints = h(x);

    for (int i = 0; i < ineq_constraints.size(); i++)
    {
        if (ineq_constraints(i)<0)
        {
            Directional_derivative_l1_norm_merit  -= tau(i)*(grad_h(x).col(i)).dot(direction);
        };
    }

    
    return compute_step_length(globalization_method, l1_norm_merit, Directional_derivative_l1_norm_merit, x, direction);
};

}