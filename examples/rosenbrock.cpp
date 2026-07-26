#include "solvers/unconstrained_solver.hpp"
#include <iostream>

// The Rosenbrock function itself
#include <Eigen/Dense>
#include <cmath>
#include <random>

// Config loader
#include "config_loader.hpp"


double rosenbrock(const Eigen::VectorXd& params, const Eigen::VectorXd& x) {
    double a = params[0];
    double b = params[1];
    return std::pow(a - x[0], 2) + b * std::pow(x[1] - std::pow(x[0], 2), 2);
}

Eigen::VectorXd rosenbrock_gradient(const Eigen::VectorXd& params, const Eigen::VectorXd& x) {
    double a = params[0];
    double b = params[1];
    Eigen::VectorXd grad(2);
    // Derivative with respect to x[0]
    grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - std::pow(x[0], 2));
    // Derivative with respect to x[1]
    grad[1] = 2 * b * (x[1] - std::pow(x[0], 2));
    return grad;
}

Eigen::MatrixXd rosenbrock_hessian(const Eigen::VectorXd& params, const Eigen::VectorXd& x) {
    double b = params[1];
    double x0 = x[0];
    double x1 = x[1];
    Eigen::MatrixXd hessian(2, 2);
    //ddf_dxdx
    hessian(0, 0) = 2 - 4 * b * (x1 - std::pow(x0, 2)) + 8 * b * std::pow(x0, 2);
    //ddf_dxdy
    hessian(0, 1) = -4 * b * x0;
    //ddf_dydx
    hessian(1, 0) = -4 * b * x0; // Symmetric
    //ddf_dydy
    hessian(1, 1) = 2 * b;
    return hessian;
}

int main()
{
    // Setup solver options
    furiaopt::UnconstrainedSolverOptions options = furiaopt::load_solver_options("config/config.json");

    // Constants for the Rosenbrock function A = 1.0, B = 100.0
    double A_param = 1.0;
    double B_param = 100.0;
    Eigen::VectorXd params(2);
    params << 1.0, 100.0;

    //Setup problem to solve
    furiaopt::NLPProblem problem;
    problem.cost_func = [params](const Eigen::VectorXd& x) -> double {
        return rosenbrock(params, x);
    };
    problem.gradient_func = [params](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        return rosenbrock_gradient(params, x);
    };
    problem.hessian_func = [params](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        return rosenbrock_hessian(params, x);
    };

    // Setup the random number generator for random initialziation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_x(-2.0, 2.0);
    std::uniform_real_distribution<double> dist_y(-1.0, 2.0);
    Eigen::VectorXd x_init(2);
    x_init << dist_x(gen), dist_y(gen);
    problem.x0 = x_init;

    //Initialize logger
    options.logger->info("Application started");

    //Initialize solver and solve the problem
    furiaopt::UnconstrainedSolver default_solver(options, problem);
    furiaopt::Result result = default_solver.solve();

    std::cout << "Optimized parameters: " << result.x.transpose() << std::endl;
    std::cout << "Solver summary: " << std::endl;
    std::cout << "  Iterations: " << result.summary.iterations << std::endl;
    std::cout << "  Initial cost: " << result.summary.initial_cost << std::endl;
    std::cout << "  Final cost: " << result.summary.final_cost << std::endl;
    std::cout << "  Converged: " << (result.summary.converged ? "Yes" : "No") << std::endl;
}