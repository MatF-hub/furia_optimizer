#include "solvers/unconstrained_solver.hpp"
#include "config_loader.hpp"
#include "example_cli.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

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

int main(int argc, char** argv)
{
    auto args = furiaopt::parse_example_args(argc, argv);

    furiaopt::UnconstrainedSolverOptions options =
        furiaopt::load_solver_options(args.config_dir + "/config.json", args.logs_output_dir);

    Eigen::VectorXd params(2);
    params << 1.0, 100.0;

    furiaopt::NLPProblem problem;
    problem.cost_func     = [params](const Eigen::VectorXd& x) { return rosenbrock(params, x); };
    problem.gradient_func = [params](const Eigen::VectorXd& x) { return rosenbrock_gradient(params, x); };
    problem.hessian_func  = [params](const Eigen::VectorXd& x) { return rosenbrock_hessian(params, x); };

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist_x(-3.0, 3.0);
    std::uniform_real_distribution<double> dist_y(-2.0, 4.0);
    Eigen::VectorXd x0(2);
    x0 << dist_x(rng), dist_y(rng);
    problem.x0 = x0;
    std::cout << "x0=" << x0.transpose() << "\n";

    struct MethodCfg { std::string name; furiaopt::DirectionMethod method; };
    std::vector<MethodCfg> methods = {
        {"GradientDescent", furiaopt::DirectionMethod::GradientDescent},
        {"BFGS",            furiaopt::DirectionMethod::BFGS},
        {"ExactNewton",     furiaopt::DirectionMethod::ExactNewton},
    };

    for (const auto& m : methods) {
        furiaopt::UnconstrainedSolverOptions opts = options;
        opts.direction_method = m.method;
        opts.logger->info("=== METHOD {} ===", m.name);

        furiaopt::UnconstrainedSolver solver(opts, problem);
        auto t0 = std::chrono::steady_clock::now();
        furiaopt::Result result = solver.solve();
        double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        opts.logger->info("METHOD {} elapsed_ms={:.4f}", m.name, elapsed_ms);

        std::cout << m.name << ": x=" << result.x.transpose()
                  << " iters=" << result.summary.iterations
                  << " cost=" << result.summary.final_cost
                  << " elapsed_ms=" << elapsed_ms
                  << " converged=" << (result.summary.converged ? "yes" : "no") << "\n";
    }
    return 0;
}
