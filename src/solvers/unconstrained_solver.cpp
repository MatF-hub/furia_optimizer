#include "solvers/unconstrained_solver.hpp"
#include "generalization_method.hpp"
#include "compute_gradient.hpp"
#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace furiaopt{

UnconstrainedSolver::UnconstrainedSolver(const UnconstrainedSolverOptions& options, const NLPProblem& problem)
    : options_(std::cref(options)), x0_(problem.x0), logger_(options_.get().logger ? options_.get().logger : std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>())) {

    if (problem.hasEqualityConstraints() || problem.hasInequalityConstraints()) {
        logger_->warn("Constraints are present in the problem, please use the ConstrainedSolver instead of UnconstrainedSolver");
    }

    cost_func_ = [&problem](const Eigen::VectorXd& x) -> double { return problem.cost_func(x); };
    gradient_func_ = [&problem](const Eigen::VectorXd& x) -> Eigen::VectorXd { return compute_gradient(problem, x); };

    // Initialize the direction strategy based on the selected method
    switch (options_.get().direction_method) {
        case DirectionMethod::GradientDescent:
        {
            auto strategy = GradientDescentDirection(problem);
            get_direction_func_ = [strategy](const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i) mutable {
                return strategy.getDirection(gradient, x_i);
            };
            break;
        }
        case DirectionMethod::BFGS:
        {
            auto strategy = BFGSDirection(problem);
            get_direction_func_ = [strategy](const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i) mutable {
                return strategy.getDirection(gradient, x_i);
            };
            break;
        }
        case DirectionMethod::ExactNewton:
        {
            logger_->info("Exact Newton direction selected: notice that for this method to converge the Hessian should be PD or at least SPD");
            auto strategy = ExactHessianDirection(problem);
            get_direction_func_ = [strategy](const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i) mutable {
                return strategy.getDirection(gradient, x_i);
            };
            break;
        }
        default:
            throw std::invalid_argument("Unsupported direction method");
    }
};

UnconstrainedSolver::UnconstrainedSolver(const UnconstrainedSolverOptions& options, const LSProblem& problem)
    : options_(std::cref(options)), x0_(problem.x0), logger_(options_.get().logger ? options_.get().logger : std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>())) {

    if (problem.hasEqualityConstraints() || problem.hasInequalityConstraints()) {
        logger_->warn("Constraints are present in the problem, please use the ConstrainedSolver instead of UnconstrainedSolver");
    }
    cost_func_ = [&problem](const Eigen::VectorXd& x) -> double {
        return problem.residual_func(x).transpose() * problem.residual_func(x);
    };
    gradient_func_ = [&problem](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        if (problem.gradient_residual_func.has_value()) {
            return 2 * problem.gradient_residual_func.value()(x) * problem.residual_func(x);
        } else {
            // Numerical approximation of the Jacobian matrix of the residual function
            throw std::invalid_argument("Gradient of residual function must be provided");
        }
    };

    // Initialize the direction strategy for least squares problems, which is always Gauss-Newton
    auto strategy = GaussNewtonDirection(problem);
    get_direction_func_ = [strategy](const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i) mutable {
        return strategy.getDirection(gradient, x_i);
    };
};

Result UnconstrainedSolver::solve(){

    logger_->info("Starting solve");
    Result result;
    result.summary.initial_cost = cost_func_(x0_);

    int iter = 0;
    double Dx_i = std::numeric_limits<double>::infinity();
    double Df_i = std::numeric_limits<double>::infinity();
    Eigen::VectorXd x_i = x0_   ;

    while (iter < options_.get().max_iter) {

        Eigen::VectorXd g_i = gradient_func_(x_i);

        double f_i = cost_func_(x_i);

        logger_->info(
            "iter={},cost={:.8f},grad_norm={:.3e},x={}",
            iter,
            f_i,
            g_i.norm(),
            furiaopt::utils::vec_to_string(x_i)
        );

        if (Dx_i <= options_.get().step_tolerance) {
            result.summary.termination_reason = TerminationReason::StepTolerance;
            break;
        }
        if (Df_i <= options_.get().function_tolerance)
        {
            result.summary.termination_reason = TerminationReason::FunctionTolerance;
            break;
        }
                
        auto direction = get_direction_func_(g_i, x_i);
        if ((g_i.transpose() * direction).norm() <= options_.get().gradient_tolerance)
        {
            result.summary.termination_reason = TerminationReason::GradientTolerance;
            break;
        }

        const double directional_derivative = g_i.dot(direction);
        
        auto step_length = compute_step_length(options_.get().globalization_method, cost_func_, directional_derivative, x_i, direction);

        Eigen::VectorXd x_new = x_i + step_length * direction;

        Dx_i = (x_new - x_i).norm()/std::max(x_i.norm(), 1e-16);
        Df_i = std::abs(cost_func_(x_new) - f_i)/std::max(std::abs(f_i), 1e-16);

        x_i = x_new;
        iter++;
    }

    result.x = x_i;
    result.summary.iterations = iter;
    result.summary.final_cost = cost_func_(x_i);
    result.summary.final_gradient_norm = gradient_func_(x_i).norm();
    result.summary.converged = iter < options_.get().max_iter;
    if (!result.summary.converged) {
        result.summary.termination_reason = TerminationReason::MaxIterations;
    }

    return result;
};

}