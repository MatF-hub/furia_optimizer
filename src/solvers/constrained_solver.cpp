#include "solvers/constrained_solver.hpp"
#include "solvers/qp_solver.hpp"
#include "generalization_method.hpp"
#include "compute_gradient.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <iostream>

inline std::string vec_to_string(const Eigen::VectorXd& v)
{
    std::ostringstream oss;
    oss << v.transpose();
    return oss.str();
}
namespace furiaopt{

ConstrainedSolver::ConstrainedSolver(const ConstrainedSolverOptions& options, const NLPProblem& problem)
    : options_(std::cref(options)), x0_(problem.x0), logger_(options_.get().logger ? options_.get().logger : std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>())){

    if (!problem.hasEqualityConstraints() && !problem.hasInequalityConstraints()) {
        logger_->warn("No constraints provided for the problem, call unconstrained solver instead");
    }

    if (problem.hasEqualityConstraints()) {
        if (!problem.gradient_equality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of equality constraints must be provided");
        }
        equality_constraint_func_ = problem.equality_constraint_func.value();
        gradient_equality_constraint_func_ = problem.gradient_equality_constraint_func.value();
    }
    else
    {
        equality_constraint_func_ = [](const Eigen::VectorXd&) -> Eigen::VectorXd { return Eigen::VectorXd::Zero(0); };
        gradient_equality_constraint_func_ = [](const Eigen::VectorXd& x) -> Eigen::MatrixXd { return Eigen::MatrixXd::Zero(x.rows(), 0); };
    };

    if (problem.hasInequalityConstraints()) {
        if (!problem.gradient_inequality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of inequality constraints must be provided");
        }
        inequality_constraint_func_ = problem.inequality_constraint_func.value();
        gradient_inequality_constraint_func_ = problem.gradient_inequality_constraint_func.value();
    }
    else
    {
        inequality_constraint_func_ = [](const Eigen::VectorXd&) -> Eigen::VectorXd { return Eigen::VectorXd::Zero(0); };
        gradient_inequality_constraint_func_ = [](const Eigen::VectorXd& x) -> Eigen::MatrixXd { return Eigen::MatrixXd::Zero(x.rows(), 0); };
    };

    cost_func_ = [&problem](const Eigen::VectorXd& x) -> double { return problem.cost_func(x); };
    gradient_func_ = [&problem](const Eigen::VectorXd& x) -> Eigen::VectorXd { return compute_gradient(problem, x); };

    // Initialize the approximate Hessian strategy based on the selected method
    auto strategy = BFGSHessianApproximation(problem);
    get_approximate_hessian_func_ = [strategy](const Eigen::VectorXd& grad_lagrangian, const Eigen::VectorXd& x) mutable {
        return strategy.getApproximateHessian(grad_lagrangian, x);
    };

    update_previous_gradient_func_ = [strategy](const Eigen::VectorXd& g_k) mutable {
        return strategy.setPreviousGradient(g_k);
    };
};

ConstrainedSolver::ConstrainedSolver(const ConstrainedSolverOptions& options, const LSProblem& problem)
    : options_(std::cref(options)), x0_(problem.x0), logger_(options_.get().logger ? options_.get().logger : std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>())){

    if (!problem.hasEqualityConstraints() && !problem.hasInequalityConstraints()) {
        logger_->warn("No constraints provided for the problem, call unconstrained solver instead");
    }

    if (problem.hasEqualityConstraints()) {
        if (!problem.gradient_equality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of equality constraints must be provided");
        }
        equality_constraint_func_ = problem.equality_constraint_func.value();
        gradient_equality_constraint_func_ = problem.gradient_equality_constraint_func.value();
    }
    else
    {
        equality_constraint_func_ = [](const Eigen::VectorXd&) -> Eigen::VectorXd { return Eigen::VectorXd::Zero(0); };
        gradient_equality_constraint_func_ = [](const Eigen::VectorXd& x) -> Eigen::MatrixXd { return Eigen::MatrixXd::Zero(x.rows(), 0); };
    };
    

    if (problem.hasInequalityConstraints()) {
        if (!problem.gradient_inequality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of inequality constraints must be provided");
        }
        inequality_constraint_func_ = problem.inequality_constraint_func.value();
        gradient_inequality_constraint_func_ = problem.gradient_inequality_constraint_func.value();
    }
    else
    {
        inequality_constraint_func_ = [](const Eigen::VectorXd&) -> Eigen::VectorXd { return Eigen::VectorXd::Zero(0); };
        gradient_inequality_constraint_func_ = [](const Eigen::VectorXd& x) -> Eigen::MatrixXd { return Eigen::MatrixXd::Zero(x.rows(), 0); };
    };

    cost_func_ = [&problem](const Eigen::VectorXd& x) -> double {
        return problem.residual_func(x).transpose() * problem.residual_func(x);
    };
    gradient_func_ = [&problem](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        if (problem.gradient_residual_func.has_value()) {
            return 2 * problem.gradient_residual_func.value()(x)* problem.residual_func(x);
        } else {
            // Numerical approximation of the Jacobian matrix of the residual function
            throw std::invalid_argument("Gradient of residual function must be provided");
        }
    };

    // Initialize the approximate Hessian strategy for least squares problems, which is always Gauss-Newton
    auto strategy = GaussNewtonHessianApproximation(problem);
    get_approximate_hessian_func_ = [strategy](const Eigen::VectorXd&, const Eigen::VectorXd& x) mutable {
        return strategy.getApproximateHessian(x);
    };

    update_previous_gradient_func_ = [](const Eigen::VectorXd&) { /*No need to update previous gradient for Gauss-Newton approximation */};
};

Result ConstrainedSolver::solve(){

    //The problem min_wrt_x  f(x)
    //                   s.t g(x)=0
    //                       h(x)>0

    //has resulting kkt conditions, satisfied at the optimimum(x', lambda', mhu'):
    //grad_L(x', lambda', mhu') = grad_f(x') - grad_g(x')*lambda' - grad_h(x')*mhu' = 0
    //g(x')=0
    //h(x')>0
    //mhu'>=0
    //h(x')^T * mhu' = 0

    //The newton step for those kkt conditions is the same as following QP:
    //min_wrt_(p_k) grad_f(x_k)^T*p_k + 0.5*p_k^T*Hessian_L(x_k, lambda_k, mhu_k)*p_K
    //  s.t         grad_g(x_k)^T*p_k + g(x_k)=0
    //              grad_h(x_k)^T*p_k + h(x_k)>=0

    logger_->info("Starting solve");
    Result result;
    result.summary.initial_cost = cost_func_(x0_);

    int iter = 0;
    double Dx_i = std::numeric_limits<double>::infinity();
    double Df_i = std::numeric_limits<double>::infinity();

    const int num_equality_constraints = gradient_equality_constraint_func_ ? gradient_equality_constraint_func_(x0_).cols() : 0;
    const int num_inequality_constraints = gradient_inequality_constraint_func_ ? gradient_inequality_constraint_func_(x0_).cols() : 0;
    Eigen::VectorXd lambda_i = Eigen::VectorXd::Zero(num_equality_constraints);
    Eigen::VectorXd mhu_i = Eigen::VectorXd::Zero(num_inequality_constraints);
    Eigen::VectorXd sigma_j = Eigen::VectorXd::Ones(num_equality_constraints);
    Eigen::VectorXd tau_j = Eigen::VectorXd::Ones(num_inequality_constraints);
    Eigen::VectorXd x_i = x0_;

    while (iter < options_.get().max_iter) {
        Eigen::VectorXd grad_f = gradient_func_(x_i);
        const Eigen::VectorXd g_i = equality_constraint_func_(x_i);
        const Eigen::VectorXd h_i = inequality_constraint_func_(x_i);
        Eigen::MatrixXd grad_eq = gradient_equality_constraint_func_(x_i);
        Eigen::MatrixXd grad_ineq = gradient_inequality_constraint_func_(x_i);

        Eigen::VectorXd grad_lagrangian = grad_f - grad_eq*lambda_i - grad_ineq*mhu_i;

        if (Dx_i <= options_.get().step_tolerance) {
            result.summary.termination_reason = TerminationReason::StepTolerance;
            break;
        }
        if (Df_i <= options_.get().function_tolerance)
        {
            result.summary.termination_reason = TerminationReason::FunctionTolerance;
            break;
        }

        QPProblem Eq_qp_problem;
        if (h_i.size() == 0 || h_i.minCoeff() >= 0) {
            // p = 0 is a feasible point for the QP subproblem if it is strictly interior (each inequality constraints hold)
            Eq_qp_problem.x0 = Eigen::VectorXd::Zero(x_i.size()); 
        } // else leave unset to run phase 1 LP to find a feasible points
        Eq_qp_problem.c = grad_f;
        Eq_qp_problem.H = get_approximate_hessian_func_(grad_lagrangian, x_i);
        Eq_qp_problem.A = grad_eq.transpose();
        Eq_qp_problem.b = g_i;
        Eq_qp_problem.C = grad_ineq.transpose();
        Eq_qp_problem.d = h_i;

        QPSolver solver(options_.get().QP_subproblem_options, Eq_qp_problem);

        Result QP_result = solver.solve();

        Eigen::VectorXd p_i = QP_result.x;
        Eigen::VectorXd D_lambda_i = QP_result.lambda - lambda_i;
        Eigen::VectorXd D_mhu_i = QP_result.mhu - mhu_i;

        //Check constraint violation.
        const double eq_constraint_violation = g_i.size() > 0 ? g_i.lpNorm<Eigen::Infinity>() : 0.0;
        const double ineq_constraint_violation = h_i.size() > 0 ? std::max( - h_i.minCoeff() , 0.0) : 0.0;
        const bool feasible = eq_constraint_violation <= options_.get().constraint_tolerance 
                              && ineq_constraint_violation <= options_.get().constraint_tolerance;

        if ((grad_lagrangian.transpose()*p_i).norm() < options_.get().gradient_tolerance && feasible) {
            result.summary.converged = true;
            result.summary.termination_reason = TerminationReason::GradientTolerance;
            break;
        }

        //Update sigma_j and tau_j
        for (int i = 0; i < sigma_j.size(); i++)
        {
            double abs_lambda_i = std::abs(QP_result.lambda(i));
            sigma_j(i)=std::max(abs_lambda_i, (sigma_j(i)+abs_lambda_i)/2);
        }
        for (int i = 0; i < tau_j.size(); i++)
        {
            double abs_mhu_i = std::abs(QP_result.mhu(i));
            tau_j(i)=std::max(abs_mhu_i, (tau_j(i)+abs_mhu_i)/2);
        }

        //Line search with armijo stopping condition on L1-Norm merit function
        double step_length = compute_step_length(options_.get().globalization_method, 
                                                 cost_func_,
                                                 equality_constraint_func_, 
                                                 inequality_constraint_func_, 
                                                 gradient_func_, 
                                                 gradient_inequality_constraint_func_,
                                                 x_i,
                                                 p_i, 
                                                 sigma_j, 
                                                 tau_j);

        if (step_length == 0.0) {
            logger_->error("SQP line search failed at iteration {}: merit directional derivative >= 0", iter);
            result.summary.termination_reason = TerminationReason::StepTolerance;   // or a new LineSearchFailure
            break;   // and do NOT let the fallback relabel this as GradientTolerance
        }

        Eigen::VectorXd x_new = x_i + step_length * p_i;
        lambda_i = lambda_i + step_length * D_lambda_i;
        mhu_i = mhu_i + step_length * D_mhu_i;

        Dx_i = (x_new - x_i).norm()/std::max(x_i.norm(), 1e-16);
        double f_i = cost_func_(x_i);
        Df_i = std::abs(cost_func_(x_new) - f_i)/std::max(std::abs(f_i), 1e-16);

        x_i = x_new;

        //Before next iteration, we update the previous gradient in the BFGS approximnation 
        //to be consistent with the new multipliers that will be used in the next iteration.
        //Notice we use graf_f,h,g compute at the previous x_i, with new multipliers lambda_k_plus_1, mhu_k_plus_1
        update_previous_gradient_func_(grad_f - grad_eq*lambda_i - grad_ineq*mhu_i); //DLagrangian(x_k, lambda_k_plus_1, mhu_k_plus_1)

        iter++;
    }

    result.summary.iterations = iter;
    result.summary.final_cost = cost_func_(x_i);
    result.summary.final_gradient_norm = gradient_func_(x_i).norm();
    result.x = x_i;
    result.lambda = lambda_i;
    result.mhu = mhu_i;
    bool eq_constraint_satisfied = 
        equality_constraint_func_(x_i).size() == 0 ||
        equality_constraint_func_(x_i).lpNorm<Eigen::Infinity>() <= options_.get().constraint_tolerance;
    bool inequality_constraint_satisfied =
        inequality_constraint_func_(x_i).size() == 0 ||
        inequality_constraint_func_(x_i).minCoeff() >= - options_.get().constraint_tolerance;

    if (!result.summary.converged && iter >= options_.get().max_iter) {
        result.summary.termination_reason = TerminationReason::MaxIterations;
        if (eq_constraint_satisfied && inequality_constraint_satisfied) 
        {
            result.summary.converged = true;
        }
        else
        {
            result.summary.converged = false;
        }
    }            

    return result;
};

}