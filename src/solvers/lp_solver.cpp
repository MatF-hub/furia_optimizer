#include "solvers/lp_solver.hpp"
#include "solvers/feasibility.hpp"
#include "barrier_ipm.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "generalization_method.hpp"
namespace furiaopt{

LPSolver::LPSolver(const IPMSolverOptions& options, const LPProblem& problem) : options_(std::cref(options)), problem_(std::cref(problem)), logger_(options_.get().logger ? options_.get().logger : std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>())) {

    cost_func_ = [&problem](const Eigen::VectorXd& x) -> double {
        return problem.c.transpose() * x;
    };

};

Result LPSolver::solve(){

    logger_->info("Starting solve");
    Result result;

    if (problem_.get().hasInequalityConstraints() || problem_.get().hasEqualityConstraints())
    {
        general_LP_solver(result);
    }
    else
    {
        logger_->error("No constraints LP solver does not make sense");
        throw std::runtime_error("No constraints LP solver does not make sense");
    }

    return result;
};

void LPSolver::general_LP_solver(Result& result)
{   
    //The LP problem is solved using the interior point method.
    //The problem is reformulated using barrier functions:
    //min c^T*x - tau*sum_i(log(C*x_i + d_i))
    //s.t Ax + b = 0

    //KKT conditions for this problem reads as follows:
    //c - A^T*lambda - tau*sum_i(C_i/(C*x_i + d_i)) = 0
    //Ax + b = 0

    //Usign Newton's method to solve the KKT conditions, we get the following update equations:
    // [ tau*C^T*(diag(C*x + d)^-2)*C | -A^T ; * [ dx     ;   = [ -c + A^T*lamnda + tau*C^T*(diag(C*x + d)^-1)*Eigen::VectorXd::Ones(c.rows());
    //   A                            |  0   ]    dlambda ]       -Ax - b                                                                     ]

    // Extract problem data safely
    const Eigen::VectorXd& c = problem_.get().c;
    const Eigen::MatrixXd& A = problem_.get().A.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& b = problem_.get().b.value_or(Eigen::VectorXd::Zero(0));
    const Eigen::MatrixXd& C = problem_.get().C.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& d = problem_.get().d.value_or(Eigen::VectorXd::Zero(0));

    // Initialize variables
    Eigen::VectorXd x;

    if (problem_.get().x0.has_value())
    {
        x = problem_.get().x0.value();
    }
    else
    {
        x = computeFeasiblePoint(c, A, b, C, d, options_.get());
    }

    // Interior Point Method using Barrier Function
    IPMProblem ipm_problem;
    ipm_problem.x0 = x;
    ipm_problem.c = c;
    ipm_problem.A = A;
    ipm_problem.b = b;
    ipm_problem.C = C;
    ipm_problem.d = d;
    furiaopt::details::solve_ipm_problem(ipm_problem, options_.get(), logger_, "LP", result);

};

}