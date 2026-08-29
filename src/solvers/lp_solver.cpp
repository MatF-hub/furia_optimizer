#include "solvers/lp_solver.hpp"
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

Eigen::VectorXd LPSolver::computeFeasiblePoint(
    const Eigen::VectorXd& c,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& d,
    const IPMSolverOptions& options)
{
    // If initialization is not provided use auxiliary problem to get initialization.
    // Max-margin phase-1 initializer that maximizes scalar distance to boundary (t).
    // min -t
    // s.t A*x + b = 0
    //     C*x + d - t >= 0
    //     t <= 1
    LPProblem aux_problem;

    const int nx = c.size();
    const int nt = 1;
    const int ns = d.size();

    // Smart initial point: strictly feasible for auxiliary problem by construction
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
    if (A.rows() > 0)
    {
        x0 = -A.completeOrthogonalDecomposition().solve(b);
    }
    double min_slack = (ns > 0) ? (C * x0 + d).minCoeff() : 0.0;
    double t0 = std::min(0.0, min_slack) - 1.0;

    Eigen::VectorXd x0_aux(nx + nt);
    x0_aux << x0, t0;
    aux_problem.x0 = x0_aux;

    // objective [0; -1]
    aux_problem.c.resize(nx + nt);
    aux_problem.c <<
        Eigen::VectorXd::Zero(nx),
        -Eigen::VectorXd::Ones(nt);

    // equality matrix [A 0]
    Eigen::MatrixXd A_aux(A.rows(), nx + nt);
    A_aux << A, Eigen::MatrixXd::Zero(A.rows(), nt);
    aux_problem.A = A_aux;

    // rhs b
    Eigen::VectorXd b_aux = b;
    aux_problem.b = b_aux;

    // inequalities: C*x - t*1 + d >= 0 AND -t + 1 >= 0 (t <= 1)
    Eigen::MatrixXd C_aux(ns + 1, nx + nt);
    C_aux << C, -Eigen::VectorXd::Ones(ns),
             Eigen::MatrixXd::Zero(1, nx), -Eigen::MatrixXd::Identity(1, 1);
    aux_problem.C = C_aux;

    Eigen::VectorXd d_aux(ns + 1);
    d_aux << d, 1.0;
    aux_problem.d = d_aux;

    LPSolver solver(options, aux_problem);

    Result aux_result = solver.solve();

    // Check that point is feasible and strictly interior.
    Eigen::VectorXd x = aux_result.x.head(nx);
    double t_final = aux_result.x(nx);

    double d_norm = (ns > 0) ? d.lpNorm<Eigen::Infinity>() : 0.0;
    double rel_tol = 1e-8 * (1.0 + d_norm);

    bool equality_feasible =
        (A.rows() == 0) || ((A * x + b).norm() <= rel_tol);

    bool strict_interior = (t_final > rel_tol);

    if (equality_feasible && strict_interior)
    {
        return x;
    }

    if (equality_feasible && t_final >= -rel_tol)
    {
        throw std::runtime_error(
            "Feasible point lies on the boundary (empty interior). Cannot start IPM Phase-2.");
    }

    throw std::runtime_error(
        "Auxiliary LP did not produce a feasible point.");
};

}