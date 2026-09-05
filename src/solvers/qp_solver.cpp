#include "solvers/qp_solver.hpp"
#include "solvers/feasibility.hpp"
#include "barrier_ipm.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace furiaopt{

QPSolver::QPSolver(const IPMSolverOptions& options, const QPProblem& problem) : options_(std::cref(options)), problem_(std::cref(problem)), logger_(options_.get().logger ? options_.get().logger : std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>())) {

    cost_func_ = [&problem](const Eigen::VectorXd& x) -> double {
        return 0.5 * x.transpose() * problem.H * x + problem.c.dot(x);
    };
};

Result QPSolver::solve(){

    logger_->info("Starting solve");
    Result result;

    // Initialize variables
    if (problem_.get().x0.has_value())
    {
        x_0_ = problem_.get().x0.value();
    }
    else
    {
        if (problem_.get().hasEqualityConstraints() || problem_.get().hasInequalityConstraints())
        {
            logger_->info("No initial guess provided, computing a feasible point for the QP problem.");
            const Eigen::VectorXd& c = problem_.get().c;
            const Eigen::MatrixXd& A = problem_.get().A.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
            const Eigen::VectorXd& b = problem_.get().b.value_or(Eigen::VectorXd::Zero(0));
            const Eigen::MatrixXd& C = problem_.get().C.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
            const Eigen::VectorXd& d = problem_.get().d.value_or(Eigen::VectorXd::Zero(0));
            x_0_ = computeFeasiblePoint(c, A, b, C, d, options_.get());
        }
        else
        {
            logger_->info("No initial guess provided, using zero vector as initial guess.");
            x_0_ = Eigen::VectorXd::Zero(problem_.get().c.size());
        }
    }
    
    result.summary.initial_cost = cost_func_(x_0_);

    if (problem_.get().hasInequalityConstraints())
    {
        general_QP_solver(result);
    }
    else if (problem_.get().hasEqualityConstraints())
    {
        equality_constrained_QP_solver(result);
    }
    else
    {
        no_constraints_QP_solver(result);
    }
    return result;
};


void QPSolver::no_constraints_QP_solver(Result& result)
{
    // For quadratic problems w/o constraints, we can directly compute the optimal solution by solving Hx + c = 0
    Eigen::MatrixXd H = problem_.get().H;
    Eigen::VectorXd c = problem_.get().c;
    Eigen::LLT<Eigen::MatrixXd> llt(H);
    if (llt.info() == Eigen::Success) {
        result.x = llt.solve(-c);
    } else {
        // If LLT fails, try LDLT decomposition, can handle both positive and negative semi definite Hessian
        Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
        if (ldlt.info() == Eigen::Success) {
            result.x = ldlt.solve(-c);
        } else {
            throw std::runtime_error("Failed to solve quadratic problem: Hessian is not semi-positive definite");
        }
    }
    result.lambda = Eigen::VectorXd::Zero(0);
    result.mhu = Eigen::VectorXd::Zero(0);
    result.summary.final_cost = cost_func_(result.x);
    result.summary.iterations = 0; // Direct solve
    result.summary.converged = true;
    result.summary.termination_reason = TerminationReason::DirectSolve;
};

void QPSolver::equality_constrained_QP_solver(Result& result)
{
    // For quadratic problems with only equality constraints, we can solve the KKT system directly:
    // [ H  - A^T ] [ x     ] = [ -c ]
    // [ A    0   ] [ lambda]   [ -b ]

    const Eigen::MatrixXd& H = problem_.get().H;
    const Eigen::VectorXd& c = problem_.get().c;
    const Eigen::MatrixXd& A = problem_.get().A.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& b = problem_.get().b.value_or(Eigen::VectorXd::Zero(0));

    size_t n = c.rows();
    size_t m_eq = A.rows();

    //Add regularization to the KKT matrix to improve stability
    const double rho_tol = 1e-8;
    const double delta_tol = 1e-8;

    Eigen::MatrixXd KKT(n + m_eq, n + m_eq);
    KKT.setZero();
    KKT.block(0, 0, n, n) = H + rho_tol * Eigen::MatrixXd::Identity(n, n);;
    KKT.block(0, n, n, m_eq) = -A.transpose();
    
    // To use LDLT, the matrix must be symmetric. By putting -A here, 
    // the system becomes perfectly symmetric. This is mathematically 
    // equivalent to multiplying the bottom row of your equation by -1:
    // -Ax = b  =>  which perfectly matches the RHS below!
    KKT.block(n, 0, m_eq, n) = -A;

    if (m_eq > 0) {
        KKT.block(n, n, m_eq, m_eq) = -delta_tol * Eigen::MatrixXd::Identity(m_eq, m_eq);
    }

    Eigen::VectorXd rhs(n + m_eq);
    rhs.head(n) = -c;
    rhs.tail(m_eq) = b; // Flipped to +b to match the -A row transformation

    auto ldlt = KKT.ldlt();

    if (ldlt.info() == Eigen::ComputationInfo::NumericalIssue)
    {
        logger_->error("KKT matrix is rank deficient or numerically unstable. Cannot solve equality constrained QP.");
        throw std::runtime_error("KKT matrix is rank deficient. Cannot solve equality constrained QP.");
    }

    Eigen::VectorXd solution = ldlt.solve(rhs);
    result.x = solution.head(n);
    if (m_eq > 0) {
        result.lambda = solution.tail(m_eq);
    } else {
        result.lambda = Eigen::VectorXd::Zero(0);
    }
    result.mhu = Eigen::VectorXd::Zero(0);
    result.summary.final_cost = cost_func_(result.x);
    result.summary.iterations = 0; // Direct solve
    result.summary.converged = true;
    result.summary.termination_reason = TerminationReason::DirectSolve;
};

void QPSolver::general_QP_solver(Result& result)
{
    //The QP problem is solved using the interior point method.
    //The problem is reformulated using barrier functions:
    //min c^T*x + 0.5*x^T*H*x - tau*sum_i(log(C*x_i + d_i))
    //s.t Ax + b = 0

    //KKT conditions for this problem reads as follows:
    //c + H*x - A^T*lambda - tau*sum_i(C_i/(C*x_i + d_i)) = 0
    //Ax + b = 0

    //Usign Newton's method to solve the KKT conditions, we get the following update equations:
    // [ H+ tau*C^T*(diag(C*x + d)^-2)*C | -A^T ; * [ dx     ;   = [ -c - H*x + A^T*lamnda + tau*C^T*(diag(C*x + d)^-1)*Eigen::VectorXd::Ones(c.rows());
    //   A                            |  0      ]    dlambda ]       -Ax - b                                                                     ]

    // Extract problem parameters safely from the reference wrapper
    const Eigen::VectorXd& c = problem_.get().c;
    const Eigen::MatrixXd& H = problem_.get().H;
    const Eigen::MatrixXd& A = problem_.get().A.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& b = problem_.get().b.value_or(Eigen::VectorXd::Zero(0));
    const Eigen::MatrixXd& C = problem_.get().C.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& d = problem_.get().d.value_or(Eigen::VectorXd::Zero(0));

    // Interior Point Method using Barrier Function
    IPMProblem ipm_problem;
    ipm_problem.x0 = x_0_;
    ipm_problem.H = H;
    ipm_problem.c = c;
    ipm_problem.A = A;
    ipm_problem.b = b;
    ipm_problem.C = C;
    ipm_problem.d = d;
    furiaopt::details::solve_ipm_problem(ipm_problem, options_.get(), logger_, "QP", result);
}

}