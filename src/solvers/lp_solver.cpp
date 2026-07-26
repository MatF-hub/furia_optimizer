#include "solvers/lp_solver.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "generalization_method.hpp"

inline std::string vec_to_string(const Eigen::VectorXd& v)
{
    std::ostringstream oss;
    oss << v.transpose();
    return oss.str();
}
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

    // Dimensions
    const size_t n = c.rows();
    const size_t m_eq = A.rows();

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

    result.summary.initial_cost = cost_func_(x);

    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m_eq);

    // IPM Hyperparameters
    double tau = options_.get().tau_initial;              // Initial barrier parameter strength
    const double mu = options_.get().tau_factor;          // Tau attenuation stepping scalar
    const int max_outer = options_.get().max_outer;       // Barrier reduction iterations
    const int max_inner = options_.get().max_inner;       // Fixed centering Newton steps per inner loop
    const double tol = options_.get().ipm_tol;        // Global convergence threshold

    // Pre-allocate KKT System: Left-Hand Side (LHS) and Right-Hand Side (RHS)
    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n + m_eq, n + m_eq);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + m_eq);

    // Top-right and Bottom-left blocks of KKT are constant throughout the loop
    KKT.block(0, n, n, m_eq) = -A.transpose();
    KKT.block(n, 0, m_eq, n) = A;

    int outer = 0;
    while (outer < max_outer) {

        logger_->info(
            "iter={},cost={:.8f},equality_constraint={:.3e},inequality_constraint={:.3e},x={}",
            outer,
            cost_func_(x),
            (A * x + b).norm(),
            (C * x + d).norm(),
            vec_to_string(x)
        );

        for (int inner = 0; inner < max_inner; ++inner) {
            
            // Vector elements: s_i = C_i * x + d_i
            Eigen::VectorXd s = C * x + d;
            
            // Check strictly feasible condition safety boundary
            if ((s.array() <= 0).any()) {
                throw std::runtime_error("IPM drifted out of the feasible region (s <= 0).");
            }

            // Compute inverse scaling matrices entries
            Eigen::VectorXd s_inv = s.cwiseInverse();
            Eigen::VectorXd s_inv2 = s_inv.cwiseAbs2();

            const double rho_tol = 1e-8;   // Primal regularization (damps undetermined primal steps)
            const double delta_tol = 1e-8; // Dual regularization (relaxes linearly dependent constraints)

            // 1. Compute LHS Hessian component: tau * C^T * diag(s)^-2 * C
            // Using .asDiagonal() avoids full matrix multiplication overhead
            // Add tiny diagonal regularization (1e-12) to prevent rank deficient KKT matrix
            KKT.block(0, 0, n, n) = tau * C.transpose() * s_inv2.asDiagonal() * C + rho_tol * Eigen::MatrixXd::Identity(n, n);

            // Bottom-right Dual Block: (+ delta * I)
            if (m_eq > 0) {
                KKT.block(n, n, m_eq, m_eq) = delta_tol * Eigen::MatrixXd::Identity(m_eq, m_eq);
            }

            // 2. Compute RHS Residuals
            Eigen::VectorXd dual_res = - c + A.transpose() * lambda + tau * C.transpose() * s_inv;
            Eigen::VectorXd primal_res = -A * x - b;

            rhs.head(n) = dual_res;
            rhs.tail(m_eq) = primal_res;

            // Check Convergence
            if (dual_res.norm() < tol && primal_res.norm() < tol) {
                break;
            }

            // 3. Solve the full KKT system using ColPivHouseholderQR (robust for dense KKTs)
            auto qr = KKT.colPivHouseholderQr();
            Eigen::VectorXd delta = qr.solve(rhs);
            Eigen::VectorXd dx = delta.head(n);
            Eigen::VectorXd dlambda = delta.tail(m_eq);

            // 4. Backtracking Line Search (Ensure inequality constraints are never violated)
            double alpha = 1.0;
            const double beta = 0.5; // Step reduction factor
            
            // Fraction-to-the-boundary rule: ensure C*(x + alpha*dx) + d > 0
            while (alpha > 1e-8) {
                Eigen::VectorXd s_next = C * (x + alpha * dx) + d;
                if ((s_next.array() > 0).all()) {
                    break;
                }
                alpha *= beta;
            }

            // Update positions
            x += alpha * dx;
            lambda += alpha * dlambda;

            if (dx.norm() < tol) break;
        }

        // Reduce barrier parameter to tighten the approximation to the true LP
        tau *= mu;
        if (tau < 1e-8) break;
        ++outer;
    }

    Eigen::VectorXd s_final = Eigen::VectorXd::Zero(d.rows());
    Eigen::VectorXd mhu = Eigen::VectorXd::Zero(C.rows());
    if (C.rows() > 0) {
        s_final = C * x + d;
        if ((s_final.array() <= 0).any()) {
            throw std::runtime_error("Final solution is not strictly feasible for inequality constraints.");
        }
        mhu = tau * s_final.cwiseInverse();
    }

    double primal_residual = (m_eq > 0) ? (A * x + b).norm() : 0.0;

    Eigen::VectorXd dual_residual_eq = c - A.transpose() * lambda;
    if (C.rows() > 0) {
        dual_residual_eq -= C.transpose() * mhu;
    }
    double dual_residual = dual_residual_eq.norm();

    double duality_gap = (C.rows() > 0) ? s_final.dot(mhu) : 0.0;

    // Pack results
    result.x = x;
    result.lambda = lambda;
    result.mhu = mhu;
    result.summary.iterations = outer;
    result.summary.final_cost = cost_func_(result.x);
    result.summary.converged = (primal_residual < tol) && 
                               (dual_residual < tol) && 
                               (duality_gap < tol);
    if (!result.summary.converged) {
        result.summary.termination_reason = TerminationReason::MaxIterations;
    };

};

Eigen::VectorXd LPSolver::computeFeasiblePoint(
    const Eigen::VectorXd& c,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& d,
    const IPMSolverOptions& options)
{
    //If initialization is not provided use auxiliary problem to get initialization.
    //min s
    //s.t A*x_0+b=0;
    //    C*x_0+d-s=0;
    //    s>=0;
    LPProblem aux_problem;

    const int nx = c.size();
    const int ns = d.size();
    aux_problem.x0 = Eigen::VectorXd::Ones(nx + ns);

    // objective [0; 1]
    aux_problem.c.resize(nx + ns);
    aux_problem.c <<
        Eigen::VectorXd::Zero(nx),
        Eigen::VectorXd::Ones(ns);

    // equality matrix [A 0; C -I]
    Eigen::MatrixXd A_aux;
    A_aux.resize( A.rows() + C.rows(), nx + ns);
    A_aux << A, Eigen::MatrixXd::Zero(A.rows(), ns),
                C, -Eigen::MatrixXd::Identity(ns, ns);

    aux_problem.A = A_aux;

    // rhs [-b; -d]
    Eigen::VectorXd b_aux;
    b_aux.resize(b.size() + d.size());
    b_aux << b, d;
    aux_problem.b = b_aux;

    // inequalities: s >= 0
    Eigen::MatrixXd C_aux;
    C_aux.resize(ns, nx + ns);
    C_aux <<  Eigen::MatrixXd::Zero(ns, nx),
        Eigen::MatrixXd::Identity(ns, ns);
    aux_problem.C = C_aux;
    
    aux_problem.d = Eigen::VectorXd::Zero(ns);

    LPSolver solver(options, aux_problem);

    Result aux_result = solver.solve();

    // Check that point is feasible.
    Eigen::VectorXd x = aux_result.x.head(nx);

    bool equality_feasible =
        (A * x + b).norm() <= 1e-10;

    bool inequality_feasible = true;
    if (C.rows() > 0)
    {
        inequality_feasible = (C * x + d).minCoeff() >= -1e-10;
    }

    if (equality_feasible && inequality_feasible)
    {
        return x;
    }

    throw std::runtime_error(
        "Auxiliary LP did not produce a feasible point.");
};

}