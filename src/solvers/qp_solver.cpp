#include "solvers/qp_solver.hpp"
#include "solvers/lp_solver.hpp"
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
        const Eigen::VectorXd& c = problem_.get().c;
        const Eigen::MatrixXd& A = problem_.get().A.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
        const Eigen::VectorXd& b = problem_.get().b.value_or(Eigen::VectorXd::Zero(0));
        const Eigen::MatrixXd& C = problem_.get().C.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
        const Eigen::VectorXd& d = problem_.get().d.value_or(Eigen::VectorXd::Zero(0));
        x_0_ = LPSolver::computeFeasiblePoint(c, A, b, C, d, options_.get());
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
    result.summary.iterations = 0;
    result.summary.converged = true;
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
    result.summary.iterations = 1; // Direct solve
    result.summary.converged = true;
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
    // Extract problem data safely
    const Eigen::VectorXd& c = problem_.get().c;
    const Eigen::MatrixXd& H = problem_.get().H;
    const Eigen::MatrixXd& A = problem_.get().A.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& b = problem_.get().b.value_or(Eigen::VectorXd::Zero(0));
    const Eigen::MatrixXd& C = problem_.get().C.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& d = problem_.get().d.value_or(Eigen::VectorXd::Zero(0));

    // Structural Dimensions
    const size_t n = c.rows();
    const size_t m_eq = A.rows();

    // Variable Initialization
    Eigen::VectorXd x = x_0_;
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m_eq);

    // IPM Control Parameters
    double tau = options_.get().tau_initial;              // Initial barrier parameter strength
    const double mu = options_.get().tau_factor;          // Tau attenuation stepping scalar
    const int max_outer = options_.get().max_outer;       // Barrier reduction iterations
    const int max_inner = options_.get().max_inner;       // Fixed centering Newton steps per inner loop
    const double tol = options_.get().ipm_tol;        // Global convergence threshold

    // Allocate KKT Memory Frame
    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n + m_eq, n + m_eq);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + m_eq);

    // Preset time-invariant constraint mappings
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
            
            // Calculate inequality clearances: s = C*x + d
            Eigen::VectorXd s = C * x + d;
            
            // Safety guard to ensure strict feasibility domain
            if ((s.array() <= 0).any()) {
                throw std::runtime_error("QP IPM path tracking drifted out of the feasible interior domain (s <= 0).");
            }

            // Invert the slack arrays for matrix assembly
            Eigen::VectorXd s_inv = s.cwiseInverse();
            Eigen::VectorXd s_inv2 = s_inv.cwiseAbs2();

            const double rho_tol = 1e-8;   // Primal regularization (damps undetermined primal steps)
            const double delta_tol = 1e-8; // Dual regularization (relaxes linearly dependent constraints)

            // 1. Build Left-Hand Side (LHS) Block: H + tau * C^T * diag(s)^-2 * C
            // Add tiny diagonal regularization (1e-12) to prevent rank deficient KKT matrix
            KKT.block(0, 0, n, n) = H + tau * C.transpose() * s_inv2.asDiagonal() * C + rho_tol * Eigen::MatrixXd::Identity(n, n);

            // Bottom-right Dual Block: (+ delta * I)
            if (m_eq > 0) {
                KKT.block(n, n, m_eq, m_eq) = delta_tol * Eigen::MatrixXd::Identity(m_eq, m_eq);
            }

            // 2. Build Right-Hand Side (RHS) Residual Vectors
            Eigen::VectorXd dual_res = -c - H * x + A.transpose() * lambda + tau * C.transpose() * s_inv;
            Eigen::VectorXd primal_res = -A * x - b;

            rhs.head(n) = dual_res;
            rhs.tail(m_eq) = primal_res;

            // Break if the centering step has already converged
            if (dual_res.norm() < tol && primal_res.norm() < tol) {
                break;
            }

            // 3. Robust Direct Matrix Solve using Column-Pivoted Householder QR
            auto qr = KKT.colPivHouseholderQr();

            if (qr.rank() < KKT.rows())
            {
                logger_->error("KKT matrix is rank deficient. IPM cannot proceed.");
                throw std::runtime_error("KKT matrix is rank deficient. IPM cannot proceed.");
            }

            Eigen::VectorXd delta = qr.solve(rhs);
            
            Eigen::VectorXd dx = delta.head(n);
            Eigen::VectorXd dlambda = delta.tail(m_eq);

            // 4. Backtracking Line Search (Fraction-to-boundary rules)
            double alpha = 1.0;
            const double beta = 0.5;
            
            while (alpha > 1e-8) {
                Eigen::VectorXd s_next = C * (x + alpha * dx) + d;
                if ((s_next.array() > 0).all()) {
                    break; // Stepping target remains strictly inside the feasible polytope
                }
                alpha *= beta;
            }

            // Apply updates
            x += alpha * dx;
            lambda += alpha * dlambda;

            // Minor inner termination criteria if step length vanishes
            if (dx.norm() < tol) break;
        }

        // Tighter optimization target scaling
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
    
    // Wrap results
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
}

}