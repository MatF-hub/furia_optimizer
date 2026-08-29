#include "barrier_ipm.hpp"
#include "utils.hpp"
#include "types.hpp"
#include <Eigen/Dense>

namespace furiaopt::details {

    void solve_ipm_problem(const IPMProblem& problem, 
                           const IPMSolverOptions& options, 
                           const std::shared_ptr<spdlog::logger>& logger,
                           const char* who,
                           Result& result)
    {
        // The barrier KKT conditions are
        //     H*x + c - A^T*lambda - tau*sum_i(C_i/(C_i*x + d_i)) = 0
        //     A*x + b = 0
        // and one Newton step on them is
        //     [ H + tau*C^T*diag(s)^-2*C   -A^T ] [ dx      ]   [ -H*x - c + A^T*lambda + tau*C^T*diag(s)^-1*1 ]
        //     [ A                           0   ] [ dlambda ] = [ -A*x - b                                     ]
        // with s = C*x + d. For an LP the H terms simply drop out.

        const std::optional<Eigen::MatrixXd>& H = problem.H;
        const Eigen::VectorXd& c = problem.c;
        const Eigen::MatrixXd& A = problem.A.value();
        const Eigen::VectorXd& b = problem.b.value();
        const Eigen::MatrixXd& C = problem.C.value();
        const Eigen::VectorXd& d = problem.d.value();

        CostFunc cost_function = [H, c](const Eigen::VectorXd& x) -> double{
            double cost = c.dot(x);
            if (H.has_value()) {
                cost += 0.5 * x.transpose() * H.value() * x;
            }
            return cost;
        };

        result.summary.initial_cost = cost_function(problem.x0);

        // Structural dimensions
        const Eigen::Index n = c.rows();
        const Eigen::Index m_eq = A.rows();
        const Eigen::Index m_ineq = C.rows();

        // Variable initialization
        Eigen::VectorXd x = problem.x0;
        Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m_eq);

        // IPM control parameters
        double tau           = options.tau_initial;   // Initial barrier parameter strength
        const double mu      = options.tau_factor;    // Tau attenuation stepping scalar
        const int max_outer  = options.max_outer;     // Barrier reduction iterations
        const int max_inner  = options.max_inner;     // Newton steps per centering loop
        const double tol     = options.ipm_tol;       // Global convergence threshold

        // Allocate the KKT frame once; only the (0,0) and (n,n) blocks change.
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n + m_eq, n + m_eq);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + m_eq);

        // Time-invariant constraint mappings
        KKT.block(0, n, n, m_eq) = -A.transpose();
        KKT.block(n, 0, m_eq, n) = A;

        int outer = 0;
        while (outer < max_outer) {

            logger->info(
                "iter={},cost={:.8f},equality_constraint={:.3e},inequality_constraint={:.3e},x={}",
                outer,
                cost_function(x),
                (A * x + b).norm(),
                (C * x + d).norm(),
                furiaopt::utils::vec_to_string(x)
            );

            for (int inner = 0; inner < max_inner; ++inner) {

                // Inequality clearances: s = C*x + d
                Eigen::VectorXd s = C * x + d;

                // Safety guard to ensure the strict feasibility domain
                if ((s.array() <= 0).any()) {
                    logger->error("[{}] IPM left the strict interior at outer={} inner={} (min s = {:.3e})",
                                who, outer, inner, m_ineq > 0 ? s.minCoeff() : 0.0);
                    throw std::runtime_error(
                        std::string(who) + " IPM path tracking drifted out of the feasible interior domain (s <= 0).");
                }

                // Inverse slack scalings for matrix assembly
                Eigen::VectorXd s_inv  = s.cwiseInverse();
                Eigen::VectorXd s_inv2 = s_inv.cwiseAbs2();

                const double rho_tol   = 1e-8;   // Primal regularization (damps undetermined primal steps)
                const double delta_tol = 1e-8;   // Dual regularization (relaxes linearly dependent constraints)

                // LHS block: H + tau * C^T * diag(s)^-2 * C + rho*I
                // Regularization keeps the reduced KKT nonsingular even when C is rank deficient in x
                KKT.block(0, 0, n, n) = tau * C.transpose() * s_inv2.asDiagonal() * C
                                        + rho_tol * Eigen::MatrixXd::Identity(n, n);
                if (H.has_value()) {
                    KKT.block(0, 0, n, n) += H.value();
                }

                // Bottom-right dual block: +delta*I
                if (m_eq > 0) {
                    KKT.block(n, n, m_eq, m_eq) = delta_tol * Eigen::MatrixXd::Identity(m_eq, m_eq);
                }

                // 2. RHS residuals
                Eigen::VectorXd dual_res = -c + A.transpose() * lambda + tau * C.transpose() * s_inv;
                if (H.has_value()) {
                    dual_res += H.value() * x;
                }

                Eigen::VectorXd primal_res = -A * x - b;

                rhs.head(n) = dual_res;
                rhs.tail(m_eq) = primal_res;

                // Centering step already converged for this tau
                if (dual_res.norm() < tol && primal_res.norm() < tol) {
                    break;
                }

                // Direct solve with column-pivoted Householder QR (robust for dense KKTs)
                auto qr = KKT.colPivHouseholderQr();
                Eigen::VectorXd delta = qr.solve(rhs);

                Eigen::VectorXd dx      = delta.head(n);
                Eigen::VectorXd dlambda = delta.tail(m_eq);

                // Backtracking Line Search w/o optimal stopping criteria
                // Fraction-to-the-boundary rule: ensure C*(x + alpha*dx) + d > 0
                double alpha       = 1.0;
                const double beta  = 0.5;
                const double frac  = 0.995;
                while (alpha > 1e-8) {
                    Eigen::VectorXd s_next = C * (x + alpha * dx) + d;
                    if (m_ineq == 0 || (s_next.array() >= (1.0 - frac) * s.array()).all()) break;
                    alpha *= beta;
                }

                // Apply the update
                x += alpha * dx;
                lambda += alpha * dlambda;

                // Check if step length is too small to continue like done for unconstrained opt problems
                if (alpha * dx.norm() < tol * std::max(x.norm(), 1e-16)) break;
            }

            // Barrier parameter is too small, terminate the outer loop
            if (m_ineq==0 || m_ineq*tau < tol) 
            {
                result.summary.termination_reason = TerminationReason::StepTolerance;
                break;
            }
            // Reduce barrier parameter to tighten the approximation to the true problem
            tau *= mu;
            ++outer;
        }

        // Recover the inequality multipliers from the barrier: mu_i = tau / s_i
        Eigen::VectorXd s_final = Eigen::VectorXd::Zero(m_ineq);
        Eigen::VectorXd mhu     = Eigen::VectorXd::Zero(m_ineq);
        if (m_ineq > 0) {
            s_final = C * x + d;
            if ((s_final.array() <= 0).any()) {
                throw std::runtime_error(
                    std::string(who) + " final solution is not strictly feasible for inequality constraints.");
            }
            mhu = tau * s_final.cwiseInverse();
        }

        // KKT residuals of the ORIGINAL problem
        const double primal_residual = (m_eq > 0) ? (A * x + b).norm() : 0.0;

        Eigen::VectorXd dual_residual_eq = c - A.transpose() * lambda;
        if (H.has_value()) {
            dual_residual_eq += H.value() * x;
        }
        if (m_ineq > 0) {
            dual_residual_eq -= C.transpose() * mhu;
        }
        const double dual_residual = dual_residual_eq.norm();

        const double duality_gap = (m_ineq > 0) ? s_final.dot(mhu) : 0.0;

        // Wrap results
        result.x      = x;
        result.lambda = lambda;
        result.mhu    = mhu;
        result.summary.iterations = outer;
        result.summary.final_cost = cost_function(x);
        result.summary.converged  = (primal_residual < tol)
                                && (dual_residual   < tol)
                                && (duality_gap     < tol);
        if (!result.summary.converged && outer >= max_outer) {
            result.summary.termination_reason = TerminationReason::MaxIterations;
        }

    }
}