#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include "solvers/unconstrained_solver.hpp"
#include "solvers/qp_solver.hpp"
#include "solvers/lp_solver.hpp"
#include "solvers/constrained_solver.hpp"
#include "direction_strategy.hpp"
#include "generalization_method.hpp"
#include "compute_gradient.hpp"

#include <iostream>

using namespace furiaopt;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Catch::Approx;

// -----------------------------------------------------------------------------
//  helpers
// -----------------------------------------------------------------------------
static UnconstrainedSolverOptions nlp_opts(DirectionMethod m, int max_iter = 1000)
{
    UnconstrainedSolverOptions o;
    o.direction_method     = m;
    o.globalization_method = GlobalizationMethod::LineSearch;
    o.max_iter             = max_iter;
    o.gradient_tolerance   = 1e-8;
    o.step_tolerance       = 1e-14;
    o.function_tolerance   = 1e-16;
    //logger is initialized to null logger by default
    return o;
}

static IPMSolverOptions ipm_opts()
{
    return IPMSolverOptions{ /*tau_initial*/ 1.0, /*tau_factor*/ 0.2,
                             /*max_outer*/ 60, /*max_inner*/ 60, /*ipm_tol*/ 1e-9 };
}

// SQP reference problems. 
static NLPProblem sqp_linear_equality()      
{                                            
    // min 0.5||x||^2  s.t. x0+x1-2 = 0
    // optimimum: x*=(1,1)  lambda*=1
    NLPProblem p;
    p.x0 = (VectorXd(2) << 0.0, 0.0).finished();
    p.cost_func     = [](const VectorXd& x) -> double { return 0.5 * x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd { return x; };
    p.equality_constraint_func = [](const VectorXd& x) -> VectorXd {
        VectorXd c(1); c[0] = x[0] + x[1] - 2.0; return c; };
    p.gradient_equality_constraint_func = [](const VectorXd&) -> MatrixXd {
        MatrixXd J(2,1); J << 1.0, 1.0; return J; };
    return p;
}

static NLPProblem sqp_active_inequality()
{
    // min 0.5||x-2||^2  s.t. 2-x0-x1 >= 0
    // optimum: x*=(1,1)  mu*=1
    NLPProblem p;
    p.x0 = (VectorXd(2) << 0.0, 0.0).finished();
    p.cost_func     = [](const VectorXd& x) -> double {
        return 0.5 * ((x[0]-2)*(x[0]-2) + (x[1]-2)*(x[1]-2)); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd {
        VectorXd g(2); g << x[0]-2, x[1]-2; return g; };
    p.inequality_constraint_func = [](const VectorXd& x) -> VectorXd {
        VectorXd c(1); c[0] = 2.0 - x[0] - x[1]; return c; };
    p.gradient_inequality_constraint_func = [](const VectorXd&) -> MatrixXd {
        MatrixXd J(2,1); J << -1.0, -1.0; return J; };
    return p;
}

static NLPProblem sqp_circle()
{
    // min x0+x1  s.t. ||x||^2 = 1
    // optimum: x*=-(sqrt2/2)(1,1)  lambda*=-sqrt2/2
    NLPProblem p;
    p.x0 = (VectorXd(2) << 0.8, 0.2).finished();
    p.cost_func     = [](const VectorXd& x) -> double { return x[0] + x[1]; };
    p.gradient_func = [](const VectorXd&) -> VectorXd {
        VectorXd g(2); g << 1.0, 1.0; return g; };
    p.equality_constraint_func = [](const VectorXd& x) -> VectorXd {
        VectorXd c(1); c[0] = x.squaredNorm() - 1.0; return c; };
    p.gradient_equality_constraint_func = [](const VectorXd& x) -> MatrixXd {
        MatrixXd J(2,1); J << 2*x[0], 2*x[1]; return J; };
    return p;
}

static NLPProblem sqp_mixed()
{
    // min ||x-1||^2  s.t. ||x||^2=1, x0>=0
    // optimum: x*=(sqrt2/2)(1,1)  lambda*=1-sqrt2  mu*=0
    NLPProblem p;
    p.x0 = (VectorXd(2) << 0.2, 0.9).finished();
    p.cost_func     = [](const VectorXd& x) -> double {
        return (x[0]-1)*(x[0]-1) + (x[1]-1)*(x[1]-1); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd {
        VectorXd g(2); g << 2*(x[0]-1), 2*(x[1]-1); return g; };
    p.equality_constraint_func = [](const VectorXd& x) -> VectorXd {
        VectorXd c(1); c[0] = x.squaredNorm() - 1.0; return c; };
    p.gradient_equality_constraint_func = [](const VectorXd& x) -> MatrixXd {
        MatrixXd J(2,1); J << 2*x[0], 2*x[1]; return J; };
    p.inequality_constraint_func = [](const VectorXd& x) -> VectorXd {
        VectorXd c(1); c[0] = x[0]; return c; };
    p.gradient_inequality_constraint_func = [](const VectorXd&) -> MatrixXd {
        MatrixXd J(2,1); J << 1.0, 0.0; return J; };
    return p;
}

// =============================================================================
//  Direction strategies
// =============================================================================
TEST_CASE("GradientDescentDirection returns -gradient", "[direction][gd]")
{
    NLPProblem p;
    p.x0 = VectorXd::Zero(2);
    p.cost_func     = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd { return (2 * x).eval(); };

    GradientDescentDirection dir(p);
    VectorXd g(2); g << 2.0, 4.0;
    VectorXd d = dir.getDirection(g, VectorXd::Zero(2));
    REQUIRE(d.isApprox(-g, 1e-12));
}

TEST_CASE("ExactHessianDirection evaluates Hessian at current x_i", "[direction][newton]")
{
    // f(x) = 0.5 (x-3)^2 : H = 1, gradient at x_i = 5 is 2 -> Newton step = -2
    NLPProblem p;
    p.x0 = VectorXd::Constant(1, 0.0);
    const double target = 3.0;
    p.cost_func     = [target](const VectorXd& x) -> double { return 0.5 * (x[0]-target)*(x[0]-target); };
    p.gradient_func = [target](const VectorXd& x) -> VectorXd { VectorXd g(1); g[0]=x[0]-target; return g; };
    p.hessian_func  = [](const VectorXd&) -> MatrixXd { return MatrixXd::Identity(1,1); };

    ExactHessianDirection strat(p);
    VectorXd x_i(1); 
    x_i << 5.0;
    VectorXd g(1);   
    g << 2.0;
    VectorXd d = strat.getDirection(g, x_i);
    REQUIRE(d[0] == Approx(-2.0).epsilon(1e-10));
}

TEST_CASE("BFGS first call returns the steepest-descent direction", "[direction][bfgs]")
{
    NLPProblem p;
    p.x0 = VectorXd::Zero(2);
    p.cost_func     = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd { return (2*x).eval(); };
    BFGSDirection strat(p);
    VectorXd g(2); 
    g << 2.0, 4.0;
    VectorXd d = strat.getDirection(g, VectorXd::Zero(2));   // H0 = I -> d = -g
    REQUIRE(d.isApprox(-g, 1e-12));
}

TEST_CASE("BFGS satisfies the secant equation when curvature is positive", "[direction][bfgs]")
{
    NLPProblem p;
    p.x0 = VectorXd::Zero(2);

    BFGSHessianApproximation approx(p);

    VectorXd g0(2);
    g0 << 0.0, 0.0;
    VectorXd x0(2);
    x0 << 0.0, 0.0;
    approx.getApproximateHessian(g0, x0); // initialize H = I

    VectorXd g1(2);
    g1 << 0.5, 0.0;
    VectorXd x1(2);
    x1 << 1.0, 0.0;

    MatrixXd H = approx.getApproximateHessian(g1, x1);
    VectorXd s = x1 - x0;
    VectorXd y = g1 - g0;

    REQUIRE((H * s).isApprox(y, 1e-12));
}

TEST_CASE("BFGS stays symmetric positive-definite through a negative-curvature step", "[direction][bfgs]")
{
    NLPProblem p;
    p.x0 = VectorXd::Zero(2);

    BFGSHessianApproximation approx(p);

    VectorXd g0(2);
    g0 << 0.0, 0.0;
    VectorXd x0(2);
    x0 << 0.0, 0.0;
    approx.getApproximateHessian(g0, x0);

    VectorXd g1(2);
    g1 << -0.5, 0.0;
    VectorXd x1(2);
    x1 << 1.0, 0.0;

    MatrixXd H = approx.getApproximateHessian(g1, x1);

    REQUIRE(H.isApprox(H.transpose(), 1e-12));

    Eigen::LLT<Eigen::MatrixXd> llt;
    llt.compute(H);
    REQUIRE(llt.info() == Eigen::Success);
}

TEST_CASE("BFGS: setPreviousGradient changes the next update", "[direction][bfgs]")
{
    // Overwriting g_k must change y = g_{k+1} - g_k, and therefore H.
    NLPProblem p; p.x0 = VectorXd::Zero(2);
    VectorXd x0(2); x0 << 0.0, 0.0;
    VectorXd x1(2); x1 << 1.0, 0.0;
    VectorXd g0(2); g0 << 0.0, 0.0;
    VectorXd g1(2); g1 << 0.5, 0.0;

    BFGSHessianApproximation a(p);
    a.getApproximateHessian(g0, x0);
    const MatrixXd H_plain = a.getApproximateHessian(g1, x1);

    BFGSHessianApproximation b(p);
    b.getApproximateHessian(g0, x0);
    b.setPreviousGradient((VectorXd(2) << 10.0, 0.0).finished());
    const MatrixXd H_over = b.getApproximateHessian(g1, x1);

    REQUIRE(H_plain(0,0) == Approx(0.5).margin(1e-12));
    REQUIRE_FALSE(H_over.isApprox(H_plain, 1e-9));
    REQUIRE(H_over.isApprox(H_over.transpose(), 1e-12));
    Eigen::LLT<MatrixXd> llt_over(H_over);
    REQUIRE(llt_over.info() == Eigen::Success);
}

TEST_CASE("BFGS: a zero step is skipped instead of producing NaN", "[direction][bfgs][regression]")
{
    // The update divides by s^T H s, which is 0 when s = 0.
    NLPProblem p; p.x0 = VectorXd::Zero(2);
    BFGSHessianApproximation a(p);
    VectorXd x(2); x << 1.0, 1.0;
    VectorXd g(2); g << 0.5, 0.5;

    a.getApproximateHessian(g, x);
    const MatrixXd H = a.getApproximateHessian(g, x);   // s = 0, y = 0
    REQUIRE_FALSE(H.hasNaN());
    REQUIRE(H.isApprox(MatrixXd::Identity(2,2), 1e-12));
}

TEST_CASE("BFGS: repeated updates stay symmetric positive definite", "[direction][bfgs]")
{
    NLPProblem p; p.x0 = VectorXd::Zero(2);
    BFGSHessianApproximation a(p);

    const double xs[5][2] = {{0,0}, {1,0}, {1,1}, {0.5,1.5}, {-0.5,1.0}};
    const double gs[5][2] = {{0,0}, {0.5,0}, {0.4,0.9}, {-0.2,0.7}, {0.9,-0.3}};

    for (int k = 0; k < 5; ++k) {
        VectorXd x(2); x << xs[k][0], xs[k][1];
        VectorXd g(2); g << gs[k][0], gs[k][1];
        const MatrixXd H = a.getApproximateHessian(g, x);
        CAPTURE(k, H);
        REQUIRE_FALSE(H.hasNaN());
        REQUIRE(H.isApprox(H.transpose(), 1e-10));
        Eigen::LLT<MatrixXd> llt(H);
        REQUIRE(llt.info() == Eigen::Success);
    }
}

TEST_CASE("Gauss-Newton Hessian equals 2 * J^T J + sigma * I", "[direction][gauss_newton]")
{
    LSProblem p;
    p.x0 = VectorXd::Zero(2);

    MatrixXd J(2, 2);
    J << 1.0, 0.0,
         0.0, 2.0;

    p.residual_func = [](const VectorXd&) -> VectorXd {
        return VectorXd::Zero(2);
    };

    p.gradient_residual_func = [J](const VectorXd&) -> MatrixXd {
        return J.transpose();
    };

    GaussNewtonHessianApproximation approx(p);
    MatrixXd H = approx.getApproximateHessian(VectorXd::Zero(2));

    MatrixXd expected = 2.0 * J.transpose() * J + 1e-10 * MatrixXd::Identity(2, 2);
    REQUIRE(H.isApprox(expected, 1e-12));
}

TEST_CASE("ExactNewton direction without Hessian function throws invalid_argument", "[guards]") {
    UnconstrainedSolverOptions opts = nlp_opts(DirectionMethod::ExactNewton);
    
    NLPProblem p;
    p.x0 = VectorXd::Zero(2);
    p.cost_func = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> Eigen::VectorXd { return 2.0 * x; };

    REQUIRE_THROWS_AS(UnconstrainedSolver(opts, p), std::invalid_argument);
}

// =============================================================================
//  Compute Gradient Function
// =============================================================================
TEST_CASE("compute_gradient NLProblem returns the correct gradient for a simple quadratic", "[gradient]")
{
    NLPProblem p; p.cost_func=[](auto&x) -> double { return x.squaredNorm(); };
    p.gradient_func=[](auto&x) -> VectorXd { return (2*x).eval(); };
    VectorXd x(2); x<<1,2;
    VectorXd expected(2); expected<<2,4;
    REQUIRE(p.hasGradient());
    REQUIRE(compute_gradient(p,x).isApprox(expected, 1e-8));
}

TEST_CASE("compute_gradient throws if gradient_func is not provided", "[gradient]")
{
    NLPProblem p; p.cost_func=[](auto&x) -> double { return x.squaredNorm(); };
    VectorXd x(2); x<<1,2;
    REQUIRE(!p.hasGradient());
    REQUIRE_THROWS_AS(compute_gradient(p,x), std::runtime_error);
}

TEST_CASE("compute gradient for LS problem", "[gradient][ls]")
{
    // Objective: f(x) = F(x)ᵀ F(x) = 0.5 * ||r(x)||²
    // where r(x) = [x₀ + 2*x₁ - 2, x₀² + x₁² - 1]ᵀ
    //
    // Thus F(x) = sqrt(0.5) * r(x)

    LSProblem p;
    p.residual_func = [](const VectorXd& x) -> VectorXd {
        VectorXd F(2);
        F[0] = std::sqrt(0.5) * (x[0] + 2.0 * x[1] - 2.0);
        F[1] = std::sqrt(0.5) * (x[0] * x[0] + x[1] * x[1] - 1.0);
        return F;
    };

    p.gradient_residual_func = [](const VectorXd& x) -> MatrixXd {
        MatrixXd J(2, 2);
        J(0, 0) = std::sqrt(0.5) * 1.0;
        J(0, 1) = std::sqrt(0.5) * 2.0;
        J(1, 0) = std::sqrt(0.5) * 2.0 * x[0];
        J(1, 1) = std::sqrt(0.5) * 2.0 * x[1];
        J.transposeInPlace();
        return J;
    };

    const Eigen::Vector2d x(1.0, 2.0);
    const Eigen::Vector2d expected(11.0, 22.0);

    REQUIRE(p.hasGradientResidualFunc());
    REQUIRE(compute_gradient(p, x).isApprox(expected, 1e-8));

    const Eigen::Vector2d x_rand = 5.0 * Eigen::Vector2d::Random();
}

TEST_CASE("compute gradient for the same LS problem on 1000 random points", "[gradient][ls]")
{
    LSProblem p;
    p.residual_func = [](const VectorXd& x) -> VectorXd {
        VectorXd F(2);
        F[0] = std::sqrt(0.5) * (x[0] + 2.0 * x[1] - 2.0);
        F[1] = std::sqrt(0.5) * (x[0] * x[0] + x[1] * x[1] - 1.0);
        return F;
    };

    p.gradient_residual_func = [](const VectorXd& x) -> MatrixXd {
        MatrixXd J(2, 2);
        J(0, 0) = std::sqrt(0.5) * 1.0;
        J(0, 1) = std::sqrt(0.5) * 2.0;
        J(1, 0) = std::sqrt(0.5) * 2.0 * x[0];
        J(1, 1) = std::sqrt(0.5) * 2.0 * x[1];
        J.transposeInPlace();
        return J;
    };

    GradientFunc g_func = [](const VectorXd& x) -> VectorXd {
        const double r0 = x[0] + 2.0 * x[1] - 2.0;
        const double r1 = x[0] * x[0] + x[1] * x[1] - 1.0;

        Eigen::Vector2d analytical_grad;
        analytical_grad[0] = r0 + 2.0 * x[0] * r1;
        analytical_grad[1] = 2.0 * r0 + 2.0 * x[1] * r1;
        return analytical_grad;
    };

    REQUIRE(p.hasGradientResidualFunc());

    for (int i = 0; i < 1000; ++i) {
        const Eigen::Vector2d x = 10.0 * Eigen::Vector2d::Random();
        const Eigen::Vector2d expected = g_func(x);
        const Eigen::Vector2d computed = compute_gradient(p, x);
        REQUIRE(computed.isApprox(expected, 1e-8));
    }
}
// =============================================================================
//  QP
// =============================================================================
TEST_CASE("Unconstrained QP has the closed-form solution H x = -c", "[qp][unconstrained]")
{
    // min 0.5*(4*x0^2 + 2*x1^2) - 8*x0 - 6*x1
    // analytic solution: x = (2,3)
    QPProblem qp;
    qp.H = (MatrixXd(2,2) << 4,0, 0,2).finished(); //I use this advanced eigen initialization method to avoid constructing a temp matrix H just to assign qp.H
    qp.c = (VectorXd(2)   << -8,-6).finished();
    qp.x0 = VectorXd::Zero(2);

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);          // no constraints -> no_constraints_QP_solver
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2); expected << 2.0, 3.0;   // x = [2, 3]
    REQUIRE(r.x.isApprox(expected, 1e-9));
    REQUIRE(r.summary.converged);
    REQUIRE(r.lambda.size() == 0);
    REQUIRE(r.mhu.size() == 0);
}

TEST_CASE("Equality-constrained QP matches the KKT solution", "[qp][equality]")
{
    // min 0.5||x||^2  s.t.  x0 + x1 = 2  (written as A*x + b = 0 with b = -2)
    // analytic solution: x = (1,1), lambda from stationarity.
    QPProblem qp;
    qp.H = MatrixXd::Identity(2,2);
    qp.c = VectorXd::Zero(2);
    qp.A = (MatrixXd(1,2) << 1,1).finished();
    qp.b = (VectorXd(1)   << -2).finished();
    qp.x0 = VectorXd::Zero(2);

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());
    VectorXd expected(2); expected << 1.0, 1.0;
    REQUIRE(r.x.isApprox(expected, 1e-8));
    REQUIRE(r.summary.converged);
    REQUIRE(r.lambda.size() == qp.A.value().rows());
    REQUIRE(r.lambda(0) == Approx(1.0).margin(1e-6));   // stationarity: Hx+c = A^T*lambda => lambda*=1
    REQUIRE(r.summary.termination_reason == TerminationReason::DirectSolve);
    REQUIRE(r.mhu.size() == 0);

    // Check feasibility: A*x + b == 0
    if (qp.A.has_value() && qp.b.has_value()) {
        REQUIRE(((*qp.A) * r.x + *qp.b).minCoeff() <= 1e-6);
    }
}

TEST_CASE("Inequality-constrained QP: inactive constraints", "[qp][inequality]")
{
    // min 0.5*xᵀHx + cᵀx
    //
    // H = [4 1]
    //     [1 2]
    //
    // c = [-8, -3]ᵀ
    //
    // subject to Cx + d >= 0, where
    //
    // C = [ 1  1]
    //     [-1  2]
    //     [ 1  0]
    //
    // d = [-1, 2, 0]ᵀ
    //
    // The unconstrained minimizer is x = [13/7, 4/7]ᵀ, which already satisfies
    // all inequalities strictly. Therefore the constrained optimum is also
    // x = [13/7, 4/7]ᵀ and none of the constraints is active.

    QPProblem qp;
    qp.H = (MatrixXd(2,2) << 4, 1, 1, 2).finished();
    qp.c = (VectorXd(2) << -8,-3).finished();
    qp.C = (MatrixXd(3,2) <<  1, 1, -1, 2, 1, 0).finished();
    qp.d = (VectorXd(3) << -1, 2, 0).finished();

    qp.x0 = (VectorXd(2) << 1.0, 1.0).finished(); // strictly feasible

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2);
    expected << 13.0/7.0, 4.0/7.0;

    REQUIRE(r.x.isApprox(expected, 1e-4));
    REQUIRE(r.lambda.size() == 0);
    REQUIRE(r.mhu.size() == qp.C.value().rows());
    const VectorXd stat = qp.H * r.x + qp.c - qp.C.value().transpose() * r.mhu;   // [CNOEC] (6.16)
    REQUIRE(stat.norm() < 1e-5);                          // dual feasibility / stationarity
    REQUIRE(r.mhu.minCoeff() >= 0.0);                     // dual sign
    REQUIRE(((*qp.C) * r.x + *qp.d).dot(r.mhu) < 1e-5);   // complementarity

    // Check feasibility: C*x + d >= 0
    if (qp.C.has_value() && qp.d.has_value()) {
        REQUIRE(((*qp.C) * r.x + *qp.d).minCoeff() >= -1e-6);
    }
}

TEST_CASE("QP: an active inequality recovers the analytic multiplier", "[qp][inequality]")
{
    // min 0.5||x||^2 - 2x0 - 2x1  s.t.  2 - x0 - x1 >= 0  =>  x* = (1,1), mu* = 1
    QPProblem qp;
    qp.H = MatrixXd::Identity(2,2);
    qp.c = (VectorXd(2) << -2.0, -2.0).finished();
    qp.C = (MatrixXd(1,2) << -1.0, -1.0).finished();
    qp.d = (VectorXd(1) << 2.0).finished();
    qp.x0 = VectorXd::Zero(2);

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2); expected << 1.0, 1.0;
    REQUIRE(r.x.isApprox(expected, 1e-4));
    REQUIRE(r.mhu(0) == Approx(1.0).margin(1e-3));
}

TEST_CASE("QP: converged flag is reachable for inequality-constrained problems", "[qp][reporting]")
{
    for (int q : {1, 2, 3, 5}) {
        QPProblem qp;
        qp.H = MatrixXd::Identity(2,2);
        qp.c = VectorXd::Zero(2);
        MatrixXd C = MatrixXd::Zero(q, 2);
        for (int i = 0; i < q; ++i) C(i, i % 2) = 1.0;
        qp.C = C;
        qp.d = VectorXd::Constant(q, 10.0);   // all inactive at x* = 0
        qp.x0 = VectorXd::Zero(2);

        IPMSolverOptions ipm_options = ipm_opts();
        QPSolver s(ipm_options, qp);
        Result r;
        REQUIRE_NOTHROW(r = s.solve());

        CAPTURE(q, r.x, r.summary.converged);
        REQUIRE(r.x.norm() < 1e-6);
        REQUIRE(r.summary.converged);
    }
}

TEST_CASE("QP: unconstrained problem with x0 unset does not throw", "[qp][guards]")
{
    // No constraints -> phase-1 must be skipped (x0 defaults to zero), not fed
    // an empty-row auxiliary LP.
    QPProblem qp;
    qp.H = (MatrixXd(2,2) << 4, 0, 0, 2).finished();
    qp.c = (VectorXd(2) << -8, -6).finished();

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE(r.x.isApprox((VectorXd(2) << 2.0, 3.0).finished(), 1e-9));
    REQUIRE(r.summary.termination_reason == TerminationReason::DirectSolve);
}

TEST_CASE("SolverSummary::termination_reason is deterministic, never indeterminate",
          "[qp][reporting][regression]")
{
    // Default value must be defined, and every direct-solve path must set it
    // explicitly - not leave it reading whatever was on the stack.
    REQUIRE(SolverSummary{}.termination_reason == TerminationReason::MaxIterations);

    int reason = -1;
    for (int k = 0; k < 3; ++k) {
        QPProblem qp;
        qp.H = (MatrixXd(2,2) << 4, 0, 0, 2).finished();
        qp.c = (VectorXd(2) << -8, -6).finished();
        qp.x0 = VectorXd::Zero(2);

        IPMSolverOptions ipm_options = ipm_opts();
        QPSolver s(ipm_options, qp);
        Result r;
        REQUIRE_NOTHROW(r = s.solve());

        const int got = static_cast<int>(r.summary.termination_reason);
        if (k == 0) reason = got;
        CAPTURE(k, got, reason);
        REQUIRE(got == reason);
        REQUIRE(got == static_cast<int>(TerminationReason::DirectSolve));
    }
}

TEST_CASE("QP with a strictly interior optimum, solved through phase-1", "[qp][feasibility]")
{
    // min 0.5||x||^2 - x0 - x1  s.t. x >= 0  =>  x* = (1,1). x0 unset -> phase-1 supplies it.
    QPProblem qp;
    qp.H = MatrixXd::Identity(2,2);
    qp.c = (VectorXd(2) << -1.0, -1.0).finished();
    qp.C = MatrixXd::Identity(2,2);
    qp.d = VectorXd::Zero(2);

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE(r.x.isApprox((VectorXd(2) << 1.0, 1.0).finished(), 1e-5));
}

// =============================================================================
//  LP
// =============================================================================
TEST_CASE("LP: bounded linear program reaches the correct vertex", "[lp]")
{
    // min -x0 - x1  s.t.  -1 <= x <= 1  =>  x* = (1,1), cost* = -2
    LPProblem lp;
    lp.c = (VectorXd(2) << -1.0, -1.0).finished();
    lp.C = (MatrixXd(4,2) << 1,0, -1,0, 0,1, 0,-1).finished();
    lp.d = (VectorXd(4) << 1, 1, 1, 1).finished();
    lp.x0 = VectorXd::Zero(2);

    IPMSolverOptions ipm_options = ipm_opts();
    LPSolver s(ipm_options, lp);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE(r.x.isApprox((VectorXd(2) << 1.0, 1.0).finished(), 1e-5));
    REQUIRE(r.summary.final_cost == Approx(-2.0).margin(1e-5));
    REQUIRE(r.mhu.minCoeff() >= 0.0);
}

TEST_CASE("LP without any constraints is rejected", "[lp][guards]")
{
    LPProblem lp;
    lp.c = (VectorXd(2) << 1.0, 1.0).finished();

    IPMSolverOptions ipm_options = ipm_opts();
    LPSolver s(ipm_options, lp);
    REQUIRE_THROWS_AS(s.solve(), std::runtime_error);
}

// =============================================================================
//  Generalization Strategies
// =============================================================================

TEST_CASE("LineSearch: Armijo returns a step giving sufficient decrease", "[linesearch]") {
    CostFunc f = [](const VectorXd& x) -> double { return x.squaredNorm(); };   // grad = 2x
    VectorXd x(1); x << 1.0;
    VectorXd d(1); d << -1.0;                 // descent
    double gTd = (2*x).dot(d);                // = -2
    double a = compute_step_length(GlobalizationMethod::LineSearch, f, gTd, x, d);
    REQUIRE(a > 0.0);
    REQUIRE(f(x + a*d) <= f(x) + 1e-4 * a * gTd + 1e-12);   // Armijo holds at returned alpha
}

TEST_CASE("LineSearch: non-descent direction returns zero step", "[linesearch]") {
    CostFunc f = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    VectorXd x(1); x << 1.0;
    VectorXd d(1); d << +1.0;                 // ascent
    REQUIRE(compute_step_length(GlobalizationMethod::LineSearch, f, (2*x).dot(d), x, d) == 0.0);
}

TEST_CASE("LineSearch: full quadratic step accepted when alpha=1", "[linesearch]") {
    CostFunc f = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    VectorXd x(1); x << 2.0;
    VectorXd d = -x;                            // d = -2.0 (exact step to minimum at 0)
    double grad_dot_d = (2 * x).dot(d);        // Directional derivative: 2 * 2 * (-2) = -8.0

    double alpha = compute_step_length(GlobalizationMethod::LineSearch, f, grad_dot_d, x, d);

    REQUIRE(alpha == Catch::Approx(1.0));
}

TEST_CASE("Trust Region throws not implemented", "[trust_region]") {
    CostFunc f = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    VectorXd x(1); x << 1.0;
    VectorXd d(1); d << -1.0;
    REQUIRE_THROWS_AS(compute_step_length(GlobalizationMethod::TrustRegion, f, (2*x).dot(d), x, d), std::runtime_error);
}

TEST_CASE("Merit Function: L1 merit function decreases along search direction", "[linesearch][merit]") {
    CostFunc f = [](const VectorXd& x) -> double { return x[0] * x[0] + x[1] * x[1]; };
    EqualityConstraintFunc c_eq = [](const VectorXd& x) -> VectorXd { VectorXd c(1); c[0] = x[0] + x[1] - 2.0; return c; };
    InequalityConstraintFunc c_ineq = [](const VectorXd& x) -> VectorXd { VectorXd c(1); c[0] = x[0]; return c; };
    GradientFunc g_f = [](const VectorXd& x) -> Eigen::VectorXd { return 2.0 * x; };
    GradientInequalityConstraintFunc g_ineq = [](const VectorXd&) -> MatrixXd { MatrixXd J(2,1); J << 1.0, 0.0; return J; };

    VectorXd x(2); x << 0.5, 0.5;
    VectorXd d(2); d << 0.5, 0.5;
    Eigen::VectorXd sigma(1); sigma << 10.0;
    Eigen::VectorXd tau(1); tau << 10.0;

    double alpha = compute_step_length(
        GlobalizationMethod::LineSearch, f, c_eq, c_ineq, g_f, g_ineq, x, d, sigma, tau
    );
    REQUIRE(alpha > 0.0);
}

static EqualityConstraintFunc          no_eq   = [](const VectorXd&) -> VectorXd { return VectorXd::Zero(0); };
static InequalityConstraintFunc        no_ineq = [](const VectorXd&) -> VectorXd { return VectorXd::Zero(0); };
static GradientInequalityConstraintFunc no_grad_ineq =
    [](const VectorXd& x) -> MatrixXd { return MatrixXd::Zero(x.rows(), 0); };

TEST_CASE("merit: the -sigma|g| term enters with the correct sign and magnitude", "[merit][linesearch]")
{
    // f(x)=x0, g(x)=x0-1 => g(0)=-1. QP relation grad_g^T p=-g gives p=1.
    // D = grad_f^T p - sigma|g| = 1 - sigma.
    CostFunc f = [](const VectorXd& x) -> double { return x[0]; };
    GradientFunc grad_f = [](const VectorXd&) -> VectorXd { return VectorXd::Constant(1, 1.0); };
    EqualityConstraintFunc g = [](const VectorXd& x) -> VectorXd {
        return VectorXd::Constant(1, x[0] - 1.0); };

    VectorXd x = VectorXd::Zero(1);
    VectorXd p = VectorXd::Constant(1, 1.0);
    VectorXd tau0 = VectorXd::Zero(0);

    SECTION("sigma too small: not a descent direction for the merit") {
        VectorXd sigma = VectorXd::Constant(1, 0.5);
        const double a = compute_step_length(GlobalizationMethod::LineSearch,
                                             f, g, no_ineq, grad_f, no_grad_ineq,
                                             x, p, sigma, tau0);
        REQUIRE(a == 0.0);
    }
    SECTION("sigma large enough: descent, and the merit really does decrease") {
        VectorXd sigma = VectorXd::Constant(1, 2.0);
        const double a = compute_step_length(GlobalizationMethod::LineSearch,
                                             f, g, no_ineq, grad_f, no_grad_ineq,
                                             x, p, sigma, tau0);
        REQUIRE(a > 0.0);
        auto merit = [&](const VectorXd& z) { return f(z) + 2.0 * std::abs(g(z)(0)); };
        REQUIRE(merit(x + a * p) < merit(x));
    }
}

TEST_CASE("merit: a violated inequality contributes -tau * grad_h^T p", "[merit][linesearch]")
{
    // f(x)=x0, h(x)=x0-1 => h(0)=-1<0 (violated). QP relation grad_h^T p>=-h gives p=1.
    CostFunc f = [](const VectorXd& x) -> double { return x[0]; };
    GradientFunc grad_f = [](const VectorXd&) -> VectorXd { return VectorXd::Constant(1, 1.0); };
    InequalityConstraintFunc h = [](const VectorXd& x) -> VectorXd {
        return VectorXd::Constant(1, x[0] - 1.0); };
    GradientInequalityConstraintFunc grad_h = [](const VectorXd&) -> MatrixXd {
        return MatrixXd::Constant(1, 1, 1.0); };

    VectorXd x = VectorXd::Zero(1);
    VectorXd p = VectorXd::Constant(1, 1.0);
    VectorXd sigma0 = VectorXd::Zero(0);

    SECTION("tau too small -> rejected") {
        VectorXd tau = VectorXd::Constant(1, 0.5);
        REQUIRE(compute_step_length(GlobalizationMethod::LineSearch,
                                    f, no_eq, h, grad_f, grad_h, x, p, sigma0, tau) == 0.0);
    }
    SECTION("tau large enough -> accepted, merit decreases") {
        VectorXd tau = VectorXd::Constant(1, 2.0);
        const double a = compute_step_length(GlobalizationMethod::LineSearch,
                                             f, no_eq, h, grad_f, grad_h, x, p, sigma0, tau);
        REQUIRE(a > 0.0);
        auto merit = [&](const VectorXd& z) { return f(z) + 2.0 * std::max(0.0, -h(z)(0)); };
        REQUIRE(merit(x + a * p) < merit(x));
    }
}

TEST_CASE("merit: a satisfied inequality contributes nothing", "[merit][linesearch]")
{
    // h(x)=x0+1>0 at x=0, so (6.31) skips it; D is just grad_f^T p.
    CostFunc f = [](const VectorXd& x) -> double { return x[0]; };
    GradientFunc grad_f = [](const VectorXd&) -> VectorXd { return VectorXd::Constant(1, 1.0); };
    InequalityConstraintFunc h = [](const VectorXd& x) -> VectorXd {
        return VectorXd::Constant(1, x[0] + 1.0); };
    GradientInequalityConstraintFunc grad_h = [](const VectorXd&) -> MatrixXd {
        return MatrixXd::Constant(1, 1, 1.0); };

    VectorXd x = VectorXd::Zero(1);
    VectorXd p = VectorXd::Constant(1, -1.0);
    VectorXd sigma0 = VectorXd::Zero(0);
    VectorXd tau = VectorXd::Constant(1, 1000.0);   // huge, but must be inert

    const double a = compute_step_length(GlobalizationMethod::LineSearch,
                                         f, no_eq, h, grad_f, grad_h, x, p, sigma0, tau);
    REQUIRE(a > 0.0);
}

TEST_CASE("merit: the descent guarantee of [CNOEC] (6.35) holds for a real QP step", "[merit][sqp]")
{
    // If H>0, sigma>|lambda~| and tau>|mu~| then D(T1,p*) <= -p^T H p < 0.
    NLPProblem p = sqp_circle();
    const VectorXd x = p.x0;

    const VectorXd grad_f = p.gradient_func.value()(x);
    const MatrixXd grad_g = p.gradient_equality_constraint_func.value()(x);
    const VectorXd g_x    = p.equality_constraint_func.value()(x);

    QPProblem qp;
    qp.H  = MatrixXd::Identity(2,2);
    qp.c  = grad_f;
    qp.A  = grad_g.transpose();
    qp.b  = g_x;
    qp.x0 = VectorXd::Zero(2);

    IPMSolverOptions io = ipm_opts();
    QPSolver qs(io, qp);
    Result qr;
    REQUIRE_NOTHROW(qr = qs.solve());

    const VectorXd p_step = qr.x;
    REQUIRE(p_step.norm() > 1e-8);

    // Alg 6.3.1 step 5, with a margin so that ">" is strict.
    VectorXd sigma(1);
    sigma(0) = std::max(std::abs(qr.lambda(0)), (1.0 + std::abs(qr.lambda(0))) / 2.0) + 1e-3;
    VectorXd tau0 = VectorXd::Zero(0);

    const double a = compute_step_length(GlobalizationMethod::LineSearch,
                                         p.cost_func,
                                         p.equality_constraint_func.value(),
                                         no_ineq,
                                         p.gradient_func.value(),
                                         no_grad_ineq,
                                         x, p_step, sigma, tau0);
    REQUIRE(a > 0.0);

    auto merit = [&](const VectorXd& z) {
        return p.cost_func(z) + sigma(0) * std::abs(p.equality_constraint_func.value()(z)(0)); };
    REQUIRE(merit(x + a * p_step) < merit(x));
}

// =============================================================================
//  Unconstrained NLP solver
// =============================================================================
TEST_CASE("Exact Newton solves a quadratic in one step", "[solver][newton]")
{
    // QP problem with H = diag(2,3), c = [0,0]^T, optimum is obviously at x = [0,0]^T.
    NLPProblem p;
    p.x0 = (VectorXd(2) << 5.0, -4.0).finished();
    Eigen::Matrix2d H; 
    H << 2,0, 0,3;
    p.cost_func     = [H](const VectorXd& x) -> double { return 0.5 * x.transpose() * H * x; };
    p.gradient_func = [H](const VectorXd& x) -> VectorXd { return (H * x).eval(); };
    p.hessian_func  = [H](const VectorXd&) -> MatrixXd { return MatrixXd(H); };
    
    auto o = nlp_opts(DirectionMethod::ExactNewton);
    UnconstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE(r.x.norm() < 1e-8);
    REQUIRE(r.summary.iterations <= 2);

    //Unconstrained so no lamba and mhu should return
    REQUIRE(r.lambda.size() == 0);
    REQUIRE(r.mhu.size() == 0);
}

TEST_CASE("BFGS converges on a strictly convex quadratic", "[solver][bfgs]")
{
    // f(x) = 1/2 xᵀHx + cᵀx
    //
    // H = [4 1]
    //     [1 3]
    //
    // c = [-1, -2]ᵀ
    //
    // H is symmetric positive definite, so the objective has a unique global
    // minimizer at x* = -H⁻¹c = [1/11, 7/11]ᵀ.

    NLPProblem p;
    p.x0 = (VectorXd(2) << 3.0, -2.0).finished();
    Eigen::Matrix2d H;
    H << 4, 1, 1, 3;
    Eigen::Vector2d c;
    c << -1, -2;
    p.cost_func = [H,c](const VectorXd& x) -> double { return 0.5 * x.dot(H * x) + c.dot(x); };

    p.gradient_func = [H,c](const VectorXd& x) -> VectorXd { return H * x + c; };

    auto options = nlp_opts(DirectionMethod::BFGS, 100);
    UnconstrainedSolver s(options, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2);
    expected << 1.0/11.0, 7.0/11.0;

    REQUIRE(r.x.isApprox(expected, 1e-6));
    REQUIRE(std::abs(r.summary.final_cost - p.cost_func(expected)) < 1e-6);

    //Unconstrained so no lamba and mhu should return
    REQUIRE(r.lambda.size() == 0);
    REQUIRE(r.mhu.size() == 0);
}

TEST_CASE("Gauss-Newton converges on an overdetermined nonlinear least-squares problem (m != n)",
          "[solver][gauss_newton][regression]")
{
    // Solve min 0.5 * ||F(x)||^2
    //
    // Residual:
    //
    //   F(x) =
    //   [ x0 - 1        ]
    //   [ x1 - 2        ]
    //   [ x0 + x1 - 3   ]
    //
    // There are m = 3 residuals and n = 2 variables.
    //
    // The minimum is exactly at:
    //   x* = [1, 2]
    //
    // because all residuals become zero:
    //
    //   F([1,2]) = [0,0,0].

    LSProblem p;
    p.x0 = (VectorXd(2) << -2.0, 5.0).finished();

    p.residual_func = [](const VectorXd& x) -> Eigen::VectorXd {
        VectorXd r(3);
        r[0] = x[0] - 1.0;
        r[1] = x[1] - 2.0;
        r[2] = x[0] + x[1] - 3.0;
        return r;
    };

    p.gradient_residual_func = [](const VectorXd&) -> Eigen::MatrixXd {
        // Transposed gradient Jᵀ (n x m) of the residual vector F(x):
        //
        // Jᵀ =
        // [ 1 0 1 ]
        // [ 0 1 1 ]

        MatrixXd JT(2, 3);
        JT << 1.0, 0.0, 1.0,
              0.0, 1.0, 1.0;
        return JT;
    };

    auto o = nlp_opts(DirectionMethod::GradientDescent, 100); //Gradient descent is correct as the problem will be seen as an NLP
    UnconstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2);
    expected << 1.0, 2.0;
    REQUIRE(r.x.isApprox(expected, 1e-6));
    REQUIRE(r.summary.final_cost < 1e-12);
    REQUIRE(r.summary.final_gradient_norm < 1e-6);
    REQUIRE(r.summary.converged);

    //Unconstrained so no lamba and mhu should return
    REQUIRE(r.lambda.size() == 0);
    REQUIRE(r.mhu.size() == 0);
}

// =============================================================================
//  Convergence-flag / termination-reason semantics 
// =============================================================================
TEST_CASE("Gradient-norm termination reports GradientTolerance", "[solver][termination][regression]")
{
    // Regression for the convergence test scaling: a gradient-tolerance stop
    // must leave final ||g|| <= gradient_tolerance (not ||g||^2).
    NLPProblem p;
    p.x0 = (VectorXd(2) << 3.0, -2.0).finished();
    p.cost_func     = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd { return (2*x).eval(); };

    auto o = nlp_opts(DirectionMethod::GradientDescent);
    o.gradient_tolerance = 1e-6;
    UnconstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());
    REQUIRE(r.summary.converged);
    REQUIRE(r.summary.termination_reason == TerminationReason::GradientTolerance);
    REQUIRE(r.summary.final_gradient_norm <= 1e-6);
}

TEST_CASE("Function-tolerance check measures relative COST CHANGE", "[solver][termination][regression]")
{
    // Regression for the `- 000` typo: with a nonzero-optimum-cost problem,
    // a function-tolerance stop must be triggered by |f_new - f_old|, not |f_new|.
    // f(x) = (x-2)^2 + 10  (optimum cost = 10, far from 0).
    NLPProblem p;
    p.x0 = VectorXd::Constant(1, 5.0);
    p.cost_func     = [](const VectorXd& x) -> double { return (x[0]-2)*(x[0]-2) + 10.0; };
    p.gradient_func = [](const VectorXd& x) -> Eigen::VectorXd { VectorXd g(1); g[0]=2*(x[0]-2); return g; };

    auto o = nlp_opts(DirectionMethod::GradientDescent, 100000);
    UnconstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());
    REQUIRE(r.x[0] == Approx(2.0).margin(1e-3));   // must reach the true minimiser
}

TEST_CASE("Termination reasons are correctly reported for max iterations and tolerances", "[termination]") {
    UnconstrainedSolverOptions opts = nlp_opts(DirectionMethod::GradientDescent, 1);
    opts.gradient_tolerance = 1e-99;
    
    NLPProblem p;
    p.x0 = (VectorXd(2) << 10.0, 10.0).finished();
    p.cost_func = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> Eigen::VectorXd { return 2.0 * x; };

    UnconstrainedSolver solver(opts, p);
    Result r;
    REQUIRE_NOTHROW(r = solver.solve());
    REQUIRE(!r.summary.converged);
    REQUIRE(r.summary.termination_reason == TerminationReason::MaxIterations);
}

TEST_CASE("Unconstrained: a stalled run is not reported as converged", "[solver][termination]")
{
    // Sign-flipped gradient: every direction is ascent, so the solver must
    // burn all its iterations and report MaxIterations.
    NLPProblem p;
    p.x0 = VectorXd::Constant(1, 5.0);
    p.cost_func     = [](const VectorXd& x) -> double { return x[0]*x[0]; };
    p.gradient_func = [](const VectorXd& x) -> VectorXd { return (-2*x).eval(); };

    UnconstrainedSolverOptions o;
    o.direction_method = DirectionMethod::GradientDescent;
    o.max_iter = 25;

    UnconstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE_FALSE(r.summary.converged);
    REQUIRE(r.summary.termination_reason == TerminationReason::MaxIterations);
    REQUIRE(r.summary.final_gradient_norm > 1.0);
}

TEST_CASE("Unconstrained: default-constructed options are usable", "[api][regression]")
{
    UnconstrainedSolverOptions o;
    REQUIRE(o.max_iter > 0);
    REQUIRE(o.gradient_tolerance > 0.0);
    REQUIRE(o.step_tolerance > 0.0);
    REQUIRE(o.function_tolerance > 0.0);

    ConstrainedSolverOptions co;
    REQUIRE(co.constraint_tolerance > 0.0);
    REQUIRE(co.QP_subproblem_options.max_outer > 0);

    NLPProblem p;
    p.x0 = (VectorXd(2) << 1.0, 1.0).finished();
    p.cost_func     = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd { return (2*x).eval(); };

    UnconstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());
    REQUIRE(r.x.norm() < 1e-3);
}

// =============================================================================
//  SQP / ConstrainedSolver
// =============================================================================
static ConstrainedSolverOptions con_opts()
{
    ConstrainedSolverOptions o;
    o.direction_method     = DirectionMethod::BFGS;
    o.globalization_method = GlobalizationMethod::LineSearch;
    o.max_iter             = 100;
    o.gradient_tolerance   = 1e-7;
    o.step_tolerance       = 1e-12;
    o.function_tolerance   = 1e-14;
    o.constraint_tolerance = 1e-6;
    // o.logger is already initialized with null logger by default
    o.QP_subproblem_options = ipm_opts();
    return o;
}

TEST_CASE("SQP: quadratic objective with linear equality constraint", "[sqp][equality]")
{
    // min 0.5*(x0^2+x1^2)  s.t. x0+x1-2 = 0
    // Solution: the projection of the origin onto the line x0+x1=2 -> x*=[1,1], f(x*)=1
    NLPProblem p = sqp_linear_equality();

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2);
    expected << 1.0, 1.0;
    
    CAPTURE(r.x);
    CAPTURE(expected);
    CAPTURE(r.summary.converged);
    CAPTURE(r.summary.termination_reason);
    CAPTURE(r.summary.iterations);

    REQUIRE(r.x.isApprox(expected, 1e-4));
    REQUIRE(r.summary.converged == true);
    REQUIRE(r.summary.termination_reason == TerminationReason::GradientTolerance);
    REQUIRE(r.lambda.size() == 1);
    REQUIRE(r.lambda(0) == Approx(1.0).margin(1e-6));   // x*=(1,1): grad_f = A^T*lambda => lambda*=1
}

TEST_CASE("SQP: quadratic objective with active inequality constraint", "[sqp][inequality]")
{
    // min 0.5*((x0-2)^2+(x1-2)^2)  s.t. x0+x1<=2  (written as 2-x0-x1>=0)
    // Unconstrained solution (2,2) violates the constraint; constrained optimum
    // is the projection onto x0+x1=2 -> x*=[1,1]
    NLPProblem p = sqp_active_inequality();

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2);
    expected << 1.0, 1.0;

    CAPTURE(r.x);
    CAPTURE(expected);
    CAPTURE(r.summary.converged);
    CAPTURE(r.summary.termination_reason);
    CAPTURE(r.summary.iterations);

    REQUIRE(r.x.isApprox(expected, 1e-5));
    REQUIRE(r.summary.converged == true);
    REQUIRE(r.summary.termination_reason == TerminationReason::GradientTolerance);
    REQUIRE(r.mhu.size() == 1);
    REQUIRE(r.mhu(0) == Approx(1.0).margin(1e-3));   // mu = tau/s is only O(tau) accurate
    REQUIRE(r.mhu(0) >= 0.0);
}

TEST_CASE("SQP: nonlinear equality constraint circle", "[sqp][nonlinear]")
{
    // min x0+x1  s.t. x0^2+x1^2-1 = 0 (the unit circle)
    // Minimum of x0+x1 on the circle occurs at x* = [-sqrt(2)/2, -sqrt(2)/2]
    NLPProblem p = sqp_circle();

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2);
    expected << -std::sqrt(2)/2,
                -std::sqrt(2)/2;

    CAPTURE(r.x);
    CAPTURE(expected);
    CAPTURE(r.summary.converged);
    CAPTURE(r.summary.termination_reason);
    CAPTURE(r.summary.iterations);

    REQUIRE(r.x.isApprox(expected, 1e-4));
    REQUIRE(r.summary.converged == true);
    REQUIRE(r.summary.termination_reason == TerminationReason::GradientTolerance);
    REQUIRE(r.lambda.size() == 1);
    REQUIRE(r.lambda(0) == Approx(-std::sqrt(2.0) / 2.0).margin(1e-4));
}

TEST_CASE("SQP: nonlinear equality with active inequality", "[sqp][mixed]")
{
    // min (x0-1)^2+(x1-1)^2  s.t. x0^2+x1^2=1, x0>=0
    // Closest point on the first quadrant of the unit circle to (1,1) is
    // x* = [sqrt(2)/2, sqrt(2)/2]
    NLPProblem p = sqp_mixed();

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    VectorXd expected(2);
    expected << std::sqrt(2)/2,
                std::sqrt(2)/2;

    CAPTURE(r.x);
    CAPTURE(expected);
    CAPTURE(r.summary.converged);
    CAPTURE(r.summary.termination_reason);
    CAPTURE(r.summary.iterations);

    REQUIRE(r.x.isApprox(expected, 1e-4));
    REQUIRE(r.summary.converged == true);
    REQUIRE(r.summary.termination_reason == TerminationReason::GradientTolerance);
    REQUIRE(r.lambda(0) == Approx(1.0 - std::sqrt(2.0)).margin(1e-5));
    REQUIRE(r.mhu(0)    == Approx(0.0).margin(1e-6));
}

TEST_CASE("SQP: starting exactly ON an active inequality boundary", "[sqp][warmstart]")
{
    // h(x0) = 0 exactly: a log-barrier cannot start at s = 0, so p = 0 is NOT
    // usable and the code must fall through to phase-1.
    NLPProblem p = sqp_active_inequality();
    p.x0 = (VectorXd(2) << 1.0, 1.0).finished();

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE(r.x.isApprox((VectorXd(2) << 1.0, 1.0).finished(), 1e-6));
}

TEST_CASE("SQP: starting from an INFEASIBLE iterate (phase-1 path)", "[sqp][warmstart]")
{
    // h(x0) = -4 < 0: p = 0 is infeasible, the max-margin phase-1 LP must
    // supply the starting step.
    NLPProblem p = sqp_active_inequality();
    p.x0 = (VectorXd(2) << 3.0, 3.0).finished();

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE(r.x.isApprox((VectorXd(2) << 1.0, 1.0).finished(), 1e-5));
}

TEST_CASE("SQP: an unconstrained stationary point is NOT a converged KKT point", "[sqp][termination]")
{
    // At x0=(0,0), grad_f=0 => |grad_L^T p|=0, yet g(x0)=-2: the convergence
    // test must also require feasibility, not just the gradient condition.
    NLPProblem p = sqp_linear_equality();
    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r;
    REQUIRE_NOTHROW(r = s.solve());

    REQUIRE(r.summary.iterations > 0);
    REQUIRE(std::abs(r.x[0] + r.x[1] - 2.0) <= o.constraint_tolerance);
}

TEST_CASE("SQP: converged=true implies the constraints are actually satisfied", "[sqp][termination]")
{
    auto check = [](NLPProblem p) {
        auto o = con_opts();
        ConstrainedSolver s(o, p);
        Result r;
        try { r = s.solve(); } catch (const std::exception&) { return; }
        if (!r.summary.converged) return;
        VectorXd g = p.equality_constraint_func.has_value()
                   ? p.equality_constraint_func.value()(r.x) : VectorXd::Zero(0);
        VectorXd h = p.inequality_constraint_func.has_value()
                   ? p.inequality_constraint_func.value()(r.x) : VectorXd::Zero(0);
        if (g.size()) REQUIRE(g.lpNorm<Eigen::Infinity>() <= o.constraint_tolerance);
        if (h.size()) REQUIRE(h.minCoeff() >= -o.constraint_tolerance);
    };
    check(sqp_linear_equality());
    check(sqp_active_inequality());
    check(sqp_circle());
    check(sqp_mixed());
}

TEST_CASE("ConstrainedSolver rejects a constraint supplied without its Jacobian", "[guards]")
{
    SECTION("equality constraint without gradient") {
        NLPProblem p;
        p.x0 = VectorXd::Zero(2);
        p.cost_func     = [](const VectorXd& x) -> double { return x.squaredNorm(); };
        p.gradient_func = [](const VectorXd& x) -> VectorXd { return (2*x).eval(); };
        p.equality_constraint_func = [](const VectorXd& x) -> VectorXd {
            return VectorXd::Constant(1, x[0] - 1.0); };
        auto o = con_opts();
        REQUIRE_THROWS_AS(ConstrainedSolver(o, p), std::invalid_argument);
    }
    SECTION("inequality constraint without gradient") {
        NLPProblem p;
        p.x0 = VectorXd::Zero(2);
        p.cost_func     = [](const VectorXd& x) -> double { return x.squaredNorm(); };
        p.gradient_func = [](const VectorXd& x) -> VectorXd { return (2*x).eval(); };
        p.inequality_constraint_func = [](const VectorXd& x) -> VectorXd {
            return VectorXd::Constant(1, x[0]); };
        auto o = con_opts();
        REQUIRE_THROWS_AS(ConstrainedSolver(o, p), std::invalid_argument);
    }
}

// =============================================================================
//  Phase-1 / Feasibility
// =============================================================================

TEST_CASE("computeFeasiblePoint returns strictly feasible interior point for box constraints", "[lp][feasibility]") {
    VectorXd c = VectorXd::Zero(2);
    MatrixXd A = MatrixXd::Zero(0, 2);
    VectorXd b = VectorXd::Zero(0);
    MatrixXd C = MatrixXd::Identity(2, 2);
    VectorXd d(2); d << 1.0, 1.0; // Box constraints x_i + 1 >= 0

    IPMSolverOptions opts;
    VectorXd x_feas = LPSolver::computeFeasiblePoint(c, A, b, C, d, opts);

    VectorXd slack = C * x_feas + d;

    CAPTURE(slack);

    REQUIRE((slack.array() > 0.0).all());
}

TEST_CASE("phase-1: bounded box returns a well-centred interior point", "[lp][feasibility]")
{
    MatrixXd C(4,2); C << 1,0, -1,0, 0,1, 0,-1;
    VectorXd d(4);   d << 1, 1, 1, 1;                  // -1 <= x <= 1
    VectorXd c = VectorXd::Zero(2);
    MatrixXd A = MatrixXd::Zero(0,2);
    VectorXd b = VectorXd::Zero(0);

    IPMSolverOptions o = ipm_opts();
    VectorXd x = LPSolver::computeFeasiblePoint(c, A, b, C, d, o);
    const VectorXd slack = C * x + d;

    CAPTURE(x, slack);
    REQUIRE((slack.array() > 0.0).all());
    REQUIRE(slack.minCoeff() > 1e-3);
    REQUIRE(x.norm() < 10.0);
}

TEST_CASE("phase-1: honours equality constraints while staying interior", "[lp][feasibility]")
{
    MatrixXd C(4,2); C << 1,0, -1,0, 0,1, 0,-1;
    VectorXd d(4);   d << 1, 1, 1, 1;
    MatrixXd A(1,2); A << 1, 1;
    VectorXd b(1);   b << 0.0;                         // x0 + x1 = 0
    VectorXd c = VectorXd::Zero(2);

    IPMSolverOptions o = ipm_opts();
    VectorXd x = LPSolver::computeFeasiblePoint(c, A, b, C, d, o);

    CAPTURE(x, (A*x + b).norm(), (C*x + d).minCoeff());
    REQUIRE((A * x + b).norm() < 1e-8);
    REQUIRE((C * x + d).minCoeff() > 1e-3);
}

TEST_CASE("phase-1: touching the boundary is not accepted as interior", "[lp][feasibility]")
{
    // x >= 0 with d = 0. A naive min(1^T s) phase-1 can return slack ~1e-14,
    // passing a naive >0 test but leaving a barrier Hessian of order 1e28.
    MatrixXd C = MatrixXd::Identity(2,2);
    VectorXd d = VectorXd::Zero(2);
    VectorXd c = VectorXd::Zero(2);
    MatrixXd A = MatrixXd::Zero(0,2);
    VectorXd b = VectorXd::Zero(0);

    IPMSolverOptions o = ipm_opts();
    VectorXd x = LPSolver::computeFeasiblePoint(c, A, b, C, d, o);

    CAPTURE(x, (C*x + d).minCoeff());
    REQUIRE((C * x + d).minCoeff() > 1e-6);
}

TEST_CASE("phase-1: unbounded feasible set does not fling the point to the horizon",
          "[lp][feasibility][known-bug]")
{
    // On x >= 0 the max-margin LP caps t at 1, but nothing bounds x along the
    // cone's recession directions - see ANALYSIS_REPORT_V7.md S2.1.
    MatrixXd C = MatrixXd::Identity(2,2);
    VectorXd d = VectorXd::Zero(2);
    VectorXd c = VectorXd::Zero(2);
    MatrixXd A = MatrixXd::Zero(0,2);
    VectorXd b = VectorXd::Zero(0);

    IPMSolverOptions o = ipm_opts();
    VectorXd x = LPSolver::computeFeasiblePoint(c, A, b, C, d, o);

    CAPTURE(x, x.norm());
    REQUIRE(x.norm() < 1e3);
}

TEST_CASE("phase-1: empty interior and infeasible sets are both rejected", "[lp][feasibility]")
{
    VectorXd c = VectorXd::Zero(2);
    MatrixXd A = MatrixXd::Zero(0,2);
    VectorXd b = VectorXd::Zero(0);
    IPMSolverOptions o = ipm_opts();

    SECTION("empty interior: x0 >= 0 and -x0 >= 0 force x0 == 0") {
        MatrixXd C(2,2); C << 1,0, -1,0;
        VectorXd d(2);   d << 0, 0;
        REQUIRE_THROWS_AS(LPSolver::computeFeasiblePoint(c, A, b, C, d, o), std::runtime_error);
    }
    SECTION("infeasible: x0 >= 1 and x0 <= -1") {
        MatrixXd C(2,2); C << 1,0, -1,0;
        VectorXd d(2);   d << -1, -1;
        REQUIRE_THROWS_AS(LPSolver::computeFeasiblePoint(c, A, b, C, d, o), std::runtime_error);
    }
}

