#pragma once
#include <Eigen/Dense>
#include <optional>
#include <memory>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>
#include "types.hpp"

namespace furiaopt
{

enum class DirectionMethod {
    GradientDescent,
    BFGS,
    ExactNewton
};

enum class GlobalizationMethod {
    LineSearch,
    TrustRegion
};
struct UnconstrainedSolverOptions {
    DirectionMethod direction_method = DirectionMethod::GradientDescent;
    GlobalizationMethod globalization_method = GlobalizationMethod::LineSearch;
    int max_iter = 1e4;
    double gradient_tolerance = 1e-6;
    double step_tolerance = 1e-10;
    double function_tolerance = 1e-12;
    std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>());
};

struct IPMSolverOptions {
    double tau_initial = 1.0;
    double tau_factor  = 0.2;
    int    max_outer   = 40;
    int    max_inner   = 50;
    double ipm_tol     = 1e-8;
    std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>());
};

struct ConstrainedSolverOptions: UnconstrainedSolverOptions {
    IPMSolverOptions QP_subproblem_options;
    double constraint_tolerance = 1e-6;
};

struct LinearConstraints {

    std::optional<Eigen::MatrixXd> A; // Coefficients for linear equality constraints, dimension (p x n)
    std::optional<Eigen::VectorXd> b; // Constants for linear equality constraints, dimension (p x 1)
    std::optional<Eigen::MatrixXd> C; // Coefficients for linear inequality constraints, dimension (q x n)
    std::optional<Eigen::VectorXd> d; // Constants for linear inequality constraints, dimension (q x 1)

    bool hasEqualityConstraints() const { return A.has_value() && b.has_value() && A.value().rows() > 0 && b.value().rows() > 0; }
    bool hasInequalityConstraints() const { return C.has_value() && d.has_value() && C.value().rows() > 0 && d.value().rows() > 0; }
};

// For Linear Programming: min f(x) = c^T*x
//                         s.t Ax + b = 0   (equality constraints)
//                         Cx + d >= 0      (inequality constraints)
struct LPProblem: public LinearConstraints{
    std::optional<Eigen::VectorXd> x0;
    Eigen::VectorXd c; // Linear term, dimension (n x 1)

};

// For Quadratic Programming: min f(x) = 0.5 * x'Hx + c'x
//                         s.t Ax + b = 0   (equality constraints)
//                         Cx + d >= 0      (inequality constraints)
struct QPProblem: public LinearConstraints {
    std::optional<Eigen::VectorXd> x0;
    Eigen::MatrixXd H; // Quadratic term, dimension (n x n)
    Eigen::VectorXd c; // Linear term, dimension (n x 1)
};

struct NonLinearConstraints {
    // equality constraints: g(x) = 0
    // Notice linear equality constrainst are rapresented as g(x) = Ax + b = 0
    std::optional<EqualityConstraintFunc> equality_constraint_func;
    std::optional<GradientEqualityConstraintFunc> gradient_equality_constraint_func;

    // Inequality constraints: h(x) >= 0
    // Notice linear Inequality constrainst are rapresented as h(x) = Cx + d >= 0
    std::optional<InequalityConstraintFunc> inequality_constraint_func;
    std::optional<GradientInequalityConstraintFunc> gradient_inequality_constraint_func;

    bool hasEqualityConstraints() const { return equality_constraint_func.has_value(); }
    bool hasInequalityConstraints() const { return inequality_constraint_func.has_value(); }
};

//Use it for Gauss-Newton methods. The cost function is defined as f(x) = F(x)^T*F(x), where F(x) is a column vector of functions.
struct LSProblem : public NonLinearConstraints {
    Eigen::VectorXd x0;
    ResidualFunc residual_func;
    std::optional<GradientResidualFunc> gradient_residual_func;

    bool hasGradientResidualFunc() const { return gradient_residual_func.has_value(); }
};

//Use it for Newton/Quasi-Newton methods. where the cost function is a generic non linear function f(x).
struct NLPProblem : public NonLinearConstraints {
    
    Eigen::VectorXd x0;
    CostFunc cost_func;

    std::optional<GradientFunc> gradient_func;
    std::optional<HessianFunc> hessian_func;

    bool hasGradient() const { return gradient_func.has_value(); }
    bool hasHessian() const { return hessian_func.has_value(); }

};

enum class TerminationReason {
    MaxIterations,
    GradientTolerance,
    StepTolerance,
    FunctionTolerance,
    DirectSolve
};

struct SolverSummary {
    int iterations = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    double final_gradient_norm = 0.0;
    bool converged = false;
    TerminationReason termination_reason = TerminationReason::MaxIterations;
};

struct Result {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(0); //It's dimension is n
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(0); //It's dimension is p, the number of equality constraints
    Eigen::VectorXd mhu = Eigen::VectorXd::Zero(0); //It's dimension is q, the number of inequality constraints
    SolverSummary summary;
};

}