#include "solvers/feasibility.hpp"
#include "solvers/qp_solver.hpp"

namespace furiaopt{

Eigen::VectorXd computeFeasiblePoint(
    const Eigen::VectorXd& c,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& d,
    const IPMSolverOptions& options)
{
    // If initialization is not provided use auxiliary problem to get initialization.
    // Max-margin phase-1 initializer that maximizes scalar distance to boundary (t) + regularization term 
    // to keep, dimensionaly as small as possible the initial guess x.
    // min eps/2 * ||x||^2 - t
    // s.t A*x + b = 0
    //     C*x + d - t >= 0
    //     t <= 1
    QPProblem aux_problem;
    const double eps = 1e-8;

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

    aux_problem.H = Eigen::MatrixXd::Zero(nx + nt, nx + nt);
    aux_problem.H.topLeftCorner(nx, nx) = eps * Eigen::MatrixXd::Identity(nx, nx);

    QPSolver solver(options, aux_problem);

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
