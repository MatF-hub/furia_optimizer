#include "solvers/qp_solver.hpp"
#include "config_loader.hpp"
#include "example_cli.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

int main(int argc, char** argv)
{
    // Import config and create logger
    auto args = furiaopt::parse_example_args(argc, argv);
    furiaopt::IPMSolverOptions options =
        furiaopt::load_ipm_solver_options(args.config_dir + "/config.json", args.logs_output_dir);

    options.logger->info("=================================================");
    options.logger->info("2D Trajectory Smoothing Optimization Pipeline Init");
    options.logger->info("=================================================");

    // 100 vars: 50 X + 50 Y
    const int points_per_dim = 50;
    const int N = points_per_dim - 1; // 49 intervals
    const int num_vars = 2 * points_per_dim; // 100 total decision variables

    options.logger->info("Total optimization variables allocated: {} ({} points for X, {} points for Y)",
                 num_vars, points_per_dim, points_per_dim);

    furiaopt::QPProblem problem;

    // Discrete acceleration operator (1D), reused for X and Y
    Eigen::MatrixXd D_1d = Eigen::MatrixXd::Zero(N - 1, points_per_dim);
    for (int i = 0; i < N - 1; ++i) {
        D_1d(i, i)     = 1.0;   // coordinate_{k-1}
        D_1d(i, i + 1) = -2.0;  // -2 * coordinate_k
        D_1d(i, i + 2) = 1.0;   // coordinate_{k+1}
    }

    // Standard formulation: 1/2 * u^T * H * u where H is block diagonal
    Eigen::MatrixXd H_1d = 2.0 * D_1d.transpose() * D_1d;

    problem.H = Eigen::MatrixXd::Zero(num_vars, num_vars);
    problem.H.block(0, 0, points_per_dim, points_per_dim) = H_1d;                     // X block
    problem.H.block(points_per_dim, points_per_dim, points_per_dim, points_per_dim) = H_1d; // Y block
    problem.c = Eigen::VectorXd::Zero(num_vars);

    // Equality Constraints: start, end, plus two unevenly-spaced pass-through points
    // chosen off the straight line (else the smoother would just return that line).
    struct EqualityAnchor { int index; double x_val; double y_val; };
    std::vector<EqualityAnchor> eq_anchors = {
        {0,   0.0, 0.0},
        {10, 10.0, 6.5},
        {30,  5.0, 1.5},
        {N,  10.0, 10.0},
    };

    const int num_eq_rows = 2 * static_cast<int>(eq_anchors.size());
    Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(num_eq_rows, num_vars);
    Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(num_eq_rows);

    int eq_row = 0;
    for (const auto& anchor : eq_anchors) {
        // X constraint: 1 * u[index] = x_val  =>  1 * u[index] - x_val = 0
        A_eq(eq_row, anchor.index) = 1.0;
        b_eq(eq_row) = -anchor.x_val;
        eq_row++;

        // Y constraint: 1 * u[points_per_dim + index] = y_val
        A_eq(eq_row, points_per_dim + anchor.index) = 1.0;
        b_eq(eq_row) = -anchor.y_val;
        eq_row++;
    }
    problem.A = A_eq;
    problem.b = b_eq;

    // Inequality Constraints: a straight corridor around the start->end line, noticeably
    // uneven left/right so the pass-through points above are close to one edge.
    const Eigen::Vector2d traj_start(eq_anchors.front().x_val, eq_anchors.front().y_val);
    const Eigen::Vector2d traj_end(eq_anchors.back().x_val, eq_anchors.back().y_val);
    const Eigen::Vector2d dir = (traj_end - traj_start).normalized();
    const Eigen::Vector2d normal(-dir.y(), dir.x());
    const double w_left = 1.0;
    const double w_right = 3.0;

    // Band is relative to traj_start, not the world origin: normal.(x_k - traj_start)
    const double offset = normal.dot(traj_start);

    const int num_ineq_rows = 2 * points_per_dim;
    Eigen::MatrixXd C_ineq = Eigen::MatrixXd::Zero(num_ineq_rows, num_vars);
    Eigen::VectorXd d_ineq = Eigen::VectorXd::Zero(num_ineq_rows);

    for (int k = 0; k < points_per_dim; ++k) {
        // normal.(x_k - traj_start) <= w_left
        C_ineq(2 * k, k) = -normal.x();
        C_ineq(2 * k, points_per_dim + k) = -normal.y();
        d_ineq(2 * k) = w_left + offset;

        // normal.(x_k - traj_start) >= -w_right
        C_ineq(2 * k + 1, k) = normal.x();
        C_ineq(2 * k + 1, points_per_dim + k) = normal.y();
        d_ineq(2 * k + 1) = w_right - offset;
    }
    problem.C = C_ineq;
    problem.d = d_ineq;

    std::ostringstream anchors_str;
    for (const auto& a : eq_anchors)
        anchors_str << "(" << a.index << "," << a.x_val << "," << a.y_val << "),";
    options.logger->info("GEOMETRY points_per_dim={} anchors={} corridor=({},{},{},{})",
                          points_per_dim, anchors_str.str(),
                          normal.x(), normal.y(), w_left, w_right);

    // No x0 set: QPSolver bootstraps a feasible one itself (computeFeasiblePoint)

    // Initialize Solver Loop and Execute Run
    furiaopt::QPSolver solver(options, problem);
    auto t0 = std::chrono::steady_clock::now();
    furiaopt::Result result = solver.solve();
    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    options.logger->info("SOLVE elapsed_ms={:.4f}", elapsed_ms);

    // Stream Out Final 2D Mapping Profile
    std::cout << "=====================================================================\n";
    std::cout << "          OPTIMIZED 2D SMOOTHED TRAJECTORY PROFILE (100 VARS)        \n";
    std::cout << "=====================================================================\n";
    std::cout << "Index\tX Coordinate\tY Coordinate\tActive Status Notes\n";
    std::cout << "---------------------------------------------------------------------\n";
    for (int i = 0; i < points_per_dim; ++i) {
        double x_res = result.x[i];
        double y_res = result.x[points_per_dim + i];

        std::cout << i << "\t" << x_res << "\t\t" << y_res;

        if (i == 0)  std::cout << "\t[EQ: Start]";
        if (i == 10) std::cout << "\t[EQ: Pass-through]";
        if (i == 30) std::cout << "\t[EQ: Pass-through]";
        if (i == N)  std::cout << "\t[EQ: End]";
        std::cout << "\n";
    }
    std::cout << "=====================================================================\n";
    std::cout << "Optimization Convergence Status: " << (result.summary.converged ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "Minimized Acceleration Profile Cost Matrix: " << result.summary.final_cost << "\n";
    std::cout << "Iterations: " << result.summary.iterations << ", Elapsed: " << elapsed_ms << " ms\n";
    std::cout << "=====================================================================\n";

    return 0;
}
