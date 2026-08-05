#pragma once

#include <Eigen/Dense> // For Eigen::VectorXd and Eigen::MatrixXd
#include <Eigen/Cholesky> // For LLT and LDLT
#include "solver_config.hpp"

namespace furiaopt{

class DirectionStrategy {

protected:
    std::reference_wrapper<const NLPProblem> problem_;

    //Not used by gradient descent, but used by all other class that inherit from DirectionStrategy, so we can just put them here to avoid code duplication.
    Eigen::LLT<Eigen::MatrixXd> llt_;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
    
public:
    DirectionStrategy(const NLPProblem& problem) : problem_(std::cref(problem)){};
    virtual Eigen::VectorXd getDirection(const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i) = 0;
    virtual ~DirectionStrategy() = default;
};

class GradientDescentDirection : public DirectionStrategy {
public:
    GradientDescentDirection(const NLPProblem& problem) : DirectionStrategy(problem){};

    Eigen::VectorXd getDirection(const Eigen::VectorXd& gradient, const Eigen::VectorXd&) override {
        return -gradient;
    };
};

class BFGSHessianApproximation {
private:
    std::reference_wrapper<const NLPProblem> problem_;
    Eigen::MatrixXd H_k_; // Approximation of the Hessian at iteration k+1
    bool initialized_ = false; // Flag to check if H_k_plus_1_ has been initialized
    Eigen::VectorXd x_k_; // Previous iterate
    Eigen::VectorXd g_k_; // Gradient at previous iterate
    double gamma_ = 0.2; // Powell Damping factor for BFGS method
public:
    BFGSHessianApproximation(const NLPProblem& problem) : problem_(std::cref(problem))
    {
        //Here we can understand dimensions of the problem and initialize H_k_plus_1_, x_k_ and g_k_ accordingly.
        int problem_size = problem.x0.size();
        H_k_ = Eigen::MatrixXd::Identity(problem_size, problem_size); // Initial Hessian approximation
    };

    Eigen::MatrixXd getApproximateHessian(const Eigen::VectorXd& g_k_plus_1, const Eigen::VectorXd& x_k_plus_1);

    //This is needed for the consistency of the BFGS update when constraints are present.
    // y = Dlagrangian(x_k_plus_1, lambda_k_plus_1, mhu_k_plus_1) - Dlagrangian(x_k, lambda_k_plus_1, mhu_k_plus_1)
    // when we call getApproximateHessian the g_k_plus_1 passed is the Dlagrangian(x_k, lambda_k, mhu_k) 
    // so after updating lambda_k and mhu_k to lambda_k_plus_1 and mhu_k_plus_1 we need to update g_k to be consistent with the new multipliers.
    void setPreviousGradient(const Eigen::VectorXd& g_k) { g_k_ = g_k; } };

class BFGSDirection : public DirectionStrategy {

    BFGSHessianApproximation hessian_approximation_;

public:
    BFGSDirection(const NLPProblem& problem) : DirectionStrategy(problem), hessian_approximation_(problem) {};
    
    Eigen::VectorXd getDirection(const Eigen::VectorXd& g_k_plus_1, const Eigen::VectorXd& x_k_plus_1) override;
};

class ExactHessianDirection : public DirectionStrategy {

public:
    ExactHessianDirection(const NLPProblem& problem) : DirectionStrategy(problem){
        if (!problem.hasHessian()) {
            throw std::invalid_argument("Hessian function must be provided for ExactHessianDirection");
        };
    };

    Eigen::VectorXd getDirection(const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i) override;
    
};

class GaussNewtonHessianApproximation {
    std::reference_wrapper<const LSProblem> problem_;
    double sigma_ = 1e-10; // Damping factor for Gauss-Newton method

public: 
    GaussNewtonHessianApproximation(const LSProblem& problem) : problem_(std::cref(problem)){};

    Eigen::MatrixXd getApproximateHessian(const Eigen::VectorXd& x_i);
};

class GaussNewtonDirection {

private:
    GaussNewtonHessianApproximation hessian_approximation_;
    Eigen::LLT<Eigen::MatrixXd> llt_;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
public:
    GaussNewtonDirection(const LSProblem& problem): hessian_approximation_(problem) {};

    Eigen::VectorXd getDirection(const Eigen::VectorXd& gradient, const Eigen::VectorXd& x_i);
};

}