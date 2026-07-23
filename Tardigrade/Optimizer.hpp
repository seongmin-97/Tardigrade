#pragma once
#include <iostream>
#include <vector>

#include "Autograd.hpp"
#include "Tensor.hpp"

namespace tardigrade
{
namespace optimizer
{
/**
 * @brief Abstract base class for optimizers.
 *
 * Manages network parameter updates using their computed gradients via Autograd.
 */
class Optimizer
{
protected:
    std::vector<Tensor> m_parameters; ///< Registered weight parameter Tensors
    double m_learningRate;            ///< Learning rate

public:
    /**
     * @brief Construct a new Optimizer object.
     * @param learningRate The learning rate.
     */
    Optimizer(double learningRate) : m_learningRate(learningRate) {}
    virtual ~Optimizer() = default;

    /**
     * @brief Registers parameter Tensors to the optimizer.
     * @param params Vector of parameter Tensors.
     */
    void AddParameters(const std::vector<Tensor> &params);

    /**
     * @brief Performs a single parameter update step.
     */
    virtual void Step() = 0;

    /**
     * @brief Resets the gradients of all registered parameters to zero.
     */
    virtual void ZeroGrad();
};

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 */
class SGD : public Optimizer
{
public:
    SGD(double learningRate);

    void Step() override;
};

/**
 * @brief Adaptive Moment Estimation (Adam) optimizer.
 */
class Adam : public Optimizer
{
private:
    double m_beta1;   ///< Exponential decay rate for the 1st moment estimates
    double m_beta2;   ///< Exponential decay rate for the 2nd moment estimates
    double m_epsilon; ///< Small constant for numerical stability
    int m_t;          ///< Time step index

    std::vector<Tensor> m_m; ///< 1st moment vectors
    std::vector<Tensor> m_v; ///< 2nd moment vectors
    bool m_initialized;      ///< Initialization status flag

    /**
     * @brief Initializes first and second moment tensors for registered parameters.
     */
    void InitializeMoments();

public:
    Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void Step() override;
};
} // namespace optimizer
} // namespace tardigrade
