#pragma once
#include <utility>
#include <vector>

#include "Tensor.hpp"

namespace tardigrade {
namespace optimizer {
/**
 * @brief Pair of pointers pointing to a weight tensor and its corresponding
 * gradient tensor.
 */
using ParamGradPair = std::pair<Tensor *, Tensor *>;

/**
 * @brief Abstract base class for optimizers.
 *
 * Manages network parameter updates using their computed gradients.
 */
class Optimizer {
protected:
  std::vector<ParamGradPair>
      m_parameters;      ///< Registered parameters (weight, gradient) pairs
  double m_learningRate; ///< Learning rate

public:
  /**
   * @brief Construct a new Optimizer object.
   * @param learningRate The learning rate.
   */
  Optimizer(double learningRate) : m_learningRate(learningRate) {}
  virtual ~Optimizer() = default;

  /**
   * @brief Registers parameter and gradient tensor pairs to the optimizer.
   * @param params Vector of ParamGradPair pointers.
   */
  void AddParameters(const std::vector<ParamGradPair> &params) {
    for (const auto &pair : params) {
      if (pair.first->shape() != pair.second->shape())
        throw std::runtime_error("Parameter and gradient shape mismatch.");
    }
    m_parameters.insert(m_parameters.end(), params.begin(), params.end());
  }

  /**
   * @brief Performs a single parameter update step.
   */
  virtual void Step() = 0;

  /**
   * @brief Resets the gradients of all registered parameters to zero.
   */
  virtual void ZeroGrad() = 0;
};

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 * @note
 * Mathematical formula:
 * \f[
 * W_t = W_{t-1} - \eta \nabla L(W_{t-1})
 * \f]
 * where \f$ \eta \f$ is the learning rate and \f$ \nabla L \f$ is the gradient.
 */
class SGD : public Optimizer {
public:
  SGD(double learningRate);

  void Step() override;
  void ZeroGrad() override;
};

/**
 * @brief Adaptive Moment Estimation (Adam) optimizer.
 * @note
 * Mathematical formulas:
 * \f[
 * m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
 * \f]
 * \f[
 * v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
 * \f]
 * \f[
 * \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
 * \f]
 * \f[
 * \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
 * \f]
 * \f[
 * W_t = W_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
 * \f]
 */
class Adam : public Optimizer {
private:
  double m_beta1;   ///< Exponential decay rate for the 1st moment estimates
  double m_beta2;   ///< Exponential decay rate for the 2nd moment estimates
  double m_epsilon; ///< Small constant for numerical stability
  int m_t;          ///< Time step index

  std::vector<Tensor> m_m; ///< 1st moment vectors
  std::vector<Tensor> m_v; ///< 2nd moment vectors
  bool m_initialized;      ///< Initialization status flag

  /**
   * @brief Initializes first and second moment tensors for registered
   * parameters.
   */
  void InitializeMoments();

public:
  /**
   * @brief Construct a new Adam object.
   * @param learningRate The learning rate.
   * @param beta1 Decay rate for first moment.
   * @param beta2 Decay rate for second moment.
   * @param epsilon Numerical stability constant.
   */
  Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999,
       double epsilon = 1e-8);

  void Step() override;
  void ZeroGrad() override;
};

} // namespace optimizer
} // namespace tardigrade
