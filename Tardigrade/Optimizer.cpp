#include "Optimizer.hpp"
#include <stdexcept>
#include <cmath>

using namespace tardigrade;
using namespace tardigrade::optimizer;

// ------------------------------------------------------------
// SGD Implementation
// ------------------------------------------------------------
SGD::SGD(double learningRate) : Optimizer(learningRate) {}

/**
 * @brief Performs a single parameter update step using SGD.
 */
void SGD::Step() 
{
    for (auto& paramPair : m_parameters) 
    {
        Tensor* weight = paramPair.first;
        Tensor* grad = paramPair.second;
            
        // W = W - lr * grad
        const size_t size = weight->size();
        for (size_t i = 0; i < size; ++i) 
        {
            (*weight)[i] -= m_learningRate * (*grad)[i];
        }
    }
}

/**
 * @brief Resets the gradients of all registered parameters to zero.
 */
void SGD::ZeroGrad() 
{
    for (auto& paramPair : m_parameters) 
    {
        Tensor* grad = paramPair.second;
        const size_t size = grad->size();
        for (size_t i = 0; i < size; ++i) 
        {
            (*grad)[i] = 0.0;
        }
    }
}

// ------------------------------------------------------------
// Adam Implementation
// ------------------------------------------------------------
Adam::Adam(double learningRate, double beta1, double beta2, double epsilon)
        : Optimizer(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_t(0), m_initialized(false) {}

void Adam::InitializeMoments() 
{
    if (m_initialized) 
        return;
        
    m_m.clear();
    m_v.clear();
    m_m.reserve(m_parameters.size());
    m_v.reserve(m_parameters.size());

    for (const auto& paramPair : m_parameters) 
    {
        Tensor* weight = paramPair.first;
        m_m.emplace_back(Tensor(weight->shape()));
        m_v.emplace_back(Tensor(weight->shape()));
    }
    m_initialized = true;
}

/**
 * @brief Performs a single parameter update step using Adam.
 */
void Adam::Step() 
{
    if (!m_initialized) InitializeMoments();
        
    m_t++;
        
    // Compute bias corrections
    double correction1 = 1.0 - std::pow(m_beta1, m_t);
    double correction2 = 1.0 - std::pow(m_beta2, m_t);

    for (size_t i = 0; i < m_parameters.size(); ++i) 
    {
        Tensor* weight = m_parameters[i].first;
        Tensor* grad = m_parameters[i].second;
            
        Tensor& m = m_m[i];
        Tensor& v = m_v[i];
            
        const size_t size = weight->size();
        for (size_t j = 0; j < size; ++j) 
        {
            double g = (*grad)[j];
                
            // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g
            m[j] = m_beta1 * m[j] + (1.0 - m_beta1) * g;
                
            // Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            v[j] = m_beta2 * v[j] + (1.0 - m_beta2) * (g * g);
                
            // Compute bias-corrected first and second moment estimates
            double m_hat = m[j] / correction1;
            double v_hat = v[j] / correction2;
                
            // Update parameters: W_t = W_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
            (*weight)[j] -= m_learningRate * m_hat / (std::sqrt(v_hat) + m_epsilon);
        }
    }
}
/**
 * @brief Resets the gradients of all registered parameters to zero.
 */
void Adam::ZeroGrad() 
{
    for (auto& paramPair : m_parameters) 
    {
        Tensor* grad = paramPair.second;
        const size_t size = grad->size();
        for (size_t i = 0; i < size; ++i) 
        {
            (*grad)[i] = 0.0;
        }
    }
}
