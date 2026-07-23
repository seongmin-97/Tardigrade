#include "Optimizer.hpp"
#include <cmath>
#include <stdexcept>

using namespace tardigrade;
using namespace tardigrade::optimizer;

void Optimizer::AddParameters(const std::vector<Tensor> &params)
{
    for (const auto &p : params)
    {
        if (!p.requiresGrad())
        {
            std::cerr << "[WARN] Tensor without requiresGrad=true registered to optimizer.\n";
        }
    }
    m_parameters.insert(m_parameters.end(), params.begin(), params.end());
}

// ------------------------------------------------------------
// Base Optimizer Implementation
// ------------------------------------------------------------
void Optimizer::ZeroGrad()
{
    for (auto &param : m_parameters)
    {
        param.zeroGrad();
    }
}

// ------------------------------------------------------------
// SGD Implementation
// ------------------------------------------------------------
SGD::SGD(double learningRate) : Optimizer(learningRate) {}

void SGD::Step()
{
    for (auto &param : m_parameters)
    {
        Tensor grad = param.grad();
        if (grad.m_impl != nullptr && grad.m_impl->m_storage.GetSize() > 0)
        {
            const size_t size = param.size();
            for (size_t i = 0; i < size; ++i)
            {
                param[i] -= m_learningRate * grad[i];
            }
        }
    }
}

// ------------------------------------------------------------
// Adam Implementation
// ------------------------------------------------------------
Adam::Adam(double learningRate, double beta1, double beta2, double epsilon)
    : Optimizer(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_t(0), m_initialized(false)
{
}

void Adam::InitializeMoments()
{
    if (m_initialized)
    {
        return;
    }

    m_m.clear();
    m_v.clear();
    m_m.reserve(m_parameters.size());
    m_v.reserve(m_parameters.size());

    for (const auto &param : m_parameters)
    {
        m_m.emplace_back(Tensor(param.shape()));
        m_v.emplace_back(Tensor(param.shape()));
    }
    m_initialized = true;
}

void Adam::Step()
{
    if (!m_initialized)
    {
        InitializeMoments();
    }

    m_t++;

    double correction1 = 1.0 - std::pow(m_beta1, m_t);
    double correction2 = 1.0 - std::pow(m_beta2, m_t);

    for (size_t i = 0; i < m_parameters.size(); ++i)
    {
        Tensor param = m_parameters[i];
        Tensor grad = param.grad();

        if (grad.m_impl == nullptr || grad.m_impl->m_storage.GetSize() == 0)
        {
            continue;
        }

        Tensor &m = m_m[i];
        Tensor &v = m_v[i];

        const size_t size = param.size();
        for (size_t j = 0; j < size; ++j)
        {
            double g = grad[j];

            m[j] = m_beta1 * m[j] + (1.0 - m_beta1) * g;
            v[j] = m_beta2 * v[j] + (1.0 - m_beta2) * (g * g);

            double m_hat = m[j] / correction1;
            double v_hat = v[j] / correction2;

            param[j] -= m_learningRate * m_hat / (std::sqrt(v_hat) + m_epsilon);
        }
    }
}
