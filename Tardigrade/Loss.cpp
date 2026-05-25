#include "Loss.hpp"

using namespace tardigrade;
using namespace tardigrade::loss;

// ------------------------------------------------------------
// Loss Base Class
// ------------------------------------------------------------
Loss::Loss(int inputSize, int batchSize)
    : m_inputSize(inputSize),
      m_batchSize(batchSize),
      m_prediction({ inputSize, batchSize }),
      m_gradient({ inputSize, batchSize })
{
}

// ------------------------------------------------------------
// SoftmaxCrossEntropy Implementation
// ------------------------------------------------------------
SoftmaxCrossEntropy::SoftmaxCrossEntropy(int inputSize, int batchSize)
    : Loss(inputSize, batchSize),
      m_probs({ inputSize, 1 }),
      m_label(0)
{
}

/**
 * @brief Computes forward pass of combined Softmax and Cross-Entropy Loss.
 */
double SoftmaxCrossEntropy::Forward(const Tensor& logits, int label)
{
    m_prediction = logits;
    m_label = label;

    int n = static_cast<int>(logits.size());

    // Step 1: Softmax (max-subtraction for numerical stability)
    double maxVal = *std::max_element(logits.data(), logits.data() + n);

    m_probs = Tensor(logits.shape());
    double sumExp = 0.0;

    for (int i = 0; i < n; ++i)
    {
        m_probs[i] = std::exp(logits[i] - maxVal);
        sumExp += m_probs[i];
    }

    for (int i = 0; i < n; ++i)
    {
        m_probs[i] /= sumExp;
    }

    // Step 2: Cross-Entropy Loss calculation
    constexpr double eps = 1e-12;
    return -std::log(m_probs[label] + eps);
}

/**
 * @brief Computes backward pass of combined Softmax and Cross-Entropy Loss.
 */
Tensor SoftmaxCrossEntropy::Backward()
{
    int n = static_cast<int>(m_probs.size());
    m_gradient = Tensor(m_probs.shape());

    for (int i = 0; i < n; ++i)
    {
        // dL/dz_k = p_k - y_k
        m_gradient[i] = m_probs[i] - (i == m_label ? 1.0 : 0.0);
    }

    return m_gradient;
}

const Tensor& SoftmaxCrossEntropy::GetProbs() const
{
    return m_probs;
}

// ------------------------------------------------------------
// MSE (Mean Squared Error) Implementation
// ------------------------------------------------------------
MSE::MSE(int inputSize, int batchSize)
    : Loss(inputSize, batchSize),
      m_target({ inputSize, batchSize })
{
}

/**
 * @brief Computes MSE forward pass using prediction and target tensors.
 */
double MSE::Forward(const Tensor& prediction, const Tensor& target)
{
    m_prediction = prediction;
    m_target = target;

    int n = static_cast<int>(prediction.size());
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double diff = prediction[i] - target[i];
        sum += diff * diff;
    }

    return sum / n;
}

/**
 * @brief Computes MSE forward pass by converting an integer label to a one-hot vector.
 */
double MSE::Forward(const Tensor& prediction, int label)
{
    Tensor target(prediction.shape());

    for (int i = 0; i < m_inputSize; ++i)
    {
        target[i] = (i == label ? 1.0 : 0.0);
    }

    return Forward(prediction, target);
}

/**
 * @brief Computes MSE backward pass.
 */
Tensor MSE::Backward()
{
    int n = static_cast<int>(m_prediction.size());
    m_gradient = Tensor(m_prediction.shape());

    for (int i = 0; i < n; ++i)
    {
        m_gradient[i] = 2.0 / n * (m_prediction[i] - m_target[i]);
    }

    return m_gradient;
}
