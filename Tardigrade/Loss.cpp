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
      m_probs({ inputSize, batchSize }),
      m_target({ 1, batchSize })
{
}

/**
 * @brief Computes forward pass of combined Softmax and Cross-Entropy Loss for mini-batch.
 * @note
 * Mathematical formula for sample i:
 * L_i = -log(probs[target[i], i] + eps)
 * Total Loss = (1 / N) * sum(L_i)
 */
double SoftmaxCrossEntropy::Forward(const Tensor& logits, const Tensor& target)
{
    m_prediction = logits;
    m_target = target;

    m_batchSize = (logits.rank() == 1) ? 1 : logits.dim(1);

    // Reshape probs and gradient if needed to match prediction shape
    if (m_probs.shape() != logits.shape())
    {
        m_probs.reshape(logits.shape());
    }

    double totalLoss = 0.0;
    constexpr double eps = 1e-12;

    for (int i = 0; i < m_batchSize; ++i)
    {
        // 1. Softmax forward per sample (column i)
        double maxVal = logits(0, i);
        for (int j = 1; j < m_inputSize; ++j)
        {
            if (logits(j, i) > maxVal)
            {
                maxVal = logits(j, i);
            }
        }

        double sumExp = 0.0;
        for (int j = 0; j < m_inputSize; ++j)
        {
            m_probs(j, i) = std::exp(logits(j, i) - maxVal);
            sumExp += m_probs(j, i);
        }

        for (int j = 0; j < m_inputSize; ++j)
        {
            m_probs(j, i) /= sumExp;
        }

        // 2. Cross-entropy loss sum
        int labelIdx = static_cast<int>(target[i]);
        if (labelIdx < 0 || labelIdx >= m_inputSize)
        {
            throw std::runtime_error("SoftmaxCrossEntropy: target label out of range");
        }
        totalLoss -= std::log(m_probs(labelIdx, i) + eps);
    }

    return totalLoss / m_batchSize;
}

/**
 * @brief Computes backward pass of combined Softmax and Cross-Entropy Loss.
 * @note
 * Mathematical formula:
 * dL/dz_{j, i} = (1 / N) * (probs[j, i] - y_{j, i})
 */
Tensor SoftmaxCrossEntropy::Backward()
{
    if (m_gradient.shape() != m_probs.shape())
    {
        m_gradient.reshape(m_probs.shape());
    }

    for (int i = 0; i < m_batchSize; ++i)
    {
        int labelIdx = static_cast<int>(m_target[i]);
        for (int j = 0; j < m_inputSize; ++j)
        {
            double y = (j == labelIdx) ? 1.0 : 0.0;
            m_gradient(j, i) = (m_probs(j, i) - y) / m_batchSize;
        }
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
 * @note
 * Mathematical formula:
 * L = (1 / (B * C)) * sum_{i, j} (y_pred_{j, i} - y_true_{j, i})^2
 */
double MSE::Forward(const Tensor& prediction, const Tensor& target)
{
    m_prediction = prediction;
    m_target = target;

    m_batchSize = (prediction.rank() == 1) ? 1 : prediction.dim(1);

    int totalElements = m_inputSize * m_batchSize;
    double sum = 0.0;

    for (int i = 0; i < totalElements; ++i)
    {
        double diff = prediction[i] - target[i];
        sum += diff * diff;
    }

    return sum / totalElements;
}

/**
 * @brief Computes MSE backward pass.
 * @note
 * Mathematical formula:
 * dL/dy_pred_{j, i} = (2 / (B * C)) * (y_pred_{j, i} - y_true_{j, i})
 */
Tensor MSE::Backward()
{
    if (m_gradient.shape() != m_prediction.shape())
    {
        m_gradient.reshape(m_prediction.shape());
    }

    int totalElements = m_inputSize * m_batchSize;
    for (int i = 0; i < totalElements; ++i)
    {
        m_gradient[i] = 2.0 / totalElements * (m_prediction[i] - m_target[i]);
    }

    return m_gradient;
}
