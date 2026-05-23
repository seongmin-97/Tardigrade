#include "Loss.hpp"

using namespace tardigrade;
using namespace tardigrade::loss;

// ============================================================
// Loss (Base)
// ============================================================
Loss::Loss(int inputSize, int batchSize)
    : m_inputSize(inputSize),
      m_batchSize(batchSize),
      m_prediction({ inputSize, batchSize }),
      m_gradient({ inputSize, batchSize })
{
}

// ============================================================
// SoftmaxCrossEntropy
// ============================================================
SoftmaxCrossEntropy::SoftmaxCrossEntropy(int inputSize, int batchSize)
    : Loss(inputSize, batchSize),
      m_probs({ inputSize, 1 }),
      m_label(0)
{
}

/**
 * @brief Softmax + Cross-Entropy 순전파
 *
 * Step 1 — Numerically stable Softmax:
 *   σ(z)_k = exp(z_k - max(z)) / Σ_j exp(z_j - max(z))
 *
 * Step 2 — Cross-Entropy Loss:
 *   L = -log(σ(z)_label + ε)
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

    // Step 2: Cross-Entropy Loss
    constexpr double eps = 1e-12;
    return -std::log(m_probs[label] + eps);
}

/**
 * @brief Softmax + Cross-Entropy 역전파
 *
 * Combined gradient (Softmax + CE):
 *   dL/dz_k = σ(z)_k - y_k
 *   where y_k = 1 if k == label, else 0
 *
 * 이 결합 gradient는 개별 Softmax backward + CE backward보다
 * 수치적으로 안정적이고 계산이 간단하다.
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

// ============================================================
// MSE (Mean Squared Error)
// ============================================================
MSE::MSE(int inputSize, int batchSize)
    : Loss(inputSize, batchSize),
      m_target({ inputSize, batchSize })
{
}

/**
 * @brief MSE 순전파 (Tensor 대 Tensor)
 *
 * L = (1/n) Σ_i (y_pred_i - y_true_i)²
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
 * @brief MSE 순전파 (int label → one-hot 변환)
 *
 * 정수 라벨을 one-hot 벡터로 변환한 뒤 Tensor 버전을 호출한다.
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
 * @brief MSE 역전파
 *
 * dL/dy_pred_i = (2/n)(y_pred_i - y_true_i)
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
