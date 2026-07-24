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
      m_target({ inputSize, batchSize }),
      m_lossTensor({ 1 })
{
}

// ------------------------------------------------------------
// SoftmaxCrossEntropy Implementation
// ------------------------------------------------------------
SoftmaxCrossEntropy::SoftmaxCrossEntropy(int inputSize, int batchSize)
    : Loss(inputSize, batchSize)
{
}

double SoftmaxCrossEntropy::Forward(const Tensor& logits, const Tensor& target)
{
    m_prediction = logits;
    m_target = target;

    m_batchSize = (logits.rank() == 1) ? 1 : logits.dim(1);
    int rows = logits.dim(0);
    int cols = m_batchSize;

    Tensor maxVals({1, cols});
    for (int j = 0; j < cols; ++j)
    {
        double maxCoeff = logits(0, j);
        for (int i = 1; i < rows; ++i)
        {
            if (logits(i, j) > maxCoeff)
            {
                maxCoeff = logits(i, j);
            }
        }
        maxVals(0, j) = maxCoeff;
    }

    Tensor shifted(logits.shape(), logits.requiresGrad());
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            shifted(i, j) = logits(i, j) - maxVals(0, j);
        }
    }

    Tensor exps = exp(shifted);
    Tensor sumExps = sum(exps, 0);

    Tensor S(logits.shape(), logits.requiresGrad());
    for (int j = 0; j < cols; ++j)
    {
        double colSum = sumExps(0, j);
        for (int i = 0; i < rows; ++i)
        {
            S(i, j) = exps(i, j) / colSum;
        }
    }

    double sumLoss = 0.0;
    constexpr double eps = 1e-15;

    for (int j = 0; j < cols; ++j)
    {
        int targetClass = static_cast<int>(target[j]);
        if (targetClass < 0 || targetClass >= rows)
        {
            throw std::runtime_error("Target index out of range in SoftmaxCrossEntropy.");
        }
        sumLoss -= std::log(S(targetClass, j) + eps);
    }

    m_lossTensor = Tensor({1}, logits.requiresGrad() || target.requiresGrad());
    m_lossTensor[0] = sumLoss / static_cast<double>(cols);

    if (logits.requiresGrad())
    {
        m_lossTensor.setGradNode(exps.gradNode());
    }

    return m_lossTensor[0];
}

Tensor SoftmaxCrossEntropy::Backward()
{
    if (m_prediction.requiresGrad())
    {
        int rows = m_prediction.dim(0);
        int cols = m_batchSize;

        Tensor S = GetProbs();
        Tensor dLogits(m_prediction.shape());
        double scale = 1.0 / static_cast<double>(cols);

        for (int j = 0; j < cols; ++j)
        {
            int targetClass = static_cast<int>(m_target[j]);
            for (int i = 0; i < rows; ++i)
            {
                double y = (i == targetClass) ? 1.0 : 0.0;
                dLogits(i, j) = scale * (S(i, j) - y);
            }
        }

        m_prediction.setGrad(dLogits);
    }

    return m_prediction.grad();
}

Tensor SoftmaxCrossEntropy::GetProbs() const
{
    int rows = m_prediction.dim(0);
    int cols = m_batchSize;

    Tensor maxVals({1, cols});
    for (int j = 0; j < cols; ++j)
    {
        double maxCoeff = m_prediction(0, j);
        for (int i = 1; i < rows; ++i)
        {
            if (m_prediction(i, j) > maxCoeff)
            {
                maxCoeff = m_prediction(i, j);
            }
        }
        maxVals(0, j) = maxCoeff;
    }

    Tensor shifted(m_prediction.shape());
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            shifted(i, j) = m_prediction(i, j) - maxVals(0, j);
        }
    }

    Tensor exps = exp(shifted);
    Tensor sumExps = sum(exps, 0);

    Tensor S(m_prediction.shape());
    for (int j = 0; j < cols; ++j)
    {
        double colSum = sumExps(0, j);
        for (int i = 0; i < rows; ++i)
        {
            S(i, j) = exps(i, j) / colSum;
        }
    }

    return S;
}

// ------------------------------------------------------------
// MSE (Mean Squared Error) Implementation
// ------------------------------------------------------------
MSE::MSE(int inputSize, int batchSize)
    : Loss(inputSize, batchSize)
{
}

double MSE::Forward(const Tensor& prediction, const Tensor& target)
{
    m_prediction = prediction;
    m_target = target;

    m_batchSize = (prediction.rank() == 1) ? 1 : prediction.dim(1);

    Tensor diff = sub(prediction, target);
    Tensor sq = mul(diff, diff);
    Tensor totalSum = sum(sq, -1);
    m_lossTensor = div(totalSum, static_cast<double>(prediction.size()));

    return m_lossTensor[0];
}

Tensor MSE::Backward()
{
    m_lossTensor.Backward();
    return m_prediction.grad();
}
