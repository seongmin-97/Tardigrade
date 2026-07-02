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

    // Compute combined softmax cross entropy loss using Autograd
    m_lossTensor = softmax_cross_entropy(logits, target);

    return m_lossTensor[0];
}

Tensor SoftmaxCrossEntropy::Backward()
{
    m_lossTensor.Backward();
    return m_prediction.grad();
}

Tensor SoftmaxCrossEntropy::GetProbs() const
{
    // Return the Softmax probabilities. Since we want it to be computed on the current predictions
    // without creating a new creator node if requiresGrad is not active.
    return softmax(m_prediction);
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

    // Compute MSE loss using Autograd
    m_lossTensor = mse_loss(prediction, target);

    return m_lossTensor[0];
}

Tensor MSE::Backward()
{
    m_lossTensor.Backward();
    return m_prediction.grad();
}
