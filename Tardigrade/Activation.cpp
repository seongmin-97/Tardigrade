#include "Activation.hpp"

#include <cmath>
#include <algorithm>

using namespace tardigrade;
using namespace tardigrade::activation;

Activation::Activation(int inputSize, int batchSize)
{
	m_size = inputSize;
	m_batchSize = batchSize;

	m_inputVector = Tensor({ m_batchSize, m_size });
	m_outputVector = Tensor({ m_batchSize, m_size });
	m_gradient = Tensor({ m_batchSize, m_size });
}

Tensor None::Forward(const Tensor& input)
{
	return input;
}

Tensor None::Backward(const Tensor& input)
{
	return input;
}

Tensor ReLU::Forward(const Tensor& input)
{
	m_inputVector = input;
	m_outputVector = input.clampedMin(0.0);

	return m_outputVector;
}

Tensor ReLU::Backward(const Tensor& input)
{
	m_gradient = input.cwiseMul(m_inputVector.step());

	return m_gradient;
}

// ------------------------------------------------------------
// Softmax Activation
// ------------------------------------------------------------

/**
 * @brief Computes the forward pass of Softmax activation per sample.
 * @note
 * Mathematical formula for column i, class k:
 * \sigma(z)_{k, i} = e^{z_{k, i} - \max_j(z_{j, i})} / \sum_j e^{z_{j, i} - \max_j(z_{j, i})}
 */
Tensor Softmax::Forward(const Tensor& input)
{
    m_inputVector = input;

    m_batchSize = (input.rank() == 1) ? 1 : input.dim(1);

    if (m_outputVector.shape() != input.shape())
    {
        m_outputVector.reshape(input.shape());
    }

    for (int i = 0; i < m_batchSize; ++i)
    {
        double maxVal = input(0, i);
        for (int j = 1; j < m_size; ++j)
        {
            if (input(j, i) > maxVal)
            {
                maxVal = input(j, i);
            }
        }

        double sumExp = 0.0;
        for (int j = 0; j < m_size; ++j)
        {
            m_outputVector(j, i) = std::exp(input(j, i) - maxVal);
            sumExp += m_outputVector(j, i);
        }

        for (int j = 0; j < m_size; ++j)
        {
            m_outputVector(j, i) /= sumExp;
        }
    }

    return m_outputVector;
}

/**
 * @brief Computes the backward pass of Softmax activation per sample.
 * @note
 * Mathematical formula for column i, class k:
 * dL/dz_{k, i} = \sigma_{k, i} * (dL/d\sigma_{k, i} - \sum_j dL/d\sigma_{j, i} * \sigma_{j, i})
 */
Tensor Softmax::Backward(const Tensor& gradOutput)
{
    if (m_gradient.shape() != m_outputVector.shape())
    {
        m_gradient.reshape(m_outputVector.shape());
    }

    for (int i = 0; i < m_batchSize; ++i)
    {
        double dot = 0.0;
        for (int j = 0; j < m_size; ++j)
        {
            dot += gradOutput(j, i) * m_outputVector(j, i);
        }

        for (int j = 0; j < m_size; ++j)
        {
            m_gradient(j, i) = m_outputVector(j, i) * (gradOutput(j, i) - dot);
        }
    }

    return m_gradient;
}