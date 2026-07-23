#include "Activation.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "Autograd.hpp"

using namespace tardigrade;
using namespace tardigrade::activation;

Activation::Activation(int inputSize, int batchSize)
{
    m_size = inputSize;
    m_batchSize = batchSize;

    m_inputVector = Tensor({ m_size, m_batchSize });
    m_outputVector = Tensor({ m_size, m_batchSize });
    m_gradient = Tensor({ m_size, m_batchSize });
}

void Activation::SetBatchSize(int batchSize)
{
    if (m_batchSize == batchSize)
    {
        return;
    }

    m_batchSize = batchSize;

    m_inputVector = Tensor({ m_size, m_batchSize });
    m_outputVector = Tensor({ m_size, m_batchSize });
    m_gradient = Tensor({ m_size, m_batchSize });
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
    m_outputVector = relu(input);

    return m_outputVector;
}

Tensor ReLU::Backward(const Tensor& input)
{
    m_gradient = Tensor(input.shape());
    for (size_t i = 0; i < m_inputVector.size(); ++i)
    {
        m_gradient[i] = (m_inputVector[i] > 0.0) ? input[i] : 0.0;
    }

    return m_gradient;
}

// ------------------------------------------------------------
// Softmax Activation
// ------------------------------------------------------------

Tensor Softmax::Forward(const Tensor& input)
{
    m_inputVector = input;
    m_batchSize = (input.rank() == 1) ? 1 : input.dim(1);
    m_outputVector = softmax(input);

    return m_outputVector;
}

Tensor Softmax::Backward(const Tensor& gradOutput)
{
    m_gradient = Tensor(m_outputVector.shape());
    int rows = m_outputVector.dim(0);
    int cols = (m_outputVector.rank() == 1) ? 1 : m_outputVector.dim(1);

    for (int j = 0; j < cols; ++j)
    {
        double dot = 0.0;
        for (int i = 0; i < rows; ++i)
        {
            dot += gradOutput(i, j) * m_outputVector(i, j);
        }

        for (int i = 0; i < rows; ++i)
        {
            m_gradient(i, j) = m_outputVector(i, j) * (gradOutput(i, j) - dot);
        }
    }

    return m_gradient;
}