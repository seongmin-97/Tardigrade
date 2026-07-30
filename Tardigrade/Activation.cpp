#include "Activation.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "Autograd.hpp"

using namespace tardigrade;
using namespace tardigrade::activation;

Activation::Activation(int inputSize)
{
    m_size = inputSize;
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
    if (m_inputVector.grad().size() > 0)
    {
        return m_inputVector.grad();
    }

    m_gradient = Tensor(input.shape());
    const double* inVal = m_inputVector.data();
    const double* gradIn = input.data();
    double* gradOut = m_gradient.data();

    for (size_t i = 0; i < m_inputVector.size(); ++i)
    {
        gradOut[i] = (inVal[i] > 0.0) ? gradIn[i] : 0.0;
    }

    return m_gradient;
}


// ------------------------------------------------------------
// Softmax Activation
// ------------------------------------------------------------

Tensor Softmax::Forward(const Tensor& input)
{
    m_inputVector = input;
    int rows = input.dim(0);
    int cols = (input.rank() == 1) ? 1 : input.dim(1);

    Tensor maxVals({1, cols});
    for (int j = 0; j < cols; ++j)
    {
        double maxCoeff = input(0, j);
        for (int i = 1; i < rows; ++i)
        {
            if (input(i, j) > maxCoeff)
            {
                maxCoeff = input(i, j);
            }
        }
        maxVals(0, j) = maxCoeff;
    }

    // Composing Softmax Forward pass using primitive tensor operations:
    // Y = exp(X - max) / sum(exp(X - max), axis=0)
    Tensor shifted = input - maxVals;
    Tensor exps = exp(shifted);
    Tensor sumExps = sum(exps, 0, true);

    m_outputVector = exps / sumExps;
    return m_outputVector;
}

Tensor Softmax::Backward(const Tensor& gradOutput)
{
    if (m_inputVector.grad().size() > 0)
    {
        return m_inputVector.grad();
    }

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