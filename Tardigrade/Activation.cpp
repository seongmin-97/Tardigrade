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
 * @brief Computes the forward pass of Softmax activation (numerically stable).
 */
Tensor Softmax::Forward(const Tensor& input)
{
	m_inputVector = input;

	int n = static_cast<int>(input.size());

	double maxVal = *std::max_element(input.data(), input.data() + n);

	m_outputVector = Tensor(input.shape());
	double sumExp = 0.0;

	for (int i = 0; i < n; ++i)
	{
		m_outputVector[i] = std::exp(input[i] - maxVal);
		sumExp += m_outputVector[i];
	}

	for (int i = 0; i < n; ++i)
	{
		m_outputVector[i] /= sumExp;
	}

	return m_outputVector;
}

/**
 * @brief Computes the backward pass of Softmax activation using Jacobian matrix.
 */
Tensor Softmax::Backward(const Tensor& gradOutput)
{
	int n = static_cast<int>(m_outputVector.size());

	m_gradient = Tensor(m_outputVector.shape());

	// dot = sum_j (dL/d_sigma_j) * sigma_j
	double dot = 0.0;
	for (int i = 0; i < n; ++i)
	{
		dot += gradOutput[i] * m_outputVector[i];
	}

	// dL/dz_i = sigma_i * (dL/d_sigma_i - dot)
	for (int i = 0; i < n; ++i)
	{
		m_gradient[i] = m_outputVector[i] * (gradOutput[i] - dot);
	}

	return m_gradient;
}