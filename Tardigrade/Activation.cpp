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

// ============================================================
// Softmax Activation
// ============================================================

/**
 * @brief Softmax 순전파 (numerically stable)
 *
 * σ(z)_k = exp(z_k - max(z)) / Σ_j exp(z_j - max(z))
 *
 * max(z) 를 빼서 exp의 오버플로우를 방지한다.
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
 * @brief Softmax 역전파 (Jacobian 기반)
 *
 * Softmax Jacobian:
 *   dσ_i/dz_j = σ_i(δ_ij - σ_j)
 *
 * Chain rule 적용 후 간소화:
 *   dL/dz_i = Σ_j (dL/dσ_j) · σ_j · (δ_ij - σ_i)
 *           = σ_i · (dL/dσ_i - Σ_j dL/dσ_j · σ_j)
 *           = σ_i · (dL/dσ_i - dot)
 *
 * where dot = Σ_j dL/dσ_j · σ_j
 */
Tensor Softmax::Backward(const Tensor& gradOutput)
{
	int n = static_cast<int>(m_outputVector.size());

	m_gradient = Tensor(m_outputVector.shape());

	// dot = Σ_j dL/dσ_j · σ_j
	double dot = 0.0;
	for (int i = 0; i < n; ++i)
	{
		dot += gradOutput[i] * m_outputVector[i];
	}

	// dL/dz_i = σ_i · (dL/dσ_i - dot)
	for (int i = 0; i < n; ++i)
	{
		m_gradient[i] = m_outputVector[i] * (gradOutput[i] - dot);
	}

	return m_gradient;
}