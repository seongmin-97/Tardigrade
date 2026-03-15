#include "Activation.hpp"

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