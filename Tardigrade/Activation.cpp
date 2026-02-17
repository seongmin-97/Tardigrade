#include "Activation.hpp"

using namespace tardigrade;
using namespace tardigrade::activation;

Activation::Activation(int inputSize)
{
	m_size = inputSize;

	m_inputVector = Tensor({ m_size });
	m_outputVector = Tensor({ m_size });
	m_gradient = Tensor({ m_size });
}

Tensor ReLU::Forward(const Tensor& input)
{
	m_inputVector = input;
	m_outputVector = input.clampedMin(0.0);

	return m_outputVector;
}

Tensor ReLU::Backward(const Tensor& input)
{
	m_gradient = input * m_inputVector.step();

	return m_gradient;
} 