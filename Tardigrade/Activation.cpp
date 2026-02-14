#include "Activation.hpp"

using namespace tardigrade;
using namespace tardigrade::activation;

Activation::Activation(int inputSize)
{
	m_size = inputSize;

	m_inputVector = Vector(m_size);
	m_outputVector = Vector(m_size);
	m_gradient = Vector(m_size);
}

Vector ReLU::Forward(const Vector& input)
{
	m_inputVector = input;
	m_outputVector = input.cwiseMax(0.0);

	return m_outputVector;
}

Vector ReLU::Backward(const Vector& input)
{
	m_gradient = input.array() * (m_inputVector.array() > 0).cast<double>();

	return m_gradient;
} 