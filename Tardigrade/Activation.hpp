#pragma once

#include "Types.hpp"

namespace tardigrade::activation
{
	class Activation
	{
	public :
		Activation(int inputSize);

		virtual Vector Forward(const Vector& input) = 0;
		virtual Vector Backward(const Vector& input) = 0;

	protected :
		Vector m_inputVector;
		Vector m_outputVector;
		Vector m_gradient;

		int m_size;
	};

	class ReLU : Activation
	{
		ReLU(int inputSize) : Activation(inputSize) {}

		Vector Forward(const Vector& input);
		Vector Backward(const Vector& input);
	};
}