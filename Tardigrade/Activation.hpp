#pragma once

#include "Tensor.hpp"

namespace tardigrade::activation
{
	class Activation
	{
	public :
		Activation(int inputSize);

		virtual Tensor Forward(const Tensor& input) = 0;
		virtual Tensor Backward(const Tensor& input) = 0;

	protected :
		Tensor m_inputVector;
		Tensor m_outputVector;
		Tensor m_gradient;

		int m_size;
	};

	class ReLU : Activation
	{
	public :
		ReLU(int inputSize) : Activation(inputSize) {}

		Tensor Forward(const Tensor& input);
		Tensor Backward(const Tensor& input);
	};
}