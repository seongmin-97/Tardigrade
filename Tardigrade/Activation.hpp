#pragma once

#include "Tensor.hpp"

namespace tardigrade::activation
{
	enum class ACTIVATION
	{
		NONE,
		ReLU,
	};

	class Activation
	{
	public :
		Activation(int inputSize, int batchSize);

		virtual Tensor Forward(const Tensor& input) = 0;
		virtual Tensor Backward(const Tensor& input) = 0;

	protected :
		Tensor m_inputVector;
		Tensor m_outputVector;
		Tensor m_gradient;

		int m_size;
		int m_batchSize;
	};

	class None : public Activation
	{
		None(int inputSize, int batchSize) : Activation(inputSize, batchSize) {}

		Tensor Forward(const Tensor& input);
		Tensor Backward(const Tensor& input);
	};

	class ReLU : public Activation
	{
	public :
		ReLU(int inputSize, int batchSize) : Activation(inputSize, batchSize) {}

		Tensor Forward(const Tensor& input);
		Tensor Backward(const Tensor& input);
	};
}