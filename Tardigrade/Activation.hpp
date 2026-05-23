#pragma once

#include "Tensor.hpp"

namespace tardigrade::activation
{
	enum class ACTIVATION
	{
		NONE,
		ReLU,
		Softmax,
	};

	class Activation
	{
	public :
		Activation(int inputSize, int batchSize);
		virtual ~Activation() = default;

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
	public :
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

	/**
	 * @brief Softmax Activation (순수 activation, Loss 결합 없음)
	 *
	 * 추론 시 또는 다른 Loss 함수와 조합할 때 사용.
	 * SoftmaxCrossEntropy(Loss)와 Forward 로직은 동일하지만,
	 * Backward가 Jacobian 행렬 기반으로 다르다.
	 *
	 * Forward:
	 *   σ(z)_k = exp(z_k - max(z)) / Σ_j exp(z_j - max(z))
	 *
	 * Backward (Jacobian):
	 *   dσ_i/dz_j = σ_i(δ_ij - σ_j)
	 *   dL/dz_i = σ_i · (dL/dσ_i - Σ_j dL/dσ_j · σ_j)
	 */
	class Softmax : public Activation
	{
	public :
		Softmax(int inputSize, int batchSize) : Activation(inputSize, batchSize) {}

		Tensor Forward(const Tensor& input);
		Tensor Backward(const Tensor& gradOutput);
	};
}