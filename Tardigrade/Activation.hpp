#pragma once

#include "Tensor.hpp"

namespace tardigrade::activation
{
	/**
	 * @brief Activation function types supported by the framework.
	 */
	enum class ACTIVATION
	{
		NONE,
		ReLU,
		Softmax,
	};

	/**
	 * @brief Abstract base class for activation functions.
	 */
	class Activation
	{
	public:
		/**
		 * @brief Construct a new Activation object.
		 * @param inputSize Size of the input features.
		 */
		explicit Activation(int inputSize = 0);
		virtual ~Activation() = default;

		/**
		 * @brief Computes the forward pass of the activation function.
		 * @param input The input tensor to the activation.
		 * @return The activated output tensor.
		 */
		virtual Tensor Forward(const Tensor& input) = 0;

		/**
		 * @brief Computes the backward pass (gradients) of the activation function.
		 * @param input The gradient of the loss w.r.t the activation output.
		 * @return The gradient of the loss w.r.t the activation input.
		 */
		virtual Tensor Backward(const Tensor& input) = 0;

	protected:
		Tensor m_inputVector;  ///< Cached input tensor from forward pass
		Tensor m_outputVector; ///< Cached output tensor from forward pass
		Tensor m_gradient;     ///< Cached gradient tensor from backward pass

		int m_size;            ///< Input feature size
	};

	/**
	 * @brief Identity activation function (None / No-op).
	 */
	class None : public Activation
	{
	public:
		explicit None(int inputSize = 0) : Activation(inputSize) {}

		Tensor Forward(const Tensor& input) override;
		Tensor Backward(const Tensor& input) override;
	};

	/**
	 * @brief Rectified Linear Unit (ReLU) activation function.
	 */
	class ReLU : public Activation
	{
	public:
		explicit ReLU(int inputSize = 0) : Activation(inputSize) {}

		Tensor Forward(const Tensor& input) override;
		Tensor Backward(const Tensor& input) override;
	};

	/**
	 * @brief Softmax activation function (pure activation without loss coupling).
	 * @note
	 * Mathematical formulas:
	 * 
	 * Forward Pass:
	 * \f[
	 * \sigma(z)_k = \frac{e^{z_k - \max(z)}}{\sum_j e^{z_j - \max(z)}}
	 * \f]
	 * 
	 * Backward Pass (Jacobian):
	 * \f[
	 * \frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j)
	 * \f]
	 * \f[
	 * \frac{\partial L}{\partial z_i} = \sigma_i \left( \frac{\partial L}{\partial \sigma_i} - \sum_j \frac{\partial L}{\partial \sigma_j} \sigma_j \right)
	 * \f]
	 */
	class Softmax : public Activation
	{
	public:
		explicit Softmax(int inputSize = 0) : Activation(inputSize) {}

		Tensor Forward(const Tensor& input) override;
		Tensor Backward(const Tensor& gradOutput) override;
	};
}