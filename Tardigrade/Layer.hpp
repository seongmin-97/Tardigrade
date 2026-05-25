#pragma once
#include <cmath>
#include <random>
#include <memory>

#include <Eigen/Dense>

#include "Tensor.hpp"
#include "Activation.hpp"

using namespace tardigrade::activation;

namespace tardigrade::layer
{
	/**
	 * @brief Abstract base class representing a layer in the neural network.
	 */
	class Layer
	{
	public :
		virtual ~Layer() = default;

		/**
		 * @brief Computes the forward pass of the layer.
		 * @param input The input tensor to this layer.
		 * @return The output tensor after transformation.
		 */
		virtual Tensor Forward(const Tensor& input) = 0;

		/**
		 * @brief Computes the backward pass (gradients) of the layer.
		 * @param input The gradient of the loss w.r.t the layer's output.
		 * @return The gradient of the loss w.r.t the layer's input.
		 */
		virtual Tensor Backward(const Tensor& input) = 0;

		/**
		 * @brief Retrieves references to the weight and gradient tensors of the layer.
		 * @return Vector of pairs containing pointers to (weight, gradient).
		 */
		virtual std::vector<std::pair<Tensor*, Tensor*>> GetParameters() { return {}; }

		/**
		 * @brief Set the input feature size of the layer.
		 * @param inputSize Size of the input features.
		 */
		virtual void SetInputSize(int inputSize) {}

		/**
		 * @brief Set the output feature size of the layer.
		 * @param outputSize Size of the output features.
		 */
		virtual void SetOutputSize(int outputSize) {}

		/**
		 * @brief Set the batch size for calculations.
		 * @param batchSize The batch size.
		 */
		virtual void SetBatchSize(int batchSize) {}
	};

	/**
	 * @brief Dense (Fully Connected) layer with support for bias and activations.
	 */
	class Dense : public Layer
	{
	public :
		/**
		 * @brief Construct a new Dense object.
		 * @param inputSize Number of input features (excluding bias).
		 * @param outputSize Number of output features.
		 * @param batchSize Number of samples processed in one step.
		 * @param activation Type of activation function to apply.
		 */
		Dense(int inputSize, int outputSize, int batchSize = 1, ACTIVATION activation = ACTIVATION::NONE);
		
		Tensor Forward(const Tensor& input) override;
		Tensor Backward(const Tensor& input) override;

		void SetInputSize(int inputSize) override;
		void SetOutputSize(int outputSize) override;
		void SetBatchSize(int batchSize) override;
		void SetActivation(ACTIVATION activation);

		/**
		 * @brief Initializes weight matrices using He (Kaiming) normal initialization.
		 * @note
		 * Bias weights (augmented row index 0) are initialized to zero.
		 */
		void InitWeight();

        std::vector<std::pair<Tensor*, Tensor*>> GetParameters() override;

	public:
		int m_inputSize;                  ///< Augmented input size (inputSize + 1 for bias)
		int m_outputSize;                 ///< Output size (number of neurons)
		int m_batchSize;                  ///< Batch size

		Tensor m_weight;                  ///< Weight matrix of shape (inputSize + 1, outputSize)
		Tensor m_gradient;                ///< Gradient matrix of shape (inputSize + 1, outputSize)

		Tensor m_inputMat;                ///< Cached augmented input from forward pass
		Tensor m_outputMat;               ///< Cached activation output from forward pass

		ACTIVATION m_enumAct;             ///< Type identifier of the activation
		std::unique_ptr<Activation> m_activation; ///< Activation function runner
	};
}