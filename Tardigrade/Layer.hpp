#pragma once
#include <vector>
#include <memory>
#include <random>
#include <cmath>

#include "Tensor.hpp"
#include "Autograd.hpp"
#include "Activation.hpp" // For ACTIVATION enum

namespace tardigrade::layer
{
    /**
     * @brief Abstract base class representing a layer in the autograd network.
     */
    class Layer
    {
    public:
        virtual ~Layer() = default;

        /**
         * @brief Computes the forward pass of the layer.
         * @param input The input tensor to this layer.
         * @return The output tensor after transformation.
         */
        virtual Tensor Forward(const Tensor& input) = 0;

        /**
         * @brief Retrieves references to the parameter tensors of the layer.
         * @return Vector of parameter Tensors.
         */
        virtual std::vector<Tensor> GetParameters()
        {
            return {};
        }

        /**
         * @brief Initializes weights dynamically.
         */
        virtual void InitWeight() {}

        virtual void SetBatchSize(int batchSize) {}

        virtual int GetBatchSize() const
        {
            return 1;
        }
    };

    /**
     * @brief Dense (Fully Connected) layer using pure Tensor Autograd operations.
     */
    class Dense : public Layer
    {
    public:
        /**
         * @brief Construct a new Dense object.
         * @param inputSize Number of input features (excluding bias).
         * @param outputSize Number of output features.
         * @param batchSize Number of samples processed in one step.
         * @param activation Type of activation function to apply.
         */
        Dense(int inputSize, int outputSize, int batchSize = 1, activation::ACTIVATION activation = activation::ACTIVATION::NONE);

        Tensor Forward(const Tensor& input) override;

        std::vector<Tensor> GetParameters() override;

        void SetBatchSize(int batchSize) override;

        int GetBatchSize() const override;

        /**
         * @brief Initializes weight matrix using He (Kaiming) normal initialization.
         */
        void InitWeight() override;

    public:
        int m_inputSize;                  ///< Augmented input size (inputSize + 1 for bias)
        int m_outputSize;                 ///< Output size (number of neurons)
        int m_batchSize;                  ///< Batch size

        Tensor m_weight;                  ///< Weight matrix of shape (inputSize + 1, outputSize)
        activation::ACTIVATION m_enumAct; ///< Type identifier of the activation
    };
}