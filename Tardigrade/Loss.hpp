#pragma once
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "Tensor.hpp"
#include "Autograd.hpp"

namespace tardigrade::loss
{
    /**
     * @brief Base class for loss functions.
     * 
     * Handles forward calculation of loss values and backward calculation of gradients.
     */
    class Loss
    {
    public:
        /**
         * @brief Construct a new Loss object.
         * @param inputSize The size of the input features.
         * @param batchSize The batch size.
         */
        Loss(int inputSize, int batchSize);
        virtual ~Loss() = default;

        /**
         * @brief Computes the forward pass of the loss function.
         * @param prediction Model predictions of shape (inputSize, batchSize).
         * @param target Ground truth target tensor.
         * @return The scalar loss value.
         */
        virtual double Forward(const Tensor& prediction, const Tensor& target) = 0;

        /**
         * @brief Computes the backward pass (gradients) of the loss function.
         * @return Gradient tensor of shape (inputSize, batchSize) representing dL/d(prediction).
         */
        virtual Tensor Backward() = 0;

        /**
         * @brief Returns prediction probabilities or predictions.
         */
        virtual Tensor GetProbs() const
        {
            return m_prediction;
        }

    protected:
        int m_inputSize;       ///< Input feature size (excluding batch dimension)
        int m_batchSize;       ///< Batch size

        Tensor m_prediction;   ///< Cached predictions from forward pass
        Tensor m_target;       ///< Cached target labels from forward pass
        Tensor m_lossTensor;   ///< Cached scalar loss Tensor
    };

    /**
     * @brief Softmax Activation coupled with Cross-Entropy Loss (Autograd version).
     */
    class SoftmaxCrossEntropy : public Loss
    {
    public:
        SoftmaxCrossEntropy(int inputSize, int batchSize);

        double Forward(const Tensor& logits, const Tensor& target) override;
        Tensor Backward() override;

        /**
         * @brief Get cached softmax probabilities (used for class prediction).
         */
        Tensor GetProbs() const override;
    };

    /**
     * @brief Mean Squared Error (MSE) loss function for regression tasks (Autograd version).
     */
    class MSE : public Loss
    {
    public:
        MSE(int inputSize, int batchSize);

        double Forward(const Tensor& prediction, const Tensor& target) override;
        Tensor Backward() override;
    };
}
