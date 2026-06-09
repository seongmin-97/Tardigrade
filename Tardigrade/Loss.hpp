#pragma once
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "Tensor.hpp"

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

    protected:
        int m_inputSize;       ///< Input feature size (excluding batch dimension)
        int m_batchSize;       ///< Batch size

        Tensor m_prediction;   ///< Cached predictions from forward pass
        Tensor m_gradient;     ///< Cached gradients from backward pass
    };

    /**
     * @brief Softmax Activation coupled with Cross-Entropy Loss.
     * @note
     * Combining Softmax and Cross-Entropy makes the gradient calculation mathematically simple and numerically stable.
     * 
     * Mathematical formulas:
     * 
     * Softmax Forward:
     * \f[
     * \sigma(z)_{k, i} = \frac{e^{z_{k, i} - \max_j(z_{j, i})}}{\sum_j e^{z_{j, i} - \max_j(z_{j, i})}}
     * \f]
     * where $k$ is the class index and $i$ is the batch index.
     * 
     * Cross-Entropy Loss (Mean over batch):
     * \f[
     * L = -\frac{1}{N} \sum_{i=1}^N \log(\sigma(z)_{target_i, i} + \epsilon)
     * \f]
     * 
     * Combined Gradient:
     * \f[
     * \frac{\partial L}{\partial z_{k, i}} = \frac{1}{N} (\sigma(z)_{k, i} - y_{k, i})
     * \f]
     * where \f$ y_{k, i} = 1 \f$ if \f$ k == target_i \f$, else \f$ 0 \f$.
     */
    class SoftmaxCrossEntropy : public Loss
    {
    public:
        SoftmaxCrossEntropy(int inputSize, int batchSize);

        double Forward(const Tensor& logits, const Tensor& target) override;
        Tensor Backward() override;

        /**
         * @brief Get cached softmax probabilities (used for class prediction).
         * @return Reference to the probability tensor.
         */
        const Tensor& GetProbs() const;

    private:
        Tensor m_probs;   ///< Cached softmax probabilities
        Tensor m_target;  ///< Cached target labels from forward pass
    };

    /**
     * @brief Mean Squared Error (MSE) loss function for regression tasks.
     * @note
     * Mathematical formulas:
     * 
     * Forward Pass (Mean over batch and features):
     * \f[
     * L = \frac{1}{B \cdot C} \sum_{i=1}^B \sum_{j=1}^C (y\_pred_{j, i} - y\_true_{j, i})^2
     * \f]
     * where $B$ is the batch size and $C$ is the feature size (inputSize).
     * 
     * Backward Pass:
     * \f[
     * \frac{\partial L}{\partial y\_pred_{j, i}} = \frac{2}{B \cdot C} (y\_pred_{j, i} - y\_true_{j, i})
     * \f]
     */
    class MSE : public Loss
    {
    public:
        MSE(int inputSize, int batchSize);

        /**
         * @brief Forward pass for regression tasks comparing prediction and target tensors.
         * @param prediction Prediction tensor.
         * @param target Target tensor.
         * @return The scalar MSE loss value.
         */
        double Forward(const Tensor& prediction, const Tensor& target) override;

        Tensor Backward() override;

    private:
        Tensor m_target;  ///< Cached targets from forward pass
    };
}
