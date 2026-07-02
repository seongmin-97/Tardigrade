#pragma once
#include "Autograd.hpp"

namespace tardigrade
{
    /**
     * @brief Node for Matrix Multiplication.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = A * B
     * Backward:
     *   dY = gradOutputs[0]
     *   dA = dY * B^T
     *   dB = A^T * dY
     */
    class MatMulNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Element-wise Addition.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = A + B
     * Backward:
     *   dY = gradOutputs[0]
     *   dA = dY
     *   dB = dY
     */
    class AddNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for ReLU Activation.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = max(0, X)
     * Backward:
     *   dX = dY * (X > 0)
     */
    class ReLUNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Softmax Activation.
     *
     * Mathematical Formula:
     * Forward:
     *   S_ij = exp(X_ij - max_k(X_kj)) / sum_k(exp(X_kj - max_m(X_mj)))
     * Backward:
     *   dX_ij = S_ij * (dY_ij - sum_k(dY_kj * S_kj))
     */
    class SoftmaxNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Mean Squared Error Loss.
     *
     * Mathematical Formula:
     * Forward:
     *   L = (1 / N) * sum((pred - target)^2)
     * Backward:
     *   dL = gradOutputs[0]
     *   dPred = dL * (2 / N) * (pred - target)
     *   dTarget = 0
     */
    class MseLossNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Tensor Transpose.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = X^T
     * Backward:
     *   dX = dY^T
     */
    class TransposeNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Tensor Row-Slicing.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = X[startRow:endRow, :]
     * Backward:
     *   dX = zeros(X.shape)
     *   dX[startRow:endRow, :] = dY
     */
    class SliceNode : public Node
    {
    public:
        int m_startRow;
        int m_endRow;

    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Softmax Activation combined with Cross Entropy Loss.
     *
     * Mathematical Formula:
     * Forward:
     *   S_ki = exp(logits_ki - max_j(logits_ji)) / sum_j(exp(logits_ji - max_j(logits_ji)))
     *   L = -(1/B) * sum_i(log(S_targeti, i + epsilon))
     * Backward:
     *   dLogits_ki = dL/B * (S_ki - y_ki)
     *   dTarget = 0
     */
    class SoftmaxCrossEntropyNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };
}
