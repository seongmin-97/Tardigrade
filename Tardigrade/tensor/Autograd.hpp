#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <functional>
#include <stdexcept>

#include "Tensor.hpp"

namespace tardigrade
{
    class TensorImpl;
    class Tensor;

    /**
     * @brief Abstract base class representing an operation node in the computational graph.
     */
    class Node
    {
    public:
        virtual ~Node() = default;

        /**
         * @brief Computes local gradients and propagates them backward.
         * @param gradOutputs The gradients w.r.t the outputs of this node.
         * @return The gradients w.r.t the inputs of this node.
         */
        virtual std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) = 0;

        /**
         * @brief Clears inputs and parents to break reference cycles and free memory.
         */
        void ClearEdges()
        {
            m_inputs.clear();
            m_parents.clear();
            m_outputs.clear();
        }

    public:
        std::vector<Tensor> m_inputs;                     ///< Input tensors to this operation
        std::vector<std::shared_ptr<Node>> m_parents;      ///< Parent nodes that created the inputs
        std::vector<std::weak_ptr<TensorImpl>> m_outputs;  ///< Weak pointers to output implementations to prevent cycles
    };

    // ------------------------------------------------------------
    // Primitive Autograd Operation Nodes
    // ------------------------------------------------------------

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
     * @brief Node for Element-wise Subtraction.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = A - B
     * Backward:
     *   dY = gradOutputs[0]
     *   dA = dY
     *   dB = -dY
     */
    class SubNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Element-wise Multiplication (Hadamard product).
     *
     * Mathematical Formula:
     * Forward:
     *   Y = A * B
     * Backward:
     *   dY = gradOutputs[0]
     *   dA = dY * B
     *   dB = dY * A
     */
    class MulNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Element-wise Division.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = A / B
     * Backward:
     *   dY = gradOutputs[0]
     *   dA = dY / B
     *   dB = -dY * A / (B^2)
     */
    class DivNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Element-wise Natural Exponential function.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = exp(X)
     * Backward:
     *   dY = gradOutputs[0]
     *   dX = dY * Y
     */
    class ExpNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Element-wise Natural Logarithm function.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = log(X)
     * Backward:
     *   dY = gradOutputs[0]
     *   dX = dY / X
     */
    class LogNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Tensor Summation along an axis.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = sum(X, axis)
     * Backward:
     *   dY = gradOutputs[0]
     *   dX = broadcast(dY, X.shape)
     */
    class SumNode : public Node
    {
    public:
        int m_axis;

    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override;
    };

    /**
     * @brief Node for Tensor Concatenation along an axis.
     *
     * Mathematical Formula:
     * Forward:
     *   Y = concat([T_1, T_2, ..., T_k], axis)
     * Backward:
     *   dT_i = dY.slice(axis, start_i, end_i)
     */
    class ConcatNode : public Node
    {
    public:
        int m_axis;
        std::vector<int> m_sizes;

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
}
