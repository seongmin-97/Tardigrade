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
}
