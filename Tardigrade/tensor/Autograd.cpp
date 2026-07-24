#include "Autograd.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace tardigrade
{
    // Helper to topologically sort the computation graph
    void buildTopo(const std::shared_ptr<Node>& node, std::vector<std::shared_ptr<Node>>& sorted, std::unordered_set<std::shared_ptr<Node>>& visited)
    {
        if (node == nullptr || visited.count(node) > 0)
        {
            return;
        }

        visited.insert(node);

        for (const auto& parent : node->m_parents)
        {
            buildTopo(parent, sorted, visited);
        }

        sorted.push_back(node);
    }

    // ------------------------------------------------------------
    // Tensor Autograd Engine Operations (Backward Execution Only)
    // ------------------------------------------------------------

    void Tensor::Backward()
    {
        if (!m_impl->m_requiresGrad)
        {
            throw std::runtime_error("Backward called on a tensor that does not require gradients.");
        }

        if (m_impl->m_storage.GetSize() != 1)
        {
            throw std::runtime_error("Backward is only supported for scalar outputs (loss must be a scalar).");
        }

        // 1. Initialize self gradient with 1.0
        m_impl->m_grad = std::make_shared<TensorImpl>(m_impl->m_shape);
        m_impl->m_grad->m_storage[0] = 1.0;

        // 2. Topological sort starting from m_gradNode
        std::vector<std::shared_ptr<Node>> sortedNodes;
        std::unordered_set<std::shared_ptr<Node>> visited;
        buildTopo(m_impl->m_gradNode, sortedNodes, visited);

        // 3. Backpropagate gradients in reverse topological order
        for (auto it = sortedNodes.rbegin(); it != sortedNodes.rend(); ++it)
        {
            std::shared_ptr<Node> node = *it;

            // Collect gradients of the outputs of this node
            std::vector<Tensor> gradOutputs;
            for (const auto& weakOut : node->m_outputs)
            {
                auto outImpl = weakOut.lock();
                if (outImpl != nullptr && outImpl->m_grad != nullptr)
                {
                    gradOutputs.push_back(Tensor(outImpl->m_grad));
                }
                else
                {
                    Shape outShape = outImpl ? outImpl->m_shape : Shape{1};
                    gradOutputs.push_back(Tensor(outShape));
                }
            }

            // Execute Backward call on this node
            std::vector<Tensor> gradInputs = node->Backward(gradOutputs);

            // Accumulate gradients into inputs
            for (size_t i = 0; i < node->m_inputs.size(); ++i)
            {
                if (i >= gradInputs.size())
                {
                    break;
                }

                Tensor input = node->m_inputs[i];
                if (input.requiresGrad())
                {
                    if (input.m_impl->m_grad == nullptr)
                    {
                        input.m_impl->m_grad = std::make_shared<TensorImpl>(input.shape());
                        std::copy(gradInputs[i].data(), gradInputs[i].data() + input.size(), input.m_impl->m_grad->m_storage.GetData());
                    }
                    else
                    {
                        Tensor targetGrad(input.m_impl->m_grad);
                        targetGrad += gradInputs[i];
                    }
                }
            }
        }

        // 4. Clean reference links to break pointer cycles
        ClearGraph();
    }

    void Tensor::ClearGraph()
    {
        if (m_impl->m_gradNode != nullptr)
        {
            std::vector<std::shared_ptr<Node>> sortedNodes;
            std::unordered_set<std::shared_ptr<Node>> visited;
            buildTopo(m_impl->m_gradNode, sortedNodes, visited);

            for (auto& node : sortedNodes)
            {
                node->ClearEdges();
            }
        }
        m_impl->m_gradNode.reset();
        m_impl->m_grad.reset();
    }

    // ------------------------------------------------------------
    // Node Backward Implementations (PURE High-Level Tensor Ops)
    // ------------------------------------------------------------

    std::vector<Tensor> MatMulNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = dY * B.transpose();
        Tensor dB = A.transpose() * dY;

        return { dA, dB };
    }

    std::vector<Tensor> AddNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        return { dY.clone(), dY.clone() };
    }

    std::vector<Tensor> SubNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor dA = dY.clone();
        Tensor dB = dY * (-1.0);
        return { dA, dB };
    }

    std::vector<Tensor> MulNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = mul(dY, B);
        Tensor dB = mul(dY, A);

        return { dA, dB };
    }

    std::vector<Tensor> DivNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = div(dY, B);
        Tensor dB = (mul(dY, A) * -1.0) / mul(B, B);

        return { dA, dB };
    }

    std::vector<Tensor> ExpNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        auto outImpl = m_outputs[0].lock();
        if (outImpl == nullptr)
        {
            throw std::runtime_error("Exp backward failed due to expired output reference.");
        }
        Tensor Y(outImpl);
        Tensor dX = mul(dY, Y);
        return { dX };
    }

    std::vector<Tensor> LogNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor X = m_inputs[0];
        Tensor dX = div(dY, X);
        return { dX };
    }

    std::vector<Tensor> SumNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor X = m_inputs[0];
        Tensor dX(X.shape());

        if (m_axis == -1 || X.rank() == 1)
        {
            dX.fill(dY[0]);
        }
        else if (X.rank() == 2)
        {
            int rows = X.dim(0);
            int cols = X.dim(1);

            if (m_axis == 0)
            {
                for (int j = 0; j < cols; ++j)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        dX(i, j) = dY(0, j);
                    }
                }
            }
            else if (m_axis == 1)
            {
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < cols; ++j)
                    {
                        dX(i, j) = dY(i, 0);
                    }
                }
            }
        }

        return { dX };
    }

    std::vector<Tensor> ConcatNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        std::vector<Tensor> grads;
        grads.reserve(m_sizes.size());

        int currentPos = 0;
        for (int sz : m_sizes)
        {
            Tensor dInput = dY.slice(m_axis, currentPos, currentPos + sz);
            grads.push_back(dInput);
            currentPos += sz;
        }

        return grads;
    }

    std::vector<Tensor> ReLUNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor X = m_inputs[0];

        Tensor dX(X.shape());
        for (size_t i = 0; i < X.size(); ++i)
        {
            dX[i] = (X[i] > 0.0) ? dY[i] : 0.0;
        }

        return { dX };
    }

    std::vector<Tensor> TransposeNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        return { dY.transpose() };
    }

    std::vector<Tensor> SliceNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor X = m_inputs[0];
        Tensor dX = Tensor::zeros(X.shape());

        int rows = m_endRow - m_startRow;
        int cols = X.dim(1);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                dX(m_startRow + r, c) = dY(r, c);
            }
        }

        return { dX };
    }
}
