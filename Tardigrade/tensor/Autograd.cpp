#include "Autograd.hpp"
#include "AutogradNodes.hpp"
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

    std::vector<Tensor> SoftmaxNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        auto outImpl = m_outputs[0].lock();
        
        if (outImpl == nullptr)
        {
            throw std::runtime_error("Softmax backward failed due to expired output reference.");
        }

        Tensor S(outImpl);
        Tensor dX(S.shape());

        int rows = S.dim(0);
        int cols = (S.rank() == 1) ? 1 : S.dim(1);

        for (int j = 0; j < cols; ++j)
        {
            double sum_dY_S = 0.0;
            for (int i = 0; i < rows; ++i)
            {
                sum_dY_S += dY(i, j) * S(i, j);
            }

            for (int i = 0; i < rows; ++i)
            {
                dX(i, j) = S(i, j) * (dY(i, j) - sum_dY_S);
            }
        }

        return { dX };
    }

    std::vector<Tensor> MseLossNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dL = gradOutputs[0];
        Tensor pred = m_inputs[0];
        Tensor target = m_inputs[1];
        double N = static_cast<double>(pred.size());

        Tensor dPred(pred.shape());
        double scale = (2.0 / N) * dL[0];

        for (size_t i = 0; i < pred.size(); ++i)
        {
            dPred[i] = scale * (pred[i] - target[i]);
        }

        Tensor dTarget(target.shape());

        return { dPred, dTarget };
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
        Tensor dX(X.shape());

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

    std::vector<Tensor> SoftmaxCrossEntropyNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dL = gradOutputs[0];
        Tensor logits = m_inputs[0];
        Tensor target = m_inputs[1];

        int rows = logits.dim(0);
        int cols = (logits.rank() == 1) ? 1 : logits.dim(1);
        double B = static_cast<double>(cols);

        Tensor dLogits(logits.shape());
        double scale = dL[0] / B;

        // Calculate Softmax S in pure Tensor operations
        Tensor S(logits.shape());
        for (int j = 0; j < cols; ++j)
        {
            double maxVal = logits(0, j);
            for (int i = 1; i < rows; ++i)
            {
                if (logits(i, j) > maxVal)
                {
                    maxVal = logits(i, j);
                }
            }

            double sum = 0.0;
            for (int i = 0; i < rows; ++i)
            {
                S(i, j) = std::exp(logits(i, j) - maxVal);
                sum += S(i, j);
            }

            for (int i = 0; i < rows; ++i)
            {
                S(i, j) /= sum;
            }
        }

        for (int j = 0; j < cols; ++j)
        {
            int targetClass = static_cast<int>(target[j]);
            for (int i = 0; i < rows; ++i)
            {
                double y = (i == targetClass) ? 1.0 : 0.0;
                dLogits(i, j) = scale * (S(i, j) - y);
            }
        }

        Tensor dTarget(target.shape());
        return { dLogits, dTarget };
    }
}
