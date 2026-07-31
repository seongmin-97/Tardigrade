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

        if (m_impl->m_grad == nullptr)
        {
            if (m_impl->m_storage.GetSize() != 1)
            {
                throw std::runtime_error("Backward is only supported for scalar outputs (loss must be a scalar).");
            }

            // 1. Initialize self gradient with 1.0
            m_impl->m_grad = std::make_shared<TensorImpl>(m_impl->m_shape);
            m_impl->m_grad->m_storage[0] = 1.0;
        }

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
    // Unbroadcast Helper Utility
    // ------------------------------------------------------------

    Tensor unbroadcast(const Tensor& grad, const Shape& targetShape)
    {
        if (grad.shape() == targetShape)
        {
            return grad;
        }

        Tensor res = grad;
        int gradRank = static_cast<int>(grad.rank());
        int targetRank = static_cast<int>(targetShape.size());

        int rankDiff = gradRank - targetRank;
        for (int i = 0; i < rankDiff; ++i)
        {
            res = sum(res, 0, false);
        }

        for (int i = 0; i < targetRank; ++i)
        {
            if (targetShape[i] == 1 && res.dim(i) > 1)
            {
                res = sum(res, i, true);
            }
        }

        if (res.shape() != targetShape)
        {
            res = res.reshape(targetShape);
        }
        return res;
    }

    // ------------------------------------------------------------
    // Node Backward Implementations (PURE High-Level Tensor Ops)
    // ------------------------------------------------------------

    std::vector<Tensor> MatMulNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = matmul(dY, B.transpose());
        Tensor dB = matmul(A.transpose(), dY);

        return { dA, dB };
    }

    std::vector<Tensor> AddNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = unbroadcast(dY, A.shape());
        Tensor dB = unbroadcast(dY, B.shape());
        return { dA, dB };
    }

    std::vector<Tensor> SubNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = unbroadcast(dY, A.shape());
        Tensor dB = unbroadcast(dY * (-1.0), B.shape());
        return { dA, dB };
    }

    std::vector<Tensor> MulNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = unbroadcast(mul(dY, B), A.shape());
        Tensor dB = unbroadcast(mul(dY, A), B.shape());

        return { dA, dB };
    }

    std::vector<Tensor> DivNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor A = m_inputs[0];
        Tensor B = m_inputs[1];

        Tensor dA = unbroadcast(div(dY, B), A.shape());
        Tensor dB = unbroadcast((mul(dY, A) * -1.0) / mul(B, B), B.shape());

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

        if (m_axis == -1)
        {
            /*
             * Global sum backward: broadcast scalar gradient to all elements.
             * dX[i] = dY[0]  for all i
             */
            dX.fill(dY.data()[0]);
        }
        else
        {
            /*
             * Axis-sum backward: broadcast dY along reduced axis.
             * Detect keepDims by comparing ranks:
             *   dY.rank() == X.rank()  => keepDims=true  (size-1 dim retained)
             *   dY.rank() == X.rank()-1 => keepDims=false (dim removed)
             */
            int normAxis = Tensor::normalizeAxis(m_axis, X.rank());
            bool keepDims = (dY.rank() == X.rank());

            size_t totalElements = X.size();
            const Shape& xStrides = X.strides();
            std::vector<int> currIdx(X.rank());

            for (size_t i = 0; i < totalElements; ++i)
            {
                size_t temp = i;
                for (int d = 0; d < X.rank(); ++d)
                {
                    currIdx[d] = temp / xStrides[d];
                    temp %= xStrides[d];
                }

                std::vector<int> dyIdx;
                for (int d = 0; d < X.rank(); ++d)
                {
                    if (d == normAxis)
                    {
                        if (keepDims)
                        {
                            dyIdx.push_back(0); // keepDims: size-1, index always 0
                        }
                        // else: axis dimension was removed, skip it
                    }
                    else
                    {
                        dyIdx.push_back(currIdx[d]);
                    }
                }

                int flatDy = dyIdx.empty() ? 0 : dY.calculateIndex(dyIdx);
                dX.data()[i] = dY.data()[flatDy];
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

    std::vector<Tensor> Im2colNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        /*
         * im2col backward: undo patch extraction via col2im accumulation.
         *
         * dX = col2im(d_col, X.shape, Kh, Kw, strideH, strideW, padH, padW)
         */
        Tensor dcol = gradOutputs[0];
        Tensor dX = col2im(dcol, m_inputShape, m_kernelH, m_kernelW, m_strideH, m_strideW, m_padH, m_padW);
        return { dX };
    }

    std::vector<Tensor> ReshapeNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        /*
         * Reshape backward: restore gradient to original input shape.
         *
         * dX = reshape(dY, X.shape)
         */
        Tensor dY = gradOutputs[0];
        Tensor dX = dY.reshape(m_inputShape);
        return { dX };
    }

    std::vector<Tensor> PermuteNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        /*
         * Permute backward: apply inverse permutation.
         *
         * If forward was Y = permute(X, pi), then:
         *   dX = permute(dY, pi_inv)  where pi_inv[pi[i]] = i
         */
        Tensor dY = gradOutputs[0];
        Tensor dX = dY.permute(m_inverseAxes);
        return { dX };
    }

    std::vector<Tensor> SliceNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        Tensor dY = gradOutputs[0];
        Tensor X = m_inputs[0];
        Tensor dX = Tensor::zeros(X.shape());
        dX.setSlice(0, m_startRow, m_endRow, dY);
        return { dX };
    }

    std::vector<Tensor> ReduceMaxNode::Backward(const std::vector<Tensor>& gradOutputs)
    {
        /*
         * reduce_max backward: scatter-add gradient to argmax positions.
         *
         * dX[argmax[i]] += dY[i]  for each output element i
         * All other positions receive zero gradient (sub-differentiability of max).
         */
        Tensor dY = gradOutputs[0];
        Tensor X = m_inputs[0];

        Tensor dX(X.shape());
        dX.fill(0.0);

        const double* idxData = m_argMaxFlatIndices.data();
        const double* dyData = dY.data();
        double* dxData = dX.data();

        size_t outSize = dY.size();
        size_t inSize = X.size();

        for (size_t i = 0; i < outSize; ++i)
        {
            int maxIdx = static_cast<int>(idxData[i]);
            if (maxIdx >= 0 && maxIdx < static_cast<int>(inSize))
            {
                dxData[maxIdx] += dyData[i];
            }
        }

        return { dX };
    }
}

