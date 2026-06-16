#include "Autograd.hpp"
#include <numeric>
#include <algorithm>
#include <iostream>

namespace tardigrade::autograd
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
    // Tensor Implementation
    // ------------------------------------------------------------

    Tensor::Tensor(const Shape& shape, bool requiresGrad)
    {
        m_impl = std::make_shared<TensorImpl>(shape, requiresGrad);
    }

    Tensor::Tensor()
    {
        m_impl = std::make_shared<TensorImpl>(Shape{});
    }

    Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
        : m_impl(impl)
    {
    }

    MatrixMap Tensor::asMatrix(int rows, int cols)
    {
        return MatrixMap(m_impl->m_storage.GetData(), rows, cols);
    }

    ConstMatrixMap Tensor::asMatrix(int rows, int cols) const
    {
        return ConstMatrixMap(m_impl->m_storage.GetData(), rows, cols);
    }

    VectorMap Tensor::asVector()
    {
        return VectorMap(m_impl->m_storage.GetData(), m_impl->m_storage.GetSize());
    }

    ConstVectorMap Tensor::asVector() const
    {
        return ConstVectorMap(m_impl->m_storage.GetData(), m_impl->m_storage.GetSize());
    }

    int Tensor::rank() const
    {
        return m_impl->m_shape.size();
    }

    int Tensor::dim(int index) const
    {
        return m_impl->m_shape.at(index);
    }

    const Shape& Tensor::shape() const
    {
        return m_impl->m_shape;
    }

    double* Tensor::data()
    {
        return m_impl->m_storage.GetData();
    }

    const double* Tensor::data() const
    {
        return m_impl->m_storage.GetData();
    }

    size_t Tensor::size() const
    {
        return m_impl->m_storage.GetSize();
    }

    double& Tensor::operator[](size_t index)
    {
        return m_impl->m_storage[index];
    }

    const double& Tensor::operator[](size_t index) const
    {
        return m_impl->m_storage[index];
    }

    void Tensor::zeroGrad()
    {
        if (m_impl->m_grad != nullptr)
        {
            m_impl->m_grad->m_storage.Resize(size());
            std::fill(m_impl->m_grad->m_storage.GetData(), m_impl->m_grad->m_storage.GetData() + size(), 0.0);
        }
    }

    Tensor Tensor::clone() const
    {
        Tensor result(m_impl->m_shape, m_impl->m_requiresGrad);
        std::copy(data(), data() + size(), result.data());
        return result;
    }

    bool Tensor::requiresGrad() const
    {
        return m_impl->m_requiresGrad;
    }

    Tensor Tensor::grad() const
    {
        if (m_impl->m_grad == nullptr)
        {
            return Tensor();
        }
        return Tensor(m_impl->m_grad);
    }

    void Tensor::setGrad(const Tensor& g)
    {
        m_impl->m_grad = g.m_impl;
    }

    std::shared_ptr<Node> Tensor::creator() const
    {
        return m_impl->m_creator;
    }

    void Tensor::setCreator(std::shared_ptr<Node> node)
    {
        m_impl->m_creator = node;
    }

    int Tensor::calculateIndex(const std::vector<int>& indices) const
    {
        int flatIndex = 0;
        int stride = 1;

        for (int i = m_impl->m_shape.size() - 1; i >= 0; --i)
        {
            flatIndex += indices[i] * stride;
            stride *= m_impl->m_shape[i];
        }

        return flatIndex;
    }

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

        // 2. Topological sort starting from m_creator
        std::vector<std::shared_ptr<Node>> sortedNodes;
        std::unordered_set<std::shared_ptr<Node>> visited;
        buildTopo(m_impl->m_creator, sortedNodes, visited);

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
                        input.asVector() += gradInputs[i].asVector();
                    }
                }
            }
        }

        // 4. Clean reference links to break pointer cycles
        ClearGraph();
    }

    void Tensor::ClearGraph()
    {
        if (m_impl->m_creator != nullptr)
        {
            std::vector<std::shared_ptr<Node>> sortedNodes;
            std::unordered_set<std::shared_ptr<Node>> visited;
            buildTopo(m_impl->m_creator, sortedNodes, visited);

            for (auto& node : sortedNodes)
            {
                node->ClearEdges();
            }
        }
        m_impl->m_creator.reset();
        m_impl->m_grad.reset();
    }

    // ------------------------------------------------------------
    // Computational Ops Node Implementations
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
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override
        {
            Tensor dY = gradOutputs[0];
            Tensor A = m_inputs[0];
            Tensor B = m_inputs[1];

            int a_rows = A.dim(0);
            int a_cols = (A.rank() == 1) ? 1 : A.dim(1);
            int b_rows = B.dim(0);
            int b_cols = (B.rank() == 1) ? 1 : B.dim(1);

            // Compute dA = dY * B^T
            Tensor dA({a_rows, a_cols});
            dA.asMatrix(a_rows, a_cols) = dY.asMatrix(a_rows, b_cols) * B.asMatrix(b_rows, b_cols).transpose();

            // Compute dB = A^T * dY
            Tensor dB({b_rows, b_cols});
            dB.asMatrix(b_rows, b_cols) = A.asMatrix(a_rows, a_cols).transpose() * dY.asMatrix(a_rows, b_cols);

            return { dA, dB };
        }
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
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override
        {
            Tensor dY = gradOutputs[0];
            // Since shape dimensions are matched, gradients are identical.
            return { dY.clone(), dY.clone() };
        }
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
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override
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
    };

    /**
     * @brief Node for Mean Squared Error Loss.
     *
     * Mathematical Formula:
     * Forward:
     *   L = (1 / N) * sum((pred - target)^2)
     * Backward:
     *   dL = gradOutputs[0] (which is typically 1.0)
     *   dPred = dL * (2 / N) * (pred - target)
     *   dTarget = 0 (since target does not require gradient)
     */
    class MseLossNode : public Node
    {
    public:
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override
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

            // target doesn't require gradients, but we return a zero gradient to align index
            Tensor dTarget(target.shape());

            return { dPred, dTarget };
        }
    };

    // ------------------------------------------------------------
    // Operations Execution
    // ------------------------------------------------------------

    Tensor matmul(const Tensor& A, const Tensor& B)
    {
        int a_rows = A.dim(0);
        int a_cols = (A.rank() == 1) ? 1 : A.dim(1);
        int b_rows = B.dim(0);
        int b_cols = (B.rank() == 1) ? 1 : B.dim(1);

        if (a_cols != b_rows)
        {
            throw std::runtime_error("Dimension mismatch for matmul.");
        }

        Tensor C({a_rows, b_cols}, A.requiresGrad() || B.requiresGrad());
        C.asMatrix(a_rows, b_cols) = A.asMatrix(a_rows, a_cols) * B.asMatrix(b_rows, b_cols);

        if (C.requiresGrad())
        {
            auto node = std::make_shared<MatMulNode>();
            node->m_inputs = { A, B };
            
            if (A.creator())
            {
                node->m_parents.push_back(A.creator());
            }
            if (B.creator())
            {
                node->m_parents.push_back(B.creator());
            }

            C.setCreator(node);
            node->m_outputs.push_back(C.m_impl);
        }

        return C;
    }

    Tensor add(const Tensor& A, const Tensor& B)
    {
        if (A.shape() != B.shape())
        {
            throw std::runtime_error("Shape mismatch for add.");
        }

        Tensor C(A.shape(), A.requiresGrad() || B.requiresGrad());
        C.asVector() = A.asVector() + B.asVector();

        if (C.requiresGrad())
        {
            auto node = std::make_shared<AddNode>();
            node->m_inputs = { A, B };
            
            if (A.creator())
            {
                node->m_parents.push_back(A.creator());
            }
            if (B.creator())
            {
                node->m_parents.push_back(B.creator());
            }

            C.setCreator(node);
            node->m_outputs.push_back(C.m_impl);
        }

        return C;
    }

    Tensor relu(const Tensor& X)
    {
        Tensor Y(X.shape(), X.requiresGrad());
        for (size_t i = 0; i < X.size(); ++i)
        {
            Y[i] = (X[i] > 0.0) ? X[i] : 0.0;
        }

        if (Y.requiresGrad())
        {
            auto node = std::make_shared<ReLUNode>();
            node->m_inputs = { X };

            if (X.creator())
            {
                node->m_parents.push_back(X.creator());
            }

            Y.setCreator(node);
            node->m_outputs.push_back(Y.m_impl);
        }

        return Y;
    }

    Tensor mse_loss(const Tensor& pred, const Tensor& target)
    {
        if (pred.shape() != target.shape())
        {
            throw std::runtime_error("Shape mismatch for mse_loss.");
        }

        Tensor loss({1}, pred.requiresGrad() || target.requiresGrad());
        double sum = 0.0;
        for (size_t i = 0; i < pred.size(); ++i)
        {
            double diff = pred[i] - target[i];
            sum += diff * diff;
        }
        loss[0] = sum / static_cast<double>(pred.size());

        if (loss.requiresGrad())
        {
            auto node = std::make_shared<MseLossNode>();
            node->m_inputs = { pred, target };

            if (pred.creator())
            {
                node->m_parents.push_back(pred.creator());
            }
            if (target.creator())
            {
                node->m_parents.push_back(target.creator());
            }

            loss.setCreator(node);
            node->m_outputs.push_back(loss.m_impl);
        }

        return loss;
    }

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
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override
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

            Eigen::MatrixXd matS = S.asMatrix(rows, cols);
            Eigen::MatrixXd matdY = dY.asMatrix(rows, cols);
            Eigen::MatrixXd matdX(rows, cols);

            for (int j = 0; j < cols; ++j)
            {
                double sum_dY_S = 0.0;
                for (int i = 0; i < rows; ++i)
                {
                    sum_dY_S += matdY(i, j) * matS(i, j);
                }

                for (int i = 0; i < rows; ++i)
                {
                    matdX(i, j) = matS(i, j) * (matdY(i, j) - sum_dY_S);
                }
            }

            dX.asMatrix(rows, cols) = matdX;
            return { dX };
        }
    };

    Tensor softmax(const Tensor& X)
    {
        int rows = X.dim(0);
        int cols = (X.rank() == 1) ? 1 : X.dim(1);

        Tensor Y(X.shape(), X.requiresGrad());
        Eigen::MatrixXd matX = X.asMatrix(rows, cols);
        Eigen::MatrixXd matY(rows, cols);

        for (int j = 0; j < cols; ++j)
        {
            double maxVal = matX.col(j).maxCoeff();
            double sum = 0.0;
            
            for (int i = 0; i < rows; ++i)
            {
                matY(i, j) = std::exp(matX(i, j) - maxVal);
                sum += matY(i, j);
            }

            for (int i = 0; i < rows; ++i)
            {
                matY(i, j) /= sum;
            }
        }

        Y.asMatrix(rows, cols) = matY;

        if (Y.requiresGrad())
        {
            auto node = std::make_shared<SoftmaxNode>();
            node->m_inputs = { X };
            
            if (X.creator())
            {
                node->m_parents.push_back(X.creator());
            }

            Y.setCreator(node);
            node->m_outputs.push_back(Y.m_impl);
        }

        return Y;
    }

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
        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override
        {
            Tensor dY = gradOutputs[0];
            Tensor dX({dY.dim(1), dY.dim(0)});
            dX.asMatrix(dY.dim(1), dY.dim(0)) = dY.asMatrix(dY.dim(0), dY.dim(1)).transpose();
            return { dX };
        }
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

        std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) override
        {
            Tensor dY = gradOutputs[0];
            Tensor X = m_inputs[0];
            Tensor dX(X.shape()); // Initialized to 0.0

            int rows = m_endRow - m_startRow;
            int cols = X.dim(1);
            dX.asMatrix(X.dim(0), cols).block(m_startRow, 0, rows, cols) = dY.asMatrix(rows, cols);

            return { dX };
        }
    };

    Tensor transpose(const Tensor& X)
    {
        if (X.rank() != 2)
        {
            throw std::runtime_error("Transpose is only supported for 2D tensors.");
        }

        int rows = X.dim(0);
        int cols = X.dim(1);

        Tensor Y({cols, rows}, X.requiresGrad());
        Y.asMatrix(cols, rows) = X.asMatrix(rows, cols).transpose();

        if (Y.requiresGrad())
        {
            auto node = std::make_shared<TransposeNode>();
            node->m_inputs = { X };
            
            if (X.creator())
            {
                node->m_parents.push_back(X.creator());
            }

            Y.setCreator(node);
            node->m_outputs.push_back(Y.m_impl);
        }

        return Y;
    }

    Tensor slice(const Tensor& X, int startRow, int endRow)
    {
        if (X.rank() != 2)
        {
            throw std::runtime_error("Slice is only supported for 2D tensors.");
        }

        int rows = X.dim(0);
        int cols = X.dim(1);

        if (startRow < 0 || endRow > rows || startRow >= endRow)
        {
            throw std::runtime_error("Slice boundaries out of bounds.");
        }

        int sliceRows = endRow - startRow;
        Tensor Y({sliceRows, cols}, X.requiresGrad());
        Y.asMatrix(sliceRows, cols) = X.asMatrix(rows, cols).block(startRow, 0, sliceRows, cols);

        if (Y.requiresGrad())
        {
            auto node = std::make_shared<SliceNode>();
            node->m_startRow = startRow;
            node->m_endRow = endRow;
            node->m_inputs = { X };

            if (X.creator())
            {
                node->m_parents.push_back(X.creator());
            }

            Y.setCreator(node);
            node->m_outputs.push_back(Y.m_impl);
        }

        return Y;
    }

    Tensor operator+(const Tensor& A, const Tensor& B)
    {
        return add(A, B);
    }
}
