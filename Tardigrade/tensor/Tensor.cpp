#include "Tensor.hpp"
#include "AutogradNodes.hpp"
#include <algorithm>
#include <iostream>

namespace tardigrade
{
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

    MatrixMap Tensor::asMatrix()
    {
        if (rank() != 2)
        {
            throw std::runtime_error("asMatrix() without arguments is only supported for 2D tensors.");
        }
        return MatrixMap(m_impl->m_storage.GetData(), dim(0), dim(1));
    }

    ConstMatrixMap Tensor::asMatrix() const
    {
        if (rank() != 2)
        {
            throw std::runtime_error("asMatrix() without arguments is only supported for 2D tensors.");
        }
        return ConstMatrixMap(m_impl->m_storage.GetData(), dim(0), dim(1));
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

    std::shared_ptr<Node> Tensor::gradNode() const
    {
        return m_impl->m_gradNode;
    }

    void Tensor::setGradNode(std::shared_ptr<Node> node)
    {
        m_impl->m_gradNode = node;
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

    Tensor Tensor::reshape(const Shape& newShape) const
    {
        size_t newTotal = 1;
        for (int d : newShape)
        {
            newTotal *= d;
        }

        if (newTotal != size())
        {
            throw std::runtime_error("Cannot reshape: total element count must match.");
        }

        auto newImpl = std::make_shared<TensorImpl>(newShape, m_impl->m_requiresGrad);
        newImpl->m_storage = m_impl->m_storage;
        newImpl->m_gradNode = m_impl->m_gradNode;
        newImpl->m_grad = m_impl->m_grad;
        return Tensor(newImpl);
    }

    Tensor Tensor::slice(int axis, int start, int end) const
    {
        if (axis < 0 || axis >= rank())
        {
            throw std::runtime_error("Axis out of bounds in slice.");
        }

        if (rank() == 2 && axis == 0)
        {
            int numRows = end - start;
            int cols = dim(1);
            Tensor res({numRows, cols});
            res.asMatrix() = asMatrix().block(start, 0, numRows, cols);
            return res;
        }
        else if (rank() == 2 && axis == 1)
        {
            int rows = dim(0);
            int numCols = end - start;
            Tensor res({rows, numCols});
            res.asMatrix() = asMatrix().block(0, start, rows, numCols);
            return res;
        }
        else if (rank() == 1)
        {
            int len = end - start;
            Tensor res({len});
            std::copy(data() + start, data() + end, res.data());
            return res;
        }
        
        throw std::runtime_error("General N-D slice currently implemented for 1D/2D.");
    }

    void Tensor::setSlice(int axis, int index, const Tensor& src)
    {
        if (axis < 0 || axis >= rank())
        {
            throw std::runtime_error("Axis out of bounds in setSlice.");
        }

        if (rank() == 2 && axis == 1)
        {
            asMatrix().col(index) = src.asVector();
        }
        else if (rank() == 2 && axis == 0)
        {
            asMatrix().row(index) = src.asVector();
        }
        else if (rank() == 1)
        {
            (*this)[index] = src[0];
        }
        else
        {
            throw std::runtime_error("General N-D setSlice currently implemented for 1D/2D.");
        }
    }

    Tensor Tensor::row(int index) const
    {
        return slice(0, index, index + 1);
    }

    void Tensor::setRow(int index, const Tensor& src)
    {
        setSlice(0, index, src);
    }

    Tensor Tensor::col(int index) const
    {
        return slice(1, index, index + 1);
    }

    void Tensor::setCol(int index, const Tensor& src)
    {
        setSlice(1, index, src);
    }

    Tensor& Tensor::operator+=(const Tensor& rhs)
    {
        if (size() != rhs.size())
        {
            throw std::runtime_error("Tensor size mismatch in operator+=.");
        }
        asVector() += rhs.asVector();
        return *this;
    }

    Tensor& Tensor::operator-=(const Tensor& rhs)
    {
        if (size() != rhs.size())
        {
            throw std::runtime_error("Tensor size mismatch in operator-=.");
        }
        asVector() -= rhs.asVector();
        return *this;
    }

    // ------------------------------------------------------------
    // Computational Forward Operations (Eigen SIMD Kernels)
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
            
            if (A.gradNode())
            {
                node->m_parents.push_back(A.gradNode());
            }
            if (B.gradNode())
            {
                node->m_parents.push_back(B.gradNode());
            }

            C.setGradNode(node);
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
            
            if (A.gradNode())
            {
                node->m_parents.push_back(A.gradNode());
            }
            if (B.gradNode())
            {
                node->m_parents.push_back(B.gradNode());
            }

            C.setGradNode(node);
            node->m_outputs.push_back(C.m_impl);
        }

        return C;
    }

    Tensor sub(const Tensor& A, const Tensor& B)
    {
        if (A.shape() != B.shape())
        {
            throw std::runtime_error("Shape mismatch for sub.");
        }

        Tensor C(A.shape(), A.requiresGrad() || B.requiresGrad());
        C.asVector() = A.asVector() - B.asVector();

        if (C.requiresGrad())
        {
            auto node = std::make_shared<SubNode>();
            node->m_inputs = { A, B };
            
            if (A.gradNode())
            {
                node->m_parents.push_back(A.gradNode());
            }
            if (B.gradNode())
            {
                node->m_parents.push_back(B.gradNode());
            }

            C.setGradNode(node);
            node->m_outputs.push_back(C.m_impl);
        }

        return C;
    }

    Tensor concat(const std::vector<Tensor>& tensors, int axis)
    {
        if (tensors.empty())
        {
            throw std::runtime_error("Cannot concat empty tensor vector.");
        }

        Shape outShape = tensors[0].shape();
        if (axis < 0 || axis >= static_cast<int>(outShape.size()))
        {
            throw std::runtime_error("Invalid concat axis.");
        }

        bool reqGrad = false;
        std::vector<int> sizes;
        sizes.reserve(tensors.size());
        sizes.push_back(tensors[0].dim(axis));

        int concatDimSum = tensors[0].dim(axis);
        for (size_t i = 1; i < tensors.size(); ++i)
        {
            if (tensors[i].rank() != tensors[0].rank())
            {
                throw std::runtime_error("Rank mismatch for concat.");
            }
            reqGrad = reqGrad || tensors[i].requiresGrad();
            sizes.push_back(tensors[i].dim(axis));
            concatDimSum += tensors[i].dim(axis);
        }
        reqGrad = reqGrad || tensors[0].requiresGrad();
        outShape[axis] = concatDimSum;

        Tensor C(outShape, reqGrad);
        int currentPos = 0;
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            int sz = tensors[i].dim(axis);
            if (C.rank() == 2 && axis == 1)
            {
                C.asMatrix().block(0, currentPos, C.dim(0), sz) = tensors[i].asMatrix();
            }
            else if (C.rank() == 2 && axis == 0)
            {
                C.asMatrix().block(currentPos, 0, sz, C.dim(1)) = tensors[i].asMatrix();
            }
            else if (C.rank() == 1)
            {
                std::copy(tensors[i].data(), tensors[i].data() + sz, C.data() + currentPos);
            }
            currentPos += sz;
        }

        if (C.requiresGrad())
        {
            auto node = std::make_shared<ConcatNode>();
            node->m_axis = axis;
            node->m_sizes = sizes;
            node->m_inputs = tensors;

            for (const auto& t : tensors)
            {
                if (t.gradNode())
                {
                    node->m_parents.push_back(t.gradNode());
                }
            }

            C.setGradNode(node);
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

            if (X.gradNode())
            {
                node->m_parents.push_back(X.gradNode());
            }

            Y.setGradNode(node);
            node->m_outputs.push_back(Y.m_impl);
        }

        return Y;
    }

    Tensor softmax(const Tensor& X)
    {
        int rows = X.dim(0);
        int cols = (X.rank() == 1) ? 1 : X.dim(1);

        Tensor Y(X.shape(), X.requiresGrad());
        ConstMatrixMap matX = X.asMatrix(rows, cols);
        MatrixMap matY = Y.asMatrix(rows, cols);

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

        if (Y.requiresGrad())
        {
            auto node = std::make_shared<SoftmaxNode>();
            node->m_inputs = { X };
            
            if (X.gradNode())
            {
                node->m_parents.push_back(X.gradNode());
            }

            Y.setGradNode(node);
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

            if (pred.gradNode())
            {
                node->m_parents.push_back(pred.gradNode());
            }
            if (target.gradNode())
            {
                node->m_parents.push_back(target.gradNode());
            }

            loss.setGradNode(node);
            node->m_outputs.push_back(loss.m_impl);
        }

        return loss;
    }

    Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& target)
    {
        if (logits.rank() != 2 || target.rank() != 2)
        {
            throw std::runtime_error("SoftmaxCrossEntropy expects 2D tensors.");
        }

        int rows = logits.dim(0);
        int cols = logits.dim(1);

        Tensor loss({1}, logits.requiresGrad() || target.requiresGrad());

        ConstMatrixMap matLogits = logits.asMatrix(rows, cols);
        Eigen::MatrixXd matS(rows, cols);

        for (int j = 0; j < cols; ++j)
        {
            double maxVal = matLogits.col(j).maxCoeff();
            double sum = 0.0;
            
            for (int i = 0; i < rows; ++i)
            {
                matS(i, j) = std::exp(matLogits(i, j) - maxVal);
                sum += matS(i, j);
            }

            for (int i = 0; i < rows; ++i)
            {
                matS(i, j) /= sum;
            }
        }

        double sumLoss = 0.0;
        constexpr double eps = 1e-15;

        for (int j = 0; j < cols; ++j)
        {
            int targetClass = static_cast<int>(target[j]);
            if (targetClass < 0 || targetClass >= rows)
            {
                throw std::runtime_error("Target index out of range in SoftmaxCrossEntropy.");
            }
            sumLoss -= std::log(matS(targetClass, j) + eps);
        }

        loss[0] = sumLoss / static_cast<double>(cols);

        if (loss.requiresGrad())
        {
            auto node = std::make_shared<SoftmaxCrossEntropyNode>();
            node->m_inputs = { logits, target };
            
            if (logits.gradNode())
            {
                node->m_parents.push_back(logits.gradNode());
            }
            if (target.gradNode())
            {
                node->m_parents.push_back(target.gradNode());
            }

            loss.setGradNode(node);
            node->m_outputs.push_back(loss.m_impl);
        }

        return loss;
    }

    Tensor Tensor::transpose() const
    {
        return tardigrade::transpose(*this);
    }

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
            
            if (X.gradNode())
            {
                node->m_parents.push_back(X.gradNode());
            }

            Y.setGradNode(node);
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

            if (X.gradNode())
            {
                node->m_parents.push_back(X.gradNode());
            }

            Y.setGradNode(node);
            node->m_outputs.push_back(Y.m_impl);
        }

        return Y;
    }

    Tensor operator+(const Tensor& lhs, const Tensor& rhs)
    {
        return add(lhs, rhs);
    }

    Tensor operator-(const Tensor& lhs, const Tensor& rhs)
    {
        return sub(lhs, rhs);
    }

    Tensor operator*(const Tensor& lhs, const Tensor& rhs)
    {
        return matmul(lhs, rhs);
    }

    Tensor operator*(const Tensor& lhs, double scalar)
    {
        Tensor res(lhs.shape());
        for (size_t i = 0; i < lhs.size(); ++i)
        {
            res[i] = lhs[i] * scalar;
        }
        return res;
    }

    Tensor operator*(double scalar, const Tensor& rhs)
    {
        return rhs * scalar;
    }
}
