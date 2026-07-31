#include "Tensor.hpp"
#include "Autograd.hpp"
#include <algorithm>
#include <iostream>
#include <limits>


namespace tardigrade
{
Tensor::Tensor(const Shape &shape, bool requiresGrad) { m_impl = std::make_shared<TensorImpl>(shape, requiresGrad); }

Tensor::Tensor() { m_impl = std::make_shared<TensorImpl>(Shape{}); }

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : m_impl(impl) {}

Tensor Tensor::zeros(const Shape &shape, bool requiresGrad)
{
    Tensor t(shape, requiresGrad);
    t.fill(0.0);
    return t;
}

Tensor Tensor::ones(const Shape &shape, bool requiresGrad)
{
    Tensor t(shape, requiresGrad);
    t.fill(1.0);
    return t;
}

Tensor Tensor::fill(const Shape &shape, double value, bool requiresGrad)
{
    Tensor t(shape, requiresGrad);
    t.fill(value);
    return t;
}

void Tensor::fill(double value) { std::fill(data(), data() + size(), value); }



int Tensor::rank() const { return m_impl->m_shape.size(); }

int Tensor::dim(int index) const { return m_impl->m_shape.at(index); }

const Shape &Tensor::shape() const { return m_impl->m_shape; }

const Shape &Tensor::strides() const { return m_impl->m_strides; }

double *Tensor::data() { return m_impl->m_storage.GetData(); }

const double *Tensor::data() const { return m_impl->m_storage.GetData(); }

size_t Tensor::size() const { return m_impl->m_storage.GetSize(); }

double Tensor::item() const
{
    if (size() != 1)
    {
        throw std::runtime_error("item() is only supported for Tensors with exactly 1 element.");
    }
    return data()[0];
}



void Tensor::zeroGrad()
{
    if (m_impl->m_grad != nullptr)
    {
        m_impl->m_grad->m_storage.Resize(size());
        Tensor(m_impl->m_grad).fill(0.0);
    }
}

Tensor Tensor::clone() const
{
    Tensor result(m_impl->m_shape, m_impl->m_requiresGrad);
    std::copy(data(), data() + size(), result.data());
    return result;
}

bool Tensor::requiresGrad() const { return m_impl->m_requiresGrad; }

Tensor Tensor::grad() const
{
    if (m_impl->m_grad == nullptr)
    {
        return Tensor();
    }
    return Tensor(m_impl->m_grad);
}

void Tensor::setGrad(const Tensor &g) { m_impl->m_grad = g.m_impl; }

std::shared_ptr<Node> Tensor::gradNode() const { return m_impl->m_gradNode; }

void Tensor::setGradNode(std::shared_ptr<Node> node) { m_impl->m_gradNode = node; }

int Tensor::normalizeAxis(int axis, int rank)
{
    int norm = axis;
    if (norm < 0)
    {
        norm += rank;
    }
    if (norm < 0 || norm >= rank)
    {
        throw std::runtime_error("Axis out of bounds in Tensor operation.");
    }
    return norm;
}

int Tensor::calculateIndex(const std::vector<int> &indices) const
{
    if (indices.size() != m_impl->m_shape.size())
    {
        throw std::runtime_error("Indices dimension mismatch.");
    }

    int flatIndex = 0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        flatIndex += indices[i] * m_impl->m_strides[i];
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
    newImpl->m_storage = m_impl->m_storage; // shared storage (view semantics)

    Tensor result(newImpl);

    if (m_impl->m_requiresGrad)
    {
        /*
         * ReshapeNode tracks gradient flow through shape changes.
         * Backward: dX = reshape(dY, X.shape)
         */
        auto node = std::make_shared<ReshapeNode>();
        node->m_inputShape = m_impl->m_shape;
        node->m_inputs = {*this};

        if (m_impl->m_gradNode)
        {
            node->m_parents.push_back(m_impl->m_gradNode);
        }

        result.setGradNode(node);
        node->m_outputs.push_back(newImpl);
    }

    return result;
}

Tensor Tensor::select(int dim, int index) const
{
    int normDim = normalizeAxis(dim, rank());
    if (index < 0 || index >= m_impl->m_shape[normDim])
    {
        throw std::runtime_error("Index out of bounds in select.");
    }

    Shape subShape;
    for (int i = 0; i < rank(); ++i)
    {
        if (i != normDim)
        {
            subShape.push_back(m_impl->m_shape[i]);
        }
    }

    Tensor subTensor(subShape);
    size_t subSize = subTensor.size();

    if (subSize == 0 && subShape.empty())
    {
        subTensor = Tensor::zeros({});
        subTensor.data()[0] = (*this)(index);
        return subTensor;
    }

    std::vector<int> currIdx(rank(), 0);
    currIdx[normDim] = index;

    size_t outerLoops = 1;
    for (int i = 0; i < normDim; ++i)
    {
        outerLoops *= m_impl->m_shape[i];
    }

    size_t innerLoops = 1;
    for (int i = normDim + 1; i < rank(); ++i)
    {
        innerLoops *= m_impl->m_shape[i];
    }

    size_t dstOffset = 0;
    for (size_t o = 0; o < outerLoops; ++o)
    {
        size_t tempO = o;
        for (int i = normDim - 1; i >= 0; --i)
        {
            currIdx[i] = tempO % m_impl->m_shape[i];
            tempO /= m_impl->m_shape[i];
        }

        for (size_t in = 0; in < innerLoops; ++in)
        {
            size_t tempIn = in;
            for (int i = rank() - 1; i > normDim; --i)
            {
                currIdx[i] = tempIn % m_impl->m_shape[i];
                tempIn /= m_impl->m_shape[i];
            }

            subTensor.data()[dstOffset++] = m_impl->m_storage[calculateIndex(currIdx)];
        }
    }

    return subTensor;
}

void Tensor::setSelect(int dim, int index, const Tensor &src)
{
    int normDim = normalizeAxis(dim, rank());
    if (index < 0 || index >= m_impl->m_shape[normDim])
    {
        throw std::runtime_error("Index out of bounds in setSelect.");
    }

    std::vector<int> currIdx(rank(), 0);
    currIdx[normDim] = index;

    size_t outerLoops = 1;
    for (int i = 0; i < normDim; ++i)
    {
        outerLoops *= m_impl->m_shape[i];
    }

    size_t innerLoops = 1;
    for (int i = normDim + 1; i < rank(); ++i)
    {
        innerLoops *= m_impl->m_shape[i];
    }

    size_t srcOffset = 0;
    for (size_t o = 0; o < outerLoops; ++o)
    {
        size_t tempO = o;
        for (int i = normDim - 1; i >= 0; --i)
        {
            currIdx[i] = tempO % m_impl->m_shape[i];
            tempO /= m_impl->m_shape[i];
        }

        for (size_t in = 0; in < innerLoops; ++in)
        {
            size_t tempIn = in;
            for (int i = rank() - 1; i > normDim; --i)
            {
                currIdx[i] = tempIn % m_impl->m_shape[i];
                tempIn /= m_impl->m_shape[i];
            }

            m_impl->m_storage[calculateIndex(currIdx)] = src.data()[srcOffset++];
        }
    }
}

Tensor Tensor::slice(int dim, int start, int end) const
{
    int normDim = normalizeAxis(dim, rank());
    if (start < 0 || end > m_impl->m_shape[normDim] || start >= end)
    {
        throw std::runtime_error("Invalid slice range.");
    }

    Shape newShape = m_impl->m_shape;
    newShape[normDim] = end - start;

    Tensor result(newShape);
    std::vector<int> currIdx(rank(), 0);

    size_t totalElements = result.size();
    const auto &resStrides = result.strides();

    for (size_t i = 0; i < totalElements; ++i)
    {
        size_t temp = i;
        for (int d = 0; d < rank(); ++d)
        {
            currIdx[d] = temp / resStrides[d];
            temp %= resStrides[d];
        }

        std::vector<int> srcIdx = currIdx;
        srcIdx[normDim] += start;

        result.data()[i] = m_impl->m_storage[calculateIndex(srcIdx)];
    }

    return result;
}

void Tensor::setSlice(int dim, int start, int end, const Tensor &src)
{
    int normDim = normalizeAxis(dim, rank());
    if (start < 0 || end > m_impl->m_shape[normDim] || start >= end)
    {
        throw std::runtime_error("Invalid slice range in setSlice.");
    }

    std::vector<int> currIdx(rank(), 0);
    size_t totalElements = src.size();
    const auto &srcStrides = src.strides();

    for (size_t i = 0; i < totalElements; ++i)
    {
        size_t temp = i;
        for (int d = 0; d < rank(); ++d)
        {
            currIdx[d] = temp / srcStrides[d];
            temp %= srcStrides[d];
        }

        std::vector<int> dstIdx = currIdx;
        dstIdx[normDim] += start;

        m_impl->m_storage[calculateIndex(dstIdx)] = src.data()[i];
    }
}

Tensor Tensor::permute(const std::vector<int>& dims) const
{
    if (dims.size() != static_cast<size_t>(rank()))
    {
        throw std::runtime_error("Permute dimension count mismatch.");
    }

    Shape newShape(rank());
    for (size_t i = 0; i < static_cast<size_t>(rank()); ++i)
    {
        newShape[i] = m_impl->m_shape[dims[i]];
    }

    Tensor result(newShape, m_impl->m_requiresGrad);
    const auto& resStrides = result.strides();
    size_t totalElements = result.size();

    std::vector<int> resIdx(rank());
    std::vector<int> srcIdx(rank());

    for (size_t i = 0; i < totalElements; ++i)
    {
        size_t temp = i;
        for (int d = 0; d < rank(); ++d)
        {
            resIdx[d] = temp / resStrides[d];
            temp %= resStrides[d];
        }

        for (size_t d = 0; d < static_cast<size_t>(rank()); ++d)
        {
            srcIdx[dims[d]] = resIdx[d];
        }

        result.data()[i] = m_impl->m_storage[calculateIndex(srcIdx)];
    }

    if (m_impl->m_requiresGrad)
    {
        /*
         * PermuteNode tracks gradient flow through axis reordering.
         * Backward: dX = permute(dY, pi_inv)  where pi_inv[pi[i]] = i
         */
        std::vector<int> invDims(rank());
        for (size_t i = 0; i < static_cast<size_t>(rank()); ++i)
        {
            invDims[dims[i]] = static_cast<int>(i);
        }

        auto node = std::make_shared<PermuteNode>();
        node->m_axes = std::vector<int>(dims.begin(), dims.end());
        node->m_inverseAxes = invDims;
        node->m_inputs = {*this};

        if (gradNode())
        {
            node->m_parents.push_back(gradNode());
        }

        result.setGradNode(node);
        node->m_outputs.push_back(result.m_impl);
    }

    return result;
}

Tensor Tensor::transpose() const
{
    if (rank() < 2)
    {
        return clone();
    }
    return transpose(rank() - 2, rank() - 1);
}

Tensor Tensor::transpose(int dim0, int dim1) const
{
    int norm0 = normalizeAxis(dim0, rank());
    int norm1 = normalizeAxis(dim1, rank());

    std::vector<int> dims(rank());
    for (int i = 0; i < rank(); ++i)
    {
        dims[i] = i;
    }
    std::swap(dims[norm0], dims[norm1]);

    /*
     * Transpose is a special case of permute (swap two axes).
     * PermuteNode handles the autograd tracking.
     */
    return permute(dims);
}

Tensor Tensor::matmul(const Tensor &B) const
{
    return tardigrade::matmul(*this, B);
}

Tensor &Tensor::operator+=(const Tensor &rhs)
{
    if (size() != rhs.size())
    {
        throw std::runtime_error("Tensor size mismatch in operator+=.");
    }
    double *lhsData = data();
    const double *rhsData = rhs.data();
    size_t n = size();
    for (size_t i = 0; i < n; ++i)
    {
        lhsData[i] += rhsData[i];
    }
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &rhs)
{
    if (size() != rhs.size())
    {
        throw std::runtime_error("Tensor size mismatch in operator-=.");
    }
    double *lhsData = data();
    const double *rhsData = rhs.data();
    size_t n = size();
    for (size_t i = 0; i < n; ++i)
    {
        lhsData[i] -= rhsData[i];
    }
    return *this;
}

// ------------------------------------------------------------
// Broadcasting Utilities
// ------------------------------------------------------------

Shape broadcastShapes(const Shape &shapeA, const Shape &shapeB)
{
    int rankA = static_cast<int>(shapeA.size());
    int rankB = static_cast<int>(shapeB.size());
    int maxRank = std::max(rankA, rankB);

    Shape outShape(maxRank);
    for (int i = 0; i < maxRank; ++i)
    {
        int dimA = (i < rankA) ? shapeA[rankA - 1 - i] : 1;
        int dimB = (i < rankB) ? shapeB[rankB - 1 - i] : 1;

        if (dimA == dimB)
        {
            outShape[maxRank - 1 - i] = dimA;
        }
        else if (dimA == 1)
        {
            outShape[maxRank - 1 - i] = dimB;
        }
        else if (dimB == 1)
        {
            outShape[maxRank - 1 - i] = dimA;
        }
        else
        {
            throw std::runtime_error("Shapes are not broadcastable.");
        }
    }
    return outShape;
}

bool isBroadcastable(const Shape &shapeA, const Shape &shapeB)
{
    try
    {
        broadcastShapes(shapeA, shapeB);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

// ------------------------------------------------------------
// Computational Forward Operations (Kernels & Autograd Graph)
// ------------------------------------------------------------

Tensor matmul(const Tensor &A, const Tensor &B)
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
    C.fill(0.0);

    double *cData = C.data();
    const double *aData = A.data();
    const double *bData = B.data();

    // Cache-friendly Row-Major Matrix Multiplication (i-k-j order)
    for (int i = 0; i < a_rows; ++i)
    {
        for (int k = 0; k < a_cols; ++k)
        {
            double aVal = aData[i * a_cols + k];
            for (int j = 0; j < b_cols; ++j)
            {
                cData[i * b_cols + j] += aVal * bData[k * b_cols + j];
            }
        }
    }

    if (C.requiresGrad())
    {
        auto node = std::make_shared<MatMulNode>();
        node->m_inputs = {A, B};

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

static void applyBroadcastBinaryOp(
    const Tensor &A,
    const Tensor &B,
    Tensor &C,
    std::function<double(double, double)> op)
{
    const Shape &outShape = C.shape();
    size_t totalElements = C.size();
    const Shape &outStrides = C.strides();

    int rankA = A.rank();
    int rankB = B.rank();
    int outRank = static_cast<int>(outShape.size());

    const Shape &shapeA = A.shape();
    const Shape &shapeB = B.shape();
    const Shape &stridesA = A.strides();
    const Shape &stridesB = B.strides();

    std::vector<int> outIdx(outRank);
    std::vector<int> idxA(rankA);
    std::vector<int> idxB(rankB);

    for (size_t i = 0; i < totalElements; ++i)
    {
        size_t temp = i;
        for (int d = 0; d < outRank; ++d)
        {
            outIdx[d] = temp / outStrides[d];
            temp %= outStrides[d];
        }

        for (int d = 0; d < rankA; ++d)
        {
            int outDimIdx = outRank - rankA + d;
            idxA[d] = (shapeA[d] == 1) ? 0 : outIdx[outDimIdx];
        }

        for (int d = 0; d < rankB; ++d)
        {
            int outDimIdx = outRank - rankB + d;
            idxB[d] = (shapeB[d] == 1) ? 0 : outIdx[outDimIdx];
        }

        size_t offsetA = 0;
        for (int d = 0; d < rankA; ++d)
        {
            offsetA += idxA[d] * stridesA[d];
        }

        size_t offsetB = 0;
        for (int d = 0; d < rankB; ++d)
        {
            offsetB += idxB[d] * stridesB[d];
        }

        C.data()[i] = op(A.data()[offsetA], B.data()[offsetB]);
    }
}

Tensor add(const Tensor &A, const Tensor &B)
{
    Shape outShape = broadcastShapes(A.shape(), B.shape());
    Tensor C(outShape, A.requiresGrad() || B.requiresGrad());

    applyBroadcastBinaryOp(A, B, C, [](double a, double b) { return a + b; });

    if (C.requiresGrad())
    {
        auto node = std::make_shared<AddNode>();
        node->m_inputs = {A, B};

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

Tensor sub(const Tensor &A, const Tensor &B)
{
    Shape outShape = broadcastShapes(A.shape(), B.shape());
    Tensor C(outShape, A.requiresGrad() || B.requiresGrad());

    applyBroadcastBinaryOp(A, B, C, [](double a, double b) { return a - b; });

    if (C.requiresGrad())
    {
        auto node = std::make_shared<SubNode>();
        node->m_inputs = {A, B};

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

Tensor mul(const Tensor &A, const Tensor &B)
{
    Shape outShape = broadcastShapes(A.shape(), B.shape());
    Tensor C(outShape, A.requiresGrad() || B.requiresGrad());

    applyBroadcastBinaryOp(A, B, C, [](double a, double b) { return a * b; });

    if (C.requiresGrad())
    {
        auto node = std::make_shared<MulNode>();
        node->m_inputs = {A, B};

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

Tensor div(const Tensor &A, const Tensor &B)
{
    Shape outShape = broadcastShapes(A.shape(), B.shape());
    Tensor C(outShape, A.requiresGrad() || B.requiresGrad());

    applyBroadcastBinaryOp(A, B, C, [](double a, double b) { return a / b; });

    if (C.requiresGrad())
    {
        auto node = std::make_shared<DivNode>();
        node->m_inputs = {A, B};

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

Tensor div(const Tensor &A, double scalar) { return A * (1.0 / scalar); }

Tensor exp(const Tensor &X)
{
    Tensor Y(X.shape(), X.requiresGrad());
    const double *xData = X.data();
    double *yData = Y.data();
    size_t n = X.size();
    for (size_t i = 0; i < n; ++i)
    {
        yData[i] = std::exp(xData[i]);
    }

    if (Y.requiresGrad())
    {
        auto node = std::make_shared<ExpNode>();
        node->m_inputs = {X};

        if (X.gradNode())
        {
            node->m_parents.push_back(X.gradNode());
        }

        Y.setGradNode(node);
        node->m_outputs.push_back(Y.m_impl);
    }

    return Y;
}

Tensor log(const Tensor &X)
{
    Tensor Y(X.shape(), X.requiresGrad());
    const double *xData = X.data();
    double *yData = Y.data();
    size_t n = X.size();
    for (size_t i = 0; i < n; ++i)
    {
        yData[i] = std::log(xData[i]);
    }

    if (Y.requiresGrad())
    {
        auto node = std::make_shared<LogNode>();
        node->m_inputs = {X};

        if (X.gradNode())
        {
            node->m_parents.push_back(X.gradNode());
        }

        Y.setGradNode(node);
        node->m_outputs.push_back(Y.m_impl);
    }

    return Y;
}

Tensor sum(const Tensor &X, int axis, bool keepDims)
{
    if (axis == -1)
    {
        Tensor Y({1}, X.requiresGrad());
        const double *xData = X.data();
        size_t n = X.size();
        double sumVal = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sumVal += xData[i];
        }
        Y.data()[0] = sumVal;

        /*
         * Create SumNode BEFORE potential keepDims reshape.
         * Graph: X -> SumNode -> Y{1} -> (ReshapeNode) -> Y_keepdim
         */
        if (Y.requiresGrad())
        {
            auto node = std::make_shared<SumNode>();
            node->m_axis = axis;
            node->m_inputs = {X};

            if (X.gradNode())
            {
                node->m_parents.push_back(X.gradNode());
            }

            Y.setGradNode(node);
            node->m_outputs.push_back(Y.m_impl);
        }

        if (keepDims)
        {
            Shape kdShape(X.rank(), 1);
            Y = Y.reshape(kdShape); // ReshapeNode created on top of SumNode
        }

        return Y;
    }

    int normAxis = Tensor::normalizeAxis(axis, X.rank());
    Shape outShape;
    for (int i = 0; i < X.rank(); ++i)
    {
        if (i == normAxis)
        {
            if (keepDims)
            {
                outShape.push_back(1);
            }
        }
        else
        {
            outShape.push_back(X.dim(i));
        }
    }

    if (outShape.empty())
    {
        outShape = {1};
    }

    Tensor Y(outShape, X.requiresGrad());
    Y.fill(0.0);

    size_t totalElements = X.size();
    const Shape &xStrides = X.strides();
    std::vector<int> currIdx(X.rank());

    for (size_t i = 0; i < totalElements; ++i)
    {
        size_t temp = i;
        for (int d = 0; d < X.rank(); ++d)
        {
            currIdx[d] = temp / xStrides[d];
            temp %= xStrides[d];
        }

        std::vector<int> outIdx;
        for (int d = 0; d < X.rank(); ++d)
        {
            if (d == normAxis)
            {
                if (keepDims)
                {
                    outIdx.push_back(0);
                }
            }
            else
            {
                outIdx.push_back(currIdx[d]);
            }
        }

        int flatDst = outIdx.empty() ? 0 : Y.calculateIndex(outIdx);
        Y.data()[flatDst] += X.data()[i];
    }

    if (Y.requiresGrad())
    {
        auto node = std::make_shared<SumNode>();
        node->m_axis = normAxis;
        node->m_inputs = {X};

        if (X.gradNode())
        {
            node->m_parents.push_back(X.gradNode());
        }

        Y.setGradNode(node);
        node->m_outputs.push_back(Y.m_impl);
    }

    return Y;
}

Tensor sum(const Tensor &X, const std::vector<int> &axes, bool keepDims)
{
    Tensor current = X;
    std::vector<int> normAxes;
    for (int a : axes)
    {
        normAxes.push_back(Tensor::normalizeAxis(a, X.rank()));
    }
    std::sort(normAxes.rbegin(), normAxes.rend());

    for (int a : normAxes)
    {
        current = sum(current, a, keepDims);
    }
    return current;
}

Tensor concat(const std::vector<Tensor> &tensors, int axis)
{
    if (tensors.empty())
    {
        throw std::runtime_error("Cannot concat empty tensor vector.");
    }

    Shape outShape = tensors[0].shape();
    int normAxis = Tensor::normalizeAxis(axis, tensors[0].rank());

    bool reqGrad = false;
    int concatDimSum = 0;

    for (size_t i = 0; i < tensors.size(); ++i)
    {
        if (tensors[i].rank() != tensors[0].rank())
        {
            throw std::runtime_error("Rank mismatch for concat.");
        }
        reqGrad = reqGrad || tensors[i].requiresGrad();
        concatDimSum += tensors[i].dim(normAxis);
    }

    outShape[normAxis] = concatDimSum;
    Tensor result(outShape, reqGrad);

    int startIdx = 0;
    for (const auto &t : tensors)
    {
        int len = t.dim(normAxis);
        result.setSlice(normAxis, startIdx, startIdx + len, t);
        startIdx += len;
    }

    return result;
}

// ------------------------------------------------------------
// Convolution & Pooling Operators (im2col, col2im, conv2d, maxPool2d)
// ------------------------------------------------------------

Tensor im2col(
    const Tensor &input,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW)
{
    if (input.rank() != 4)
    {
        throw std::runtime_error("im2col requires 4D input [N, C, H, W].");
    }

    int N = input.dim(0);
    int C = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);

    int outH = (H + 2 * padH - kernelH) / strideH + 1;
    int outW = (W + 2 * padW - kernelW) / strideW + 1;

    int colRows = C * kernelH * kernelW;
    int colCols = N * outH * outW;

    Tensor col({colRows, colCols}, input.requiresGrad());
    double* colData = col.data();
    const double* inData = input.data();

    for (int c = 0; c < C; ++c)
    {
        for (int kh = 0; kh < kernelH; ++kh)
        {
            for (int kw = 0; kw < kernelW; ++kw)
            {
                int rowIdx = (c * kernelH + kh) * kernelW + kw;

                for (int n = 0; n < N; ++n)
                {
                    for (int oh = 0; oh < outH; ++oh)
                    {
                        int ih = oh * strideH - padH + kh;
                        for (int ow = 0; ow < outW; ++ow)
                        {
                            int iw = ow * strideW - padW + kw;
                            int colIdx = (n * outH + oh) * outW + ow;

                            double val = 0.0;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            {
                                size_t flatIn = ((n * C + c) * H + ih) * W + iw;
                                val = inData[flatIn];
                            }

                            colData[rowIdx * colCols + colIdx] = val;
                        }
                    }
                }
            }
        }
    }

    if (input.requiresGrad())
    {
        /*
         * Im2colNode backward:
         *   dX = col2im(d_col, X.shape, Kh, Kw, strideH, strideW, padH, padW)
         */
        auto node = std::make_shared<Im2colNode>();
        node->m_kernelH = kernelH;
        node->m_kernelW = kernelW;
        node->m_strideH = strideH;
        node->m_strideW = strideW;
        node->m_padH = padH;
        node->m_padW = padW;
        node->m_inputShape = input.shape();
        node->m_inputs = {input};

        if (input.gradNode())
        {
            node->m_parents.push_back(input.gradNode());
        }

        col.setGradNode(node);
        node->m_outputs.push_back(col.m_impl);
    }

    return col;
}

Tensor col2im(
    const Tensor &col,
    const Shape &inputShape,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW)
{
    if (inputShape.size() != 4)
    {
        throw std::runtime_error("col2im requires 4D inputShape [N, C, H, W].");
    }

    int N = inputShape[0];
    int C = inputShape[1];
    int H = inputShape[2];
    int W = inputShape[3];

    int outH = (H + 2 * padH - kernelH) / strideH + 1;
    int outW = (W + 2 * padW - kernelW) / strideW + 1;

    int colCols = N * outH * outW;

    Tensor img(inputShape);
    img.fill(0.0);

    double *imgData = img.data();
    const double *colData = col.data();

    for (int c = 0; c < C; ++c)
    {
        for (int kh = 0; kh < kernelH; ++kh)
        {
            for (int kw = 0; kw < kernelW; ++kw)
            {
                int rowIdx = (c * kernelH + kh) * kernelW + kw;

                for (int n = 0; n < N; ++n)
                {
                    for (int oh = 0; oh < outH; ++oh)
                    {
                        int ih = oh * strideH - padH + kh;
                        for (int ow = 0; ow < outW; ++ow)
                        {
                            int iw = ow * strideW - padW + kw;
                            int colIdx = (n * outH + oh) * outW + ow;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            {
                                size_t flatIn = ((n * C + c) * H + ih) * W + iw;
                                imgData[flatIn] += colData[rowIdx * colCols + colIdx];
                            }
                        }
                    }
                }
            }
        }
    }

    return img;
}

/*
 * reduce_max: Axis-wise maximum reduction with argmax tracking for backward.
 *
 * Forward:
 *   Y_i = max_{j in axis} X_{..., j, ...}
 * Backward (scatter-add via ReduceMaxNode):
 *   dX[argmax[i]] += dY[i]
 */
Tensor reduce_max(const Tensor& X, int axis, bool keepDims)
{
    int normAxis = Tensor::normalizeAxis(axis, X.rank());

    Shape outShape;
    for (int i = 0; i < X.rank(); ++i)
    {
        if (i == normAxis)
        {
            if (keepDims) outShape.push_back(1);
            // else: remove this axis
        }
        else
        {
            outShape.push_back(X.dim(i));
        }
    }
    if (outShape.empty()) outShape = {1};

    Tensor Y(outShape, X.requiresGrad());
    Tensor argMaxFlat(outShape); // stores flat linear indices into X

    const double* xData = X.data();
    double* yData = Y.data();
    double* idxData = argMaxFlat.data();

    // Compute outer x inner loop counts
    size_t outerStride = 1;
    for (int i = 0; i < normAxis; ++i) outerStride *= X.dim(i);
    int axisSize = X.dim(normAxis);
    size_t innerStride = 1;
    for (int i = normAxis + 1; i < X.rank(); ++i) innerStride *= X.dim(i);

    for (size_t outer = 0; outer < outerStride; ++outer)
    {
        for (size_t inner = 0; inner < innerStride; ++inner)
        {
            size_t outFlat = outer * innerStride + inner;
            double maxVal = -std::numeric_limits<double>::infinity();
            size_t maxIdx = 0;

            for (int k = 0; k < axisSize; ++k)
            {
                size_t xFlat = (outer * static_cast<size_t>(axisSize) + k) * innerStride + inner;
                if (xData[xFlat] > maxVal)
                {
                    maxVal = xData[xFlat];
                    maxIdx = xFlat;
                }
            }

            yData[outFlat] = maxVal;
            idxData[outFlat] = static_cast<double>(maxIdx);
        }
    }

    if (X.requiresGrad())
    {
        auto node = std::make_shared<ReduceMaxNode>();
        node->m_axis = normAxis;
        node->m_keepDims = keepDims;
        node->m_argMaxFlatIndices = argMaxFlat;
        node->m_inputs = {X};

        if (X.gradNode())
        {
            node->m_parents.push_back(X.gradNode());
        }

        Y.setGradNode(node);
        node->m_outputs.push_back(Y.m_impl);
    }

    return Y;
}

/*
 * convolve: Named composition of primitive ops for 2D cross-correlation.
 * This is NOT a new opaque op — it builds a transparent autograd graph.
 *
 * Mathematical Formula:
 *   Y_{[C_out, N*H_out*W_out]} = W_{flat} \cdot \text{im2col}(X)
 *   Y_{[N, C_out, H_out, W_out]} = \text{permute}(\text{reshape}(Y_{flat}), \{1,0,2,3\})
 *
 * Nodes created: Im2colNode → ReshapeNode(kernel) → MatMulNode → ReshapeNode → PermuteNode
 */
Tensor convolve(const Tensor& input, const Tensor& kernel, int stride, int padding)
{
    if (input.rank() != 4 || kernel.rank() != 4)
    {
        throw std::runtime_error("convolve requires 4D input [N,C_in,H,W] and 4D kernel [C_out,C_in,Kh,Kw].");
    }

    int N = input.dim(0);
    int C_in = input.dim(1);
    int C_out = kernel.dim(0);
    int Kh = kernel.dim(2);
    int Kw = kernel.dim(3);

    if (input.dim(1) != kernel.dim(1))
    {
        throw std::runtime_error("convolve: input C_in does not match kernel C_in.");
    }

    int outH = (input.dim(2) + 2 * padding - Kh) / stride + 1;
    int outW = (input.dim(3) + 2 * padding - Kw) / stride + 1;

    // Step 1: im2col — [C_in*Kh*Kw, N*outH*outW]  (Im2colNode)
    Tensor col = im2col(input, Kh, Kw, stride, stride, padding, padding);

    // Step 2: flatten kernel — [C_out, C_in*Kh*Kw]  (ReshapeNode)
    Tensor W_flat = kernel.reshape({C_out, C_in * Kh * Kw});

    // Step 3: matmul — [C_out, N*outH*outW]  (MatMulNode)
    Tensor Y_flat = matmul(W_flat, col);

    // Step 4: reshape + permute — [N, C_out, outH, outW]  (ReshapeNode + PermuteNode)
    return Y_flat.reshape({C_out, N, outH, outW}).permute({1, 0, 2, 3});
}

/*
 * Free function transpose: delegates to Tensor::transpose().
 * PermuteNode is created inside the member function.
 */
Tensor transpose(const Tensor& X)
{
    return X.transpose();
}

/*
 * Free function permute: delegates to Tensor::permute(dims).
 * PermuteNode is created inside the member function.
 */
Tensor permute(const Tensor& X, const std::vector<int>& dims)
{
    return X.permute(dims);
}

Tensor slice(const Tensor &X, int startRow, int endRow)
{
    Tensor Y = X.slice(0, startRow, endRow);
    if (X.requiresGrad())
    {
        auto node = std::make_shared<SliceNode>();
        node->m_startRow = startRow;
        node->m_endRow = endRow;
        node->m_inputs = {X};

        if (X.gradNode())
        {
            node->m_parents.push_back(X.gradNode());
        }

        Y.setGradNode(node);
        node->m_outputs.push_back(Y.m_impl);
    }

    return Y;
}

// ------------------------------------------------------------
// Global Tensor Arithmetic & Comparison Operators
// ------------------------------------------------------------

Tensor operator+(const Tensor &lhs, const Tensor &rhs) { return add(lhs, rhs); }

Tensor operator+(const Tensor &lhs, double scalar) { return add(lhs, Tensor::fill(lhs.shape(), scalar)); }

Tensor operator+(double scalar, const Tensor &rhs) { return add(Tensor::fill(rhs.shape(), scalar), rhs); }

Tensor operator-(const Tensor &lhs, const Tensor &rhs) { return sub(lhs, rhs); }

Tensor operator-(const Tensor &lhs, double scalar) { return sub(lhs, Tensor::fill(lhs.shape(), scalar)); }

Tensor operator-(double scalar, const Tensor &rhs) { return sub(Tensor::fill(rhs.shape(), scalar), rhs); }

Tensor operator*(const Tensor &lhs, const Tensor &rhs) { return mul(lhs, rhs); }

Tensor operator*(const Tensor &lhs, double scalar) { return mul(lhs, Tensor::fill(lhs.shape(), scalar)); }

Tensor operator*(double scalar, const Tensor &rhs) { return mul(Tensor::fill(rhs.shape(), scalar), rhs); }

Tensor operator%(const Tensor &lhs, const Tensor &rhs) { return matmul(lhs, rhs); }

Tensor operator/(const Tensor &lhs, const Tensor &rhs) { return div(lhs, rhs); }

Tensor operator/(const Tensor &lhs, double scalar) { return div(lhs, Tensor::fill(lhs.shape(), scalar)); }

Tensor operator/(double scalar, const Tensor &rhs) { return div(Tensor::fill(rhs.shape(), scalar), rhs); }

static Tensor applyCompOp(
    const Tensor &lhs,
    const Tensor &rhs,
    std::function<bool(double, double)> comp)
{
    Shape outShape = broadcastShapes(lhs.shape(), rhs.shape());
    Tensor res(outShape);

    applyBroadcastBinaryOp(lhs, rhs, res, [comp](double a, double b) {
        return comp(a, b) ? 1.0 : 0.0;
    });

    return res;
}

Tensor operator==(const Tensor &lhs, const Tensor &rhs) { return applyCompOp(lhs, rhs, std::equal_to<double>()); }

Tensor operator==(const Tensor &lhs, double scalar) { return applyCompOp(lhs, Tensor::fill(lhs.shape(), scalar), std::equal_to<double>()); }

Tensor operator==(double scalar, const Tensor &rhs) { return applyCompOp(Tensor::fill(rhs.shape(), scalar), rhs, std::equal_to<double>()); }

Tensor operator!=(const Tensor &lhs, const Tensor &rhs) { return applyCompOp(lhs, rhs, std::not_equal_to<double>()); }

Tensor operator!=(const Tensor &lhs, double scalar) { return applyCompOp(lhs, Tensor::fill(lhs.shape(), scalar), std::not_equal_to<double>()); }

Tensor operator!=(double scalar, const Tensor &rhs) { return applyCompOp(Tensor::fill(rhs.shape(), scalar), rhs, std::not_equal_to<double>()); }

Tensor operator<(const Tensor &lhs, const Tensor &rhs) { return applyCompOp(lhs, rhs, std::less<double>()); }

Tensor operator<(const Tensor &lhs, double scalar) { return applyCompOp(lhs, Tensor::fill(lhs.shape(), scalar), std::less<double>()); }

Tensor operator<(double scalar, const Tensor &rhs) { return applyCompOp(Tensor::fill(rhs.shape(), scalar), rhs, std::less<double>()); }

Tensor operator>(const Tensor &lhs, const Tensor &rhs) { return applyCompOp(lhs, rhs, std::greater<double>()); }

Tensor operator>(const Tensor &lhs, double scalar) { return applyCompOp(lhs, Tensor::fill(lhs.shape(), scalar), std::greater<double>()); }

Tensor operator>(double scalar, const Tensor &rhs) { return applyCompOp(Tensor::fill(rhs.shape(), scalar), rhs, std::greater<double>()); }

Tensor operator<=(const Tensor &lhs, const Tensor &rhs) { return applyCompOp(lhs, rhs, std::less_equal<double>()); }

Tensor operator<=(const Tensor &lhs, double scalar) { return applyCompOp(lhs, Tensor::fill(lhs.shape(), scalar), std::less_equal<double>()); }

Tensor operator<=(double scalar, const Tensor &rhs) { return applyCompOp(Tensor::fill(rhs.shape(), scalar), rhs, std::less_equal<double>()); }

Tensor operator>=(const Tensor &lhs, const Tensor &rhs) { return applyCompOp(lhs, rhs, std::greater_equal<double>()); }

Tensor operator>=(const Tensor &lhs, double scalar) { return applyCompOp(lhs, Tensor::fill(lhs.shape(), scalar), std::greater_equal<double>()); }

Tensor operator>=(double scalar, const Tensor &rhs) { return applyCompOp(Tensor::fill(rhs.shape(), scalar), rhs, std::greater_equal<double>()); }

} // namespace tardigrade


