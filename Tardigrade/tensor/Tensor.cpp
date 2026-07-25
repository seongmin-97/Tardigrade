#include "Tensor.hpp"
#include "Autograd.hpp"
#include <algorithm>
#include <iostream>

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

MatrixMap Tensor::asMatrix(int rows, int cols) { return MatrixMap(m_impl->m_storage.GetData(), rows, cols); }

ConstMatrixMap Tensor::asMatrix(int rows, int cols) const
{
    return ConstMatrixMap(m_impl->m_storage.GetData(), rows, cols);
}

VectorMap Tensor::asVector() { return VectorMap(m_impl->m_storage.GetData(), m_impl->m_storage.GetSize()); }

ConstVectorMap Tensor::asVector() const
{
    return ConstVectorMap(m_impl->m_storage.GetData(), m_impl->m_storage.GetSize());
}

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

Tensor Tensor::reshape(const Shape &newShape) const
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

Tensor Tensor::permute(const std::vector<int> &dims) const
{
    if (dims.size() != rank())
    {
        throw std::runtime_error("Permute dimension count mismatch.");
    }

    Shape newShape(rank());
    for (size_t i = 0; i < rank(); ++i)
    {
        newShape[i] = m_impl->m_shape[dims[i]];
    }

    Tensor result(newShape);
    const auto &resStrides = result.strides();
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

        for (size_t d = 0; d < rank(); ++d)
        {
            srcIdx[dims[d]] = resIdx[d];
        }

        result.data()[i] = m_impl->m_storage[calculateIndex(srcIdx)];
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

    return permute(dims);
}

Tensor &Tensor::operator+=(const Tensor &rhs)
{
    if (size() != rhs.size())
    {
        throw std::runtime_error("Tensor size mismatch in operator+=.");
    }
    asVector() += rhs.asVector();
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &rhs)
{
    if (size() != rhs.size())
    {
        throw std::runtime_error("Tensor size mismatch in operator-=.");
    }
    asVector() -= rhs.asVector();
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
    C.asMatrix(a_rows, b_cols) = A.asMatrix(a_rows, a_cols) * B.asMatrix(b_rows, b_cols);

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
    Y.asVector() = X.asVector().array().exp();

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
    Y.asVector() = X.asVector().array().log();

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
        Y.data()[0] = X.asVector().sum();

        if (keepDims)
        {
            Shape kdShape(X.rank(), 1);
            Y = Y.reshape(kdShape);
        }

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

    Tensor col({colRows, colCols});
    double *colData = col.data();
    const double *inData = input.data();

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

Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding)
{
    if (input.rank() != 4 || weight.rank() != 4)
    {
        throw std::runtime_error("conv2d requires 4D input [N, C, H, W] and 4D weight [C_out, C_in, Kh, Kw].");
    }

    int N = input.dim(0);
    int C_in = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);

    int C_out = weight.dim(0);
    int weightC_in = weight.dim(1);
    int Kh = weight.dim(2);
    int Kw = weight.dim(3);

    if (C_in != weightC_in)
    {
        throw std::runtime_error("Input channel count must match weight C_in.");
    }

    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;

    Tensor col = im2col(input, Kh, Kw, stride, stride, padding, padding);

    Tensor weightFlat = weight.reshape({C_out, C_in * Kh * Kw});
    Tensor outMat = matmul(weightFlat, col); // [C_out, N * outH * outW]

    Tensor outMatPerm = outMat.reshape({C_out, N, outH, outW}).permute({1, 0, 2, 3});
    Tensor Y = outMatPerm.reshape({N, C_out, outH, outW});

    if (bias.m_impl != nullptr && bias.size() > 0)
    {
        Tensor biasReshaped = bias;
        if (bias.rank() == 1)
        {
            biasReshaped = bias.reshape({1, C_out, 1, 1});
        }
        Y = add(Y, biasReshaped);
    }

    bool reqGrad = input.requiresGrad() || weight.requiresGrad() || (bias.m_impl && bias.requiresGrad());
    Y.m_impl->m_requiresGrad = reqGrad;

    if (reqGrad)
    {
        auto node = std::make_shared<Conv2dNode>();
        node->m_stride = stride;
        node->m_padding = padding;
        node->m_inputs = {input, weight};
        if (bias.m_impl != nullptr)
        {
            node->m_inputs.push_back(bias);
        }

        if (input.gradNode())
        {
            node->m_parents.push_back(input.gradNode());
        }
        if (weight.gradNode())
        {
            node->m_parents.push_back(weight.gradNode());
        }
        if (bias.m_impl && bias.gradNode())
        {
            node->m_parents.push_back(bias.gradNode());
        }

        Y.setGradNode(node);
        node->m_outputs.push_back(Y.m_impl);
    }

    return Y;
}


Tensor conv1d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding)
{
    if (input.rank() != 3 || weight.rank() != 3)
    {
        throw std::runtime_error("conv1d requires 3D input [N, C, L] and 3D weight [C_out, C_in, Kl].");
    }

    int N = input.dim(0);
    int C_in = input.dim(1);
    int L = input.dim(2);

    int C_out = weight.dim(0);
    int Kl = weight.dim(2);

    Tensor input4D = input.reshape({N, C_in, 1, L});
    Tensor weight4D = weight.reshape({C_out, C_in, 1, Kl});

    Tensor out4D = conv2d(input4D, weight4D, bias, stride, padding);
    int outL = out4D.dim(3);

    return out4D.reshape({N, C_out, outL});
}

Tensor conv3d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding)
{
    throw std::runtime_error("conv3d operation is currently a placeholder for N-D sliding window extension.");
}

Tensor maxPool2d(
    const Tensor &input,
    int kernelSize,
    int stride,
    int padding)
{
    if (input.rank() != 4)
    {
        throw std::runtime_error("maxPool2d requires 4D input [N, C, H, W].");
    }

    if (stride <= 0)
    {
        stride = kernelSize;
    }

    int N = input.dim(0);
    int C = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);

    int outH = (H + 2 * padding - kernelSize) / stride + 1;
    int outW = (W + 2 * padding - kernelSize) / stride + 1;

    Tensor Y({N, C, outH, outW}, input.requiresGrad());
    Tensor argMaxIndices({N, C, outH, outW});

    const double *inData = input.data();
    double *yData = Y.data();
    double *idxData = argMaxIndices.data();

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int oh = 0; oh < outH; ++oh)
            {
                for (int ow = 0; ow < outW; ++ow)
                {
                    double maxVal = -1e9;
                    int maxIdx = -1;

                    for (int kh = 0; kh < kernelSize; ++kh)
                    {
                        int ih = oh * stride - padding + kh;
                        for (int kw = 0; kw < kernelSize; ++kw)
                        {
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            {
                                int flatIn = ((n * C + c) * H + ih) * W + iw;
                                double val = inData[flatIn];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = flatIn;
                                }
                            }
                        }
                    }

                    int flatOut = ((n * C + c) * outH + oh) * outW + ow;
                    yData[flatOut] = maxVal;
                    idxData[flatOut] = static_cast<double>(maxIdx);
                }
            }
        }
    }

    bool reqGrad = input.requiresGrad();
    Y.m_impl->m_requiresGrad = reqGrad;

    if (reqGrad)
    {
        auto node = std::make_shared<MaxPool2dNode>();
        node->m_kernelSize = kernelSize;
        node->m_stride = stride;
        node->m_padding = padding;
        node->m_argMaxIndices = argMaxIndices;
        node->m_inputs = {input};

        if (input.gradNode())
        {
            node->m_parents.push_back(input.gradNode());
        }

        Y.setGradNode(node);
        node->m_outputs.push_back(Y.m_impl);
    }

    return Y;
}


Tensor relu(const Tensor &X)
{
    Tensor Y(X.shape(), X.requiresGrad());
    const double *xData = X.data();
    double *yData = Y.data();
    for (size_t i = 0; i < X.size(); ++i)
    {
        yData[i] = (xData[i] > 0.0) ? xData[i] : 0.0;
    }

    if (Y.requiresGrad())
    {
        auto node = std::make_shared<ReLUNode>();
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

Tensor transpose(const Tensor &X)
{
    Tensor Y = X.transpose();
    if (X.requiresGrad())
    {
        auto node = std::make_shared<TransposeNode>();
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


