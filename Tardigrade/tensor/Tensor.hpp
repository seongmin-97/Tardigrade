#pragma once
#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <Eigen/Dense>

namespace tardigrade
{
    using Shape = std::vector<int>;

    using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixMap = Eigen::Map<MatrixXdRowMajor>;
    using VectorMap = Eigen::Map<Eigen::VectorXd>;
    using ConstMatrixMap = Eigen::Map<const MatrixXdRowMajor>;
    using ConstVectorMap = Eigen::Map<const Eigen::VectorXd>;

    class TensorImpl;
    class Tensor;
    class Node;

    /**
     * @brief A shared data storage class holding the actual 1D double vector.
     */
    class Storage
    {
    private:
        std::shared_ptr<std::vector<double>> m_data;

    public:
        Storage(size_t size)
        {
            m_data = std::make_shared<std::vector<double>>(size, 0.0);
        }

        Storage()
        {
            m_data = std::make_shared<std::vector<double>>();
        }

        double* GetData()
        {
            return m_data->data();
        }

        const double* GetData() const
        {
            return m_data->data();
        }

        size_t GetSize() const
        {
            return m_data->size();
        }

        void Resize(size_t newSize)
        {
            m_data->resize(newSize, 0.0);
        }

        double& operator[](size_t index)
        {
            return (*m_data)[index];
        }

        const double& operator[](size_t index) const
        {
            return (*m_data)[index];
        }
    };

    /**
     * @brief Core implementation class of Tensor carrying shape, strides, data storage, and Autograd nodes.
     */
    class TensorImpl : public std::enable_shared_from_this<TensorImpl>
    {
    public:
        Shape m_shape;
        Shape m_strides;
        Storage m_storage;
        std::shared_ptr<Node> m_gradNode;
        std::shared_ptr<TensorImpl> m_grad;
        bool m_requiresGrad;

    public:
        TensorImpl(const Shape& shape, bool requiresGrad = false)
            : m_shape(shape), m_strides(calculateStrides(shape)), m_storage(0), m_gradNode(nullptr), m_grad(nullptr), m_requiresGrad(requiresGrad)
        {
            size_t totalSize = 1;
            for (int dim : shape)
            {
                totalSize *= dim;
            }
            m_storage.Resize(totalSize);
        }

        static Shape calculateStrides(const Shape& shape)
        {
            Shape strides(shape.size(), 1);
            if (!shape.empty())
            {
                strides.back() = 1;
                for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
                {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }
            return strides;
        }
    };

    /**
     * @brief High-performance multi-dimensional Tensor supporting Automatic Differentiation.
     */
    /**
     * @brief High-performance multi-dimensional Tensor supporting Automatic Differentiation.
     */
    class Tensor
    {
    public:
        std::shared_ptr<TensorImpl> m_impl;

    public:
        /** @brief Constructs a Tensor with a given shape and autograd flag */
        Tensor(const Shape& shape, bool requiresGrad = false);

        /** @brief Constructs an empty scalar-like Tensor */
        Tensor();

        /** @brief Constructs a Tensor from a shared implementation pointer */
        Tensor(std::shared_ptr<TensorImpl> impl);

        // Factory methods for Tensor initialization
        /** @brief Factory method creating a Tensor filled with zeros */
        static Tensor zeros(const Shape& shape, bool requiresGrad = false);

        /** @brief Factory method creating a Tensor filled with ones */
        static Tensor ones(const Shape& shape, bool requiresGrad = false);

        /** @brief Factory method creating a Tensor filled with a specific constant value */
        static Tensor fill(const Shape& shape, double value, bool requiresGrad = false);

        /** @brief Fills the tensor storage with a constant value */
        void fill(double value);

        // Dimensional getters
        /** @brief Returns the rank (number of dimensions) of the Tensor */
        int rank() const;

        /** @brief Returns the size of a specific dimension at the given index */
        int dim(int index) const;

        /** @brief Returns the shape vector of the Tensor */
        const Shape& shape() const;

        /** @brief Returns the memory stride vector for N-D indexing */
        const Shape& strides() const;

        /** @brief Returns a pointer to the raw 1D underlying double data array */
        double* data();

        /** @brief Returns a const pointer to the raw 1D underlying double data array */
        const double* data() const;

        /** @brief Returns the total number of scalar elements in the Tensor */
        size_t size() const;

        /** @brief Implicit conversion operator allowing 0D/1D single element Tensor to be converted to double */
        operator double() const
        {
            if (size() != 1)
            {
                throw std::runtime_error("Implicit conversion to double is only supported for Tensors with exactly 1 element.");
            }
            return data()[0];
        }

        /** @brief PyTorch-style scalar value extraction method for 0D/1D single element Tensors */
        double item() const;

        // Operator overloading for PyTorch-style Sub-Tensor & Element-wise access
        /** @brief PyTorch-style indexing: returns a Sub-Tensor view along dimension 0 (supports t[i][j] chaining) */
        Tensor operator[](int index) const
        {
            return select(0, index);
        }



        /** @brief Multidimensional variadic indexing operator returning a reference to a scalar value: t(i, j, k) */
        template<typename... Args>
        double& operator()(Args... indices)
        {
            std::vector<int> idx = { static_cast<int>(indices)... };
            return m_impl->m_storage[calculateIndex(idx)];
        }

        /** @brief Const multidimensional variadic indexing operator: t(i, j, k) */
        template<typename... Args>
        double operator()(Args... indices) const
        {
            std::vector<int> idx = { static_cast<int>(indices)... };
            return m_impl->m_storage[calculateIndex(idx)];
        }

        // Reshape, Permute & N-D Axis Slicing
        /** @brief Returns a reshaped Tensor with the specified new shape (element count must match) */
        Tensor reshape(const Shape& newShape) const;

        /** @brief Reorders (permutes) the axes of the N-D Tensor according to the given permutation vector */
        Tensor permute(const std::vector<int>& dims) const;

        /** @brief Extracts a Sub-Tensor slice along the specified dimension at a given index */
        Tensor select(int dim, int index) const;

        /** @brief Sets a Sub-Tensor slice along the specified dimension at a given index from a source Tensor */
        void setSelect(int dim, int index, const Tensor& src);

        /** @brief Extracts a range slice along the specified dimension: [start, end) */
        Tensor slice(int dim, int start, int end) const;

        /** @brief Sets a range slice along the specified dimension from a source Tensor */
        void setSlice(int dim, int start, int end, const Tensor& src);

        /** @brief Transposes the last two dimensions of the Tensor */
        Tensor transpose() const;

        /** @brief Swaps two specified dimensions (dim0 and dim1) of the Tensor */
        Tensor transpose(int dim0, int dim1) const;

        // In-place Arithmetic Operators
        /** @brief In-place addition of another Tensor */
        Tensor& operator+=(const Tensor& rhs);

        /** @brief In-place subtraction of another Tensor */
        Tensor& operator-=(const Tensor& rhs);

        // Eigen Expression Assignment Operator
        /** @brief Assigns data from an Eigen matrix/vector expression directly */
        template<typename Derived>
        Tensor& operator=(const Eigen::DenseBase<Derived>& expr)
        {
            if (rank() == 2)
            {
                asMatrix() = expr;
            }
            else if (rank() == 1)
            {
                asVector() = expr;
            }
            else
            {
                throw std::runtime_error("Assignment from Eigen expression is only supported for 1D or 2D tensors.");
            }
            return *this;
        }

        // Autograd Graph & Backward
        /** @brief Triggers automatic differentiation backpropagation starting from this scalar loss Tensor */
        void Backward();

        /** @brief Clears computational graph nodes and parent links to break reference cycles */
        void ClearGraph();

        /** @brief Resets the gradient accumulator of this Tensor to zero */
        void zeroGrad();

        /** @brief Creates a deep copy of the Tensor data and shape */
        Tensor clone() const;

        /** @brief Returns true if this Tensor tracks gradients for Autograd */
        bool requiresGrad() const;

        /** @brief Returns the gradient Tensor associated with this Tensor */
        Tensor grad() const;

        /** @brief Sets the gradient Tensor associated with this Tensor */
        void setGrad(const Tensor& g);

        /** @brief Returns the computational graph Node that created this Tensor */
        std::shared_ptr<Node> gradNode() const;

        /** @brief Sets the computational graph Node that created this Tensor */
        void setGradNode(std::shared_ptr<Node> node);

    // Internal Backend accessors (restricted to core tensor ops)
    public:
        /** @brief Returns an Eigen MatrixMap view for 2D Tensors */
        MatrixMap asMatrix();

        /** @brief Returns a const Eigen MatrixMap view for 2D Tensors */
        ConstMatrixMap asMatrix() const;

        /** @brief Returns an Eigen MatrixMap view with explicit row and col dimensions */
        MatrixMap asMatrix(int rows, int cols);

        /** @brief Returns a const Eigen MatrixMap view with explicit row and col dimensions */
        ConstMatrixMap asMatrix(int rows, int cols) const;

        /** @brief Returns an Eigen VectorMap view for 1D Tensors */
        VectorMap asVector();

        /** @brief Returns a const Eigen VectorMap view for 1D Tensors */
        ConstVectorMap asVector() const;

    public:
        /** @brief Calculates 1D linear storage index from N-D multi-index using strides */
        int calculateIndex(const std::vector<int>& indices) const;

        /** @brief Normalizes a potentially negative axis index into a valid non-negative axis index */
        static int normalizeAxis(int axis, int rank);
    };

    // Helper functions for Broadcasting
    /** @brief Computes the resulting broadcast shape from two input shapes according to NumPy rules */
    Shape broadcastShapes(const Shape& shapeA, const Shape& shapeB);

    /** @brief Returns true if two shapes can be broadcast together */
    bool isBroadcastable(const Shape& shapeA, const Shape& shapeB);

    // Primitive Forward Operations
    /** @brief Matrix Multiplication: Y = A * B */
    Tensor matmul(const Tensor& A, const Tensor& B);

    /** @brief Element-wise or Broadcasted Addition: Y = A + B */
    Tensor add(const Tensor& A, const Tensor& B);

    /** @brief Element-wise or Broadcasted Subtraction: Y = A - B */
    Tensor sub(const Tensor& A, const Tensor& B);

    /** @brief Element-wise or Broadcasted Multiplication: Y = A * B */
    Tensor mul(const Tensor& A, const Tensor& B);

    /** @brief Element-wise or Broadcasted Division: Y = A / B */
    Tensor div(const Tensor& A, const Tensor& B);

    /** @brief Element-wise Natural Exponential function: Y = exp(X) */
    Tensor exp(const Tensor& X);

    /** @brief Element-wise Natural Logarithm function: Y = log(X) */
    Tensor log(const Tensor& X);

    /** @brief Summation of Tensor elements along a single axis */
    Tensor sum(const Tensor& X, int axis = -1, bool keepDims = false);

    /** @brief Summation of Tensor elements along multiple axes */
    Tensor sum(const Tensor& X, const std::vector<int>& axes, bool keepDims = false);

    /** @brief Element-wise Rectified Linear Unit activation: Y = max(0, X) */
    Tensor relu(const Tensor& X);

    /** @brief Transposes the last two dimensions of Tensor X */
    Tensor transpose(const Tensor& X);

    /** @brief Permutes (reorders) dimensions of Tensor X */
    Tensor permute(const Tensor& X, const std::vector<int>& dims);

    /** @brief Slices rows of a 2D Tensor X: [startRow, endRow) */
    Tensor slice(const Tensor& X, int startRow, int endRow);

    /** @brief Concatenates a list of Tensors along a specified axis */
    Tensor concat(const std::vector<Tensor>& tensors, int axis = 0);

    // Convolution & Pooling Primitive Operations
    /** @brief Converts 4D image tensor patches [N, C, H, W] into a 2D column matrix for fast GEMM convolution */
    Tensor im2col(const Tensor& input, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW);

    /** @brief Reconstructs 4D image tensor [N, C, H, W] from a 2D column gradient matrix for convolution backward */
    Tensor col2im(const Tensor& col, const Shape& inputShape, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW);

    /** @brief 2D Convolution operation: input [N, C_in, H, W], weight [C_out, C_in, Kh, Kw] */
    Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias = Tensor(), int stride = 1, int padding = 0);

    /** @brief 1D Convolution operation: input [N, C_in, L], weight [C_out, C_in, Kl] */
    Tensor conv1d(const Tensor& input, const Tensor& weight, const Tensor& bias = Tensor(), int stride = 1, int padding = 0);

    /** @brief 3D Convolution operation placeholder */
    Tensor conv3d(const Tensor& input, const Tensor& weight, const Tensor& bias = Tensor(), int stride = 1, int padding = 0);

    /** @brief 2D Max Pooling operation: extracts local maximum values over sliding window */
    Tensor maxPool2d(const Tensor& input, int kernelSize, int stride = -1, int padding = 0);

    /** @brief 2D Average Pooling operation: computes local mean values over sliding window */
    Tensor avgPool2d(const Tensor& input, int kernelSize, int stride = -1, int padding = 0);

    // Global Tensor arithmetic operators (Tensor vs Tensor, Tensor vs Scalar)
    /** @brief Global operator+ for Tensor + Tensor (supports Broadcasting) */
    Tensor operator+(const Tensor& lhs, const Tensor& rhs);

    /** @brief Global operator+ for Tensor + double scalar */
    Tensor operator+(const Tensor& lhs, double scalar);

    /** @brief Global operator+ for double scalar + Tensor */
    Tensor operator+(double scalar, const Tensor& rhs);

    /** @brief Global operator- for Tensor - Tensor (supports Broadcasting) */
    Tensor operator-(const Tensor& lhs, const Tensor& rhs);

    /** @brief Global operator- for Tensor - double scalar */
    Tensor operator-(const Tensor& lhs, double scalar);

    /** @brief Global operator- for double scalar - Tensor */
    Tensor operator-(double scalar, const Tensor& rhs);

    /** @brief Global operator* for Element-wise Tensor * Tensor (supports Broadcasting) */
    Tensor operator*(const Tensor& lhs, const Tensor& rhs);

    /** @brief Global operator* for Tensor * double scalar */
    Tensor operator*(const Tensor& lhs, double scalar);

    /** @brief Global operator* for double scalar * Tensor */
    Tensor operator*(double scalar, const Tensor& rhs);

    /** @brief Global operator/ for Element-wise Tensor / Tensor (supports Broadcasting) */
    Tensor operator/(const Tensor& lhs, const Tensor& rhs);

    /** @brief Global operator/ for Tensor / double scalar */
    Tensor operator/(const Tensor& lhs, double scalar);

    /** @brief Global operator/ for double scalar / Tensor */
    Tensor operator/(double scalar, const Tensor& rhs);

    // Element-wise Comparison operators
    /** @brief Element-wise comparison operator== returning 1.0 (true) or 0.0 (false) Tensor */
    Tensor operator==(const Tensor& lhs, const Tensor& rhs);
    Tensor operator==(const Tensor& lhs, double scalar);
    Tensor operator==(double scalar, const Tensor& rhs);

    /** @brief Element-wise comparison operator!= returning 1.0 (true) or 0.0 (false) Tensor */
    Tensor operator!=(const Tensor& lhs, const Tensor& rhs);
    Tensor operator!=(const Tensor& lhs, double scalar);
    Tensor operator!=(double scalar, const Tensor& rhs);

    /** @brief Element-wise comparison operator< returning 1.0 (true) or 0.0 (false) Tensor */
    Tensor operator<(const Tensor& lhs, const Tensor& rhs);
    Tensor operator<(const Tensor& lhs, double scalar);
    Tensor operator<(double scalar, const Tensor& rhs);

    /** @brief Element-wise comparison operator> returning 1.0 (true) or 0.0 (false) Tensor */
    Tensor operator>(const Tensor& lhs, const Tensor& rhs);
    Tensor operator>(const Tensor& lhs, double scalar);
    Tensor operator>(double scalar, const Tensor& rhs);

    /** @brief Element-wise comparison operator<= returning 1.0 (true) or 0.0 (false) Tensor */
    Tensor operator<=(const Tensor& lhs, const Tensor& rhs);
    Tensor operator<=(const Tensor& lhs, double scalar);
    Tensor operator<=(double scalar, const Tensor& rhs);

    /** @brief Element-wise comparison operator>= returning 1.0 (true) or 0.0 (false) Tensor */
    Tensor operator>=(const Tensor& lhs, const Tensor& rhs);
    Tensor operator>=(const Tensor& lhs, double scalar);
    Tensor operator>=(double scalar, const Tensor& rhs);
}


