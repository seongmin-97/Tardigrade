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
     * @brief Core implementation class of Tensor carrying shape, data storage, and Autograd nodes.
     */
    class TensorImpl : public std::enable_shared_from_this<TensorImpl>
    {
    public:
        Shape m_shape;
        Storage m_storage;
        std::shared_ptr<Node> m_gradNode;
        std::shared_ptr<TensorImpl> m_grad;
        bool m_requiresGrad;

    public:
        TensorImpl(const Shape& shape, bool requiresGrad = false)
            : m_shape(shape), m_storage(0), m_gradNode(nullptr), m_grad(nullptr), m_requiresGrad(requiresGrad)
        {
            size_t totalSize = 1;
            for (int dim : shape)
            {
                totalSize *= dim;
            }
            m_storage.Resize(totalSize);
        }
    };

    /**
     * @brief High-performance multi-dimensional Tensor supporting Automatic Differentiation.
     */
    class Tensor
    {
    public:
        std::shared_ptr<TensorImpl> m_impl;

    public:
        Tensor(const Shape& shape, bool requiresGrad = false);
        Tensor();
        Tensor(std::shared_ptr<TensorImpl> impl);

        // Factory methods for Tensor initialization
        static Tensor zeros(const Shape& shape, bool requiresGrad = false);
        static Tensor ones(const Shape& shape, bool requiresGrad = false);
        static Tensor fill(const Shape& shape, double value, bool requiresGrad = false);

        // Fill instance data with a constant value
        void fill(double value);

        // Dimensional getters
        int rank() const;
        int dim(int index) const;
        const Shape& shape() const;
        double* data();
        const double* data() const;
        size_t size() const;

        // Operator overloading for 1D/Multi-D access
        double& operator[](size_t index);
        const double& operator[](size_t index) const;

        template<typename... Args>
        double& operator()(Args... indices)
        {
            std::vector<int> idx = { static_cast<int>(indices)... };
            return m_impl->m_storage[calculateIndex(idx)];
        }

        template<typename... Args>
        double operator()(Args... indices) const
        {
            std::vector<int> idx = { static_cast<int>(indices)... };
            return m_impl->m_storage[calculateIndex(idx)];
        }

        // Reshape & N-D Axis Slicing
        Tensor reshape(const Shape& newShape) const;
        Tensor slice(int axis, int start, int end) const;
        void setSlice(int axis, int index, const Tensor& src);
        Tensor row(int index) const;
        void setRow(int index, const Tensor& src);
        Tensor col(int index) const;
        void setCol(int index, const Tensor& src);
        Tensor transpose() const;

        // In-place Arithmetic Operators
        Tensor& operator+=(const Tensor& rhs);
        Tensor& operator-=(const Tensor& rhs);

        // Eigen Expression Assignment Operator
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
        void Backward();
        void ClearGraph();
        void zeroGrad();
        Tensor clone() const;
        bool requiresGrad() const;
        Tensor grad() const;
        void setGrad(const Tensor& g);
        std::shared_ptr<Node> gradNode() const;
        void setGradNode(std::shared_ptr<Node> node);

    // Internal Backend accessors (restricted to core tensor ops)
    public:
        MatrixMap asMatrix();
        ConstMatrixMap asMatrix() const;
        MatrixMap asMatrix(int rows, int cols);
        ConstMatrixMap asMatrix(int rows, int cols) const;
        VectorMap asVector();
        ConstVectorMap asVector() const;

    private:
        int calculateIndex(const std::vector<int>& indices) const;
    };

    // Primitive Forward Operations
    Tensor matmul(const Tensor& A, const Tensor& B);
    Tensor add(const Tensor& A, const Tensor& B);
    Tensor sub(const Tensor& A, const Tensor& B);
    Tensor mul(const Tensor& A, const Tensor& B);
    Tensor div(const Tensor& A, const Tensor& B);
    Tensor div(const Tensor& A, double scalar);
    Tensor exp(const Tensor& X);
    Tensor log(const Tensor& X);
    Tensor sum(const Tensor& X, int axis = -1);
    Tensor relu(const Tensor& X);
    Tensor transpose(const Tensor& X);
    Tensor slice(const Tensor& X, int startRow, int endRow);
    Tensor concat(const std::vector<Tensor>& tensors, int axis = 0);

    // Global Tensor arithmetic operators
    Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    Tensor operator-(const Tensor& lhs, const Tensor& rhs);
    Tensor operator*(const Tensor& lhs, const Tensor& rhs);
    Tensor operator*(const Tensor& lhs, double scalar);
    Tensor operator*(double scalar, const Tensor& rhs);
    Tensor operator/(const Tensor& lhs, const Tensor& rhs);
    Tensor operator/(const Tensor& lhs, double scalar);
}
