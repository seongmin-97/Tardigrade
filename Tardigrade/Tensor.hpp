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
        std::shared_ptr<Node> m_creator;
        std::shared_ptr<TensorImpl> m_grad;
        bool m_requiresGrad;

    public:
        TensorImpl(const Shape& shape, bool requiresGrad = false)
            : m_shape(shape), m_storage(0), m_creator(nullptr), m_grad(nullptr), m_requiresGrad(requiresGrad)
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
     *
     * Wrapper class around std::shared_ptr<TensorImpl> matching PyTorch's signature.
     */
    class Tensor
    {
    public:
        std::shared_ptr<TensorImpl> m_impl;

    public:
        /**
         * @brief Constructs a Tensor with a given shape.
         */
        Tensor(const Shape& shape, bool requiresGrad = false);

        /**
         * @brief Default constructor creating an empty Tensor.
         */
        Tensor();

        /**
         * @brief Constructs a Tensor from a shared implementation pointer.
         */
        Tensor(std::shared_ptr<TensorImpl> impl);

        /**
         * @brief Shared pointer maps to Eigen matrix representation.
         */
        MatrixMap asMatrix(int rows, int cols);
        ConstMatrixMap asMatrix(int rows, int cols) const;

        /**
         * @brief Shared pointer maps to Eigen vector representation.
         */
        VectorMap asVector();
        ConstVectorMap asVector() const;

        // Basic Dimensional getters
        int rank() const;
        int dim(int index) const;
        const Shape& shape() const;
        double* data();
        const double* data() const;
        size_t size() const;

        // Operator overloading for 1D access
        double& operator[](size_t index);
        const double& operator[](size_t index) const;

        // Multi-dimensional index accessors
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

        /**
         * @brief Initiates backpropagation from this loss Tensor.
         */
        void Backward();

        /**
         * @brief Breaks references within the computational graph to free memory.
         */
        void ClearGraph();

        /**
         * @brief Zero out the current accumulated gradient.
         */
        void zeroGrad();

        /**
         * @brief Creates a deep copy of the Tensor's data, detached from the graph.
         */
        Tensor clone() const;

        // Autograd status getters
        bool requiresGrad() const;
        Tensor grad() const;
        void setGrad(const Tensor& g);
        std::shared_ptr<Node> creator() const;
        void setCreator(std::shared_ptr<Node> node);

    private:
        int calculateIndex(const std::vector<int>& indices) const;
    };
}
