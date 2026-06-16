#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <functional>
#include <stdexcept>
#include <Eigen/Dense>

#include "Storage.hpp"

namespace tardigrade::autograd
{
    using Shape = std::vector<int>;

    using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixMap = Eigen::Map<MatrixXdRowMajor>;
    using VectorMap = Eigen::Map<Eigen::VectorXd>;
    using ConstMatrixMap = Eigen::Map<const MatrixXdRowMajor>;
    using ConstVectorMap = Eigen::Map<const Eigen::VectorXd>;

    class TensorImpl;
    class Tensor;

    /**
     * @brief Abstract base class representing an operation node in the computational graph.
     */
    class Node
    {
    public:
        virtual ~Node() = default;

        /**
         * @brief Computes local gradients and propagates them backward.
         * @param gradOutputs The gradients w.r.t the outputs of this node.
         * @return The gradients w.r.t the inputs of this node.
         */
        virtual std::vector<Tensor> Backward(const std::vector<Tensor>& gradOutputs) = 0;

        /**
         * @brief Clears inputs and parents to break reference cycles and free memory.
         */
        void ClearEdges()
        {
            m_inputs.clear();
            m_parents.clear();
            m_outputs.clear();
        }

    public:
        std::vector<Tensor> m_inputs;                     ///< Input tensors to this operation
        std::vector<std::shared_ptr<Node>> m_parents;      ///< Parent nodes that created the inputs
        std::vector<std::weak_ptr<TensorImpl>> m_outputs;  ///< Weak pointers to output implementations to prevent cycles
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

    // ------------------------------------------------------------
    // Computational Ops (Forward Functions)
    // ------------------------------------------------------------

    Tensor matmul(const Tensor& A, const Tensor& B);
    Tensor add(const Tensor& A, const Tensor& B);
    Tensor relu(const Tensor& X);
    Tensor softmax(const Tensor& X);
    Tensor mse_loss(const Tensor& pred, const Tensor& target);
    Tensor transpose(const Tensor& X);
    Tensor slice(const Tensor& X, int startRow, int endRow);

    // Operator overloads matching standard syntax
    Tensor operator+(const Tensor& A, const Tensor& B);
}
