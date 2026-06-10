#pragma once
#include <vector>
#include <numeric>

#include <Eigen/Dense>

namespace tardigrade
{
    using Data = std::vector<double>;
    using Shape = std::vector<int>;

    using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixMap = Eigen::Map<MatrixXdRowMajor>;
    using VectorMap = Eigen::Map<Eigen::VectorXd>;
    using ConstMatrixMap = Eigen::Map<const MatrixXdRowMajor>;
    using ConstVectorMap = Eigen::Map<const Eigen::VectorXd>;

    /**
     * @brief A multi-dimensional tensor class wrapping Eigen vectors/matrices.
     *
     * This class acts as the foundational data structure for the Tardigrade framework.
     * It manages multi-dimensional data in a flat `std::vector` and provides
     * zero-copy mappings to Eigen::Matrix and Eigen::Vector for high-performance
     * operations.
     */
    class Tensor
    {
    private:
        Data m_data;
        Shape m_shape;

    public:
        /**
         * @brief Constructs a tensor with the given shape.
         * @param shape The dimensions of the tensor (e.g., {rows, cols}).
         */
        Tensor(const std::vector<int>& shape) : m_shape(shape)
        {
            long long totalSize = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int>());
            m_data.resize(totalSize, 0.0);
        }

        /**
         * @brief Default constructor.
         */
        Tensor() = default;

        /**
         * @brief Maps the underlying data to an Eigen Matrix.
         * @param rows The number of rows.
         * @param cols The number of columns.
         * @return Eigen::Map<Eigen::MatrixXd> mapped to the tensor's memory.
         * @throw std::runtime_error if dimensions do not match the total size.
         */
        MatrixMap asMatrix(int rows, int cols)
        {
            if (rows * cols != m_data.size())
                throw std::runtime_error("Tensor size mismatch for Matrix mapping");

            return MatrixMap(m_data.data(), rows, cols);
        }

        /**
         * @brief Maps the underlying data to a constant Eigen Matrix.
         * @param rows The number of rows.
         * @param cols The number of columns.
         * @return Eigen::Map<const Eigen::MatrixXd> mapped to the tensor's memory.
         * @throw std::runtime_error if dimensions do not match the total size.
         */
        ConstMatrixMap asMatrix(int rows, int cols) const {
            if (rows * cols != static_cast<int>(m_data.size()))
                throw std::runtime_error("Size mismatch");
            return ConstMatrixMap(m_data.data(), rows, cols);
        }

        /**
         * @brief Maps the underlying data to an Eigen Vector.
         * @return Eigen::Map<Eigen::VectorXd> mapped to the tensor's memory.
         */
        VectorMap asVector()
        {
            return VectorMap(m_data.data(), m_data.size());
        }

        /**
         * @brief Maps the underlying data to a constant Eigen Vector.
         * @return Eigen::Map<const Eigen::VectorXd> mapped to the tensor's memory.
         */
        ConstVectorMap asVector() const 
        {
            return ConstVectorMap(m_data.data(), static_cast<Eigen::Index>(m_data.size()));
        }

        /**
         * @brief Reshapes the tensor in-place.
         * @param newShape The new dimensions for the tensor.
         * @throw std::runtime_error if the new size doesn't match the original size.
         */
        void reshape(const std::vector<int>& newShape)
        {
            long long newSize = std::accumulate(newShape.begin(), newShape.end(), 1LL, std::multiplies<int>());

            if (newSize != static_cast<long long>(m_data.size()))
            {
                m_data.resize(newSize, 0.0);
            }

            m_shape = newShape;
        }

        /**
         * @brief Transposes a 2D tensor.
         * @return A new transposed Tensor.
         * @throw std::runtime_error if the tensor is not 2-dimensional.
         */
        Tensor transpose() const 
        {
            if (m_shape.size() != 2) 
                throw std::runtime_error("Transpose is only supported for 2D tensors.");

            Tensor result({ m_shape[1], m_shape[0] });

            ConstMatrixMap mat(m_data.data(), m_shape[0], m_shape[1]);
            result.asMatrix(m_shape[1], m_shape[0]) = mat.transpose();

            return result;
        }

        /**
         * @brief Retrieves a specific row from a 2D tensor.
         * @param i The row index.
         * @return Eigen representation of the row.
         * @throw std::runtime_error if the tensor is not 2-dimensional.
         */
        auto row(int i) 
        {
            if (m_shape.size() != 2) 
                throw std::runtime_error("row() can only be called on 2D tensors (Matrices).");
            
            return asMatrix(m_shape[0], m_shape[1]).row(i);
        }

        /**
         * @brief Retrieves a specific column from a 2D tensor.
         * @param j The column index.
         * @return Eigen representation of the column.
         * @throw std::runtime_error if the tensor is not 2-dimensional.
         */
        auto col(int j) {
            if (m_shape.size() != 2) 
                throw std::runtime_error("col() can only be called on 2D tensors (Matrices).");
           
            return asMatrix(m_shape[0], m_shape[1]).col(j);
        }

        /**
         * @brief Returns the rank (number of dimensions) of the tensor.
         * @return Number of dimensions.
         */
        int rank() const { return m_shape.size(); }

        /**
         * @brief Returns the size of a specific dimension.
         * @param index The dimension index.
         * @return Size of the specified dimension.
         */
        int dim(int index) const { return m_shape.at(index); }

        /**
         * @brief Returns the shape of the tensor.
         * @return Shape vector.
         */
        const std::vector<int>& shape() const { return m_shape; }

        /**
         * @brief Returns a pointer to the underlying raw data.
         * @return Raw data pointer.
         */
        double* data() { return m_data.data(); }

        /**
         * @brief Returns a constant pointer to the underlying raw data.
         * @return Constant raw data pointer.
         */
        const double* data() const { return m_data.data(); }

        /**
         * @brief Returns the total number of elements in the tensor.
         * @return Total size.
         */
        size_t size() const { return m_data.size(); }
        
        double& operator[](size_t index) { return m_data[index]; }
        const double& operator[](size_t index) const { return m_data[index]; }

        template<typename... Args>
        double& operator()(Args... indices) 
        {
            std::vector<int> idx = { static_cast<int>(indices)... };
            return m_data[calculateIndex(idx)];
        }

        /**
         * @brief Accesses an element given multi-dimensional indices (const version).
         * @tparam Args Parameter pack for indices.
         * @param indices Indices of the element.
         * @return The element value.
         */
        template<typename... Args>
        double operator()(Args... indices) const
        {
            std::vector<int> idx = { static_cast<int>(indices)... };
            return m_data[calculateIndex(idx)];
        }

        /**
         * @brief Performs matrix multiplication.
         * @param other The right-hand side tensor.
         * @return Resulting Tensor of the multiplication.
         * @throw std::runtime_error on dimension mismatch.
         */
        Tensor operator*(const Tensor& other) const 
        {
            int a_rows = this->dim(0);
            int a_cols = (this->rank() == 1) ? 1 : this->dim(1);

            int b_rows = other.dim(0);
            int b_cols = (other.rank() == 1) ? 1 : other.dim(1);

            if (a_cols != b_rows) 
                throw std::runtime_error("Matrix multiplication dimension mismatch. Expected " + std::to_string(a_cols) + " cols, but got " + std::to_string(b_rows) + " rows.");

            std::vector<int> resShape;
            if (other.rank() == 1) resShape = { a_rows };
            else resShape = { a_rows, b_cols };

            Tensor result(resShape);

            ConstMatrixMap matA(this->data(), a_rows, a_cols);
            ConstMatrixMap matB(other.data(), b_rows, b_cols);

            result.asMatrix(a_rows, b_cols) = matA * matB;

            return result;
        }

        /**
         * @brief Performs element-wise addition.
         * @param other The right-hand side tensor.
         * @return Resulting Tensor.
         * @throw std::runtime_error on shape mismatch.
         */
        Tensor operator+(const Tensor& other) const 
        {
            if (m_shape != other.m_shape)
                throw std::runtime_error("Shape mismatch for element-wise addition.");

            Tensor result(m_shape);
            result.asVector().array() = this->asVector().array() + other.asVector().array();
        
            return result;
        }

        /**
         * @brief Performs element-wise subtraction.
         * @param other The right-hand side tensor.
         * @return Resulting Tensor.
         * @throw std::runtime_error on shape mismatch.
         */
        Tensor operator-(const Tensor& other) const 
        {
            if (m_shape != other.m_shape)
                throw std::runtime_error("Shape mismatch for element-wise subtraction.");

            Tensor result(m_shape);
            result.asVector().array() = this->asVector().array() - other.asVector().array();
        
            return result;
        }

        /**
         * @brief Multiplies tensor elements by a scalar.
         * @param scalar The scalar value.
         * @return Resulting Tensor.
         */
        Tensor operator*(double scalar) const 
        {
            Tensor result(m_shape);
            result.asVector() = this->asVector() * scalar;

            return result;
        }

        friend Tensor operator*(double scalar, const Tensor& tensor) 
        {
            return tensor * scalar;
        }

        /**
         * @brief Divides tensor elements by a scalar.
         * @param scalar The scalar value.
         * @return Resulting Tensor.
         * @throw std::runtime_error if scalar is zero.
         */
        Tensor operator/(double scalar) const 
        {
            if (scalar == 0.0)
                throw std::runtime_error("Division by zero.");

            Tensor result(m_shape);
            result.asVector() = this->asVector() / scalar;

            return result;
        }

        /**
         * @brief Applies a minimum clamp (lower bound) to all elements.
         * @param threshold The minimum value allowed.
         * @return Clamped Tensor.
         */
        Tensor clampedMin(double threshold) const 
        {
            Tensor result(m_shape);
            result.asVector() = asVector().cwiseMax(threshold);

            return result;
        }

        /**
         * @brief Applies a unit step function to all elements.
         * @return Tensor where elements > 0 are 1.0, otherwise 0.0.
         */
        Tensor step() const 
        {
            Tensor result(m_shape);
            result.asVector() = (asVector().array() > 0.0).cast<double>();

            return result;
        }

        /**
         * @brief Performs element-wise (Hadamard) multiplication.
         * @param other The right-hand side tensor.
         * @return Resulting Tensor.
         * @throw std::runtime_error on shape mismatch.
         */
        Tensor cwiseMul(const Tensor& other) const 
        {
            if (m_shape != other.m_shape)
                throw std::runtime_error("Shape mismatch for element-wise multiplication.");

            Tensor result(m_shape);
            result.asVector().array() = this->asVector().array() * other.asVector().array();

            return result;
        }

        /**
         * @brief Performs element-wise division.
         * @param other The right-hand side tensor.
         * @return Resulting Tensor.
         * @throw std::runtime_error on shape mismatch.
         */
        Tensor cwiseDiv(const Tensor& other) const 
        {
            if (m_shape != other.m_shape)
                throw std::runtime_error("Shape mismatch for element-wise division.");

            Tensor result(m_shape);
            result.asVector().array() = this->asVector().array() / other.asVector().array();
        
            return result;
        }

    private:
        /**
         * @brief Calculates a flat index from a multi-dimensional index array.
         * @param indices Multi-dimensional indices.
         * @return Flattened index.
         */
        int calculateIndex(const std::vector<int>& indices) const 
        {
            int flatIndex = 0;
            int stride = 1;

            for (int i = m_shape.size() - 1; i >= 0; --i) 
            {
                flatIndex += indices[i] * stride;
                stride *= m_shape[i];
            }

            return flatIndex;
        }
    };
}
