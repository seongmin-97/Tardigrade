#pragma once
#include <vector>
#include <numeric>

#include <Eigen/Dense>

namespace tardigrade
{
    using Data = std::vector<double>;
    using Shape = std::vector<int>;

    using MatrixMap = Eigen::Map<Eigen::MatrixXd>;
    using VectorMap = Eigen::Map<Eigen::VectorXd>;
    using ConstMatrixMap = Eigen::Map<const Eigen::MatrixXd>;
    using ConstVectorMap = Eigen::Map<const Eigen::VectorXd>;

    class Tensor
    {
    private:
        Data m_data;
        Shape m_shape;

    public:
        Tensor(const std::vector<int>& shape) : m_shape(shape)
        {
            long long totalSize = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int>());
            m_data.resize(totalSize, 0.0);
        }

        Tensor() = default;

        MatrixMap asMatrix(int rows, int cols)
        {
            if (rows * cols != m_data.size())
                throw std::runtime_error("Tensor size mismatch for Matrix mapping");

            return MatrixMap(m_data.data(), rows, cols);
        }

        ConstMatrixMap asMatrix(int rows, int cols) const {
            if (rows * cols != static_cast<int>(m_data.size()))
                throw std::runtime_error("Size mismatch");
            return ConstMatrixMap(m_data.data(), rows, cols);
        }

        VectorMap asVector()
        {
            return VectorMap(m_data.data(), m_data.size());
        }

        ConstVectorMap asVector() const 
        {
            return ConstVectorMap(m_data.data(), static_cast<Eigen::Index>(m_data.size()));
        }

        void reshape(const std::vector<int>& newShape)
        {
            long long newSize = std::accumulate(newShape.begin(), newShape.end(), 1LL, std::multiplies<int>());

            if (newSize != m_data.size())
                m_data.resize(newSize);

            m_shape = newShape;
        }

        Tensor transpose() const 
        {
            if (m_shape.size() != 2) 
                throw std::runtime_error("Transpose is only supported for 2D tensors.");

            Tensor result({ m_shape[1], m_shape[0] });

            ConstMatrixMap mat(m_data.data(), m_shape[0], m_shape[1]);
            result.asMatrix(m_shape[1], m_shape[0]) = mat.transpose();

            return result;
        }

        auto row(int i) 
        {
            if (m_shape.size() != 2) 
                throw std::runtime_error("row() can only be called on 2D tensors (Matrices).");
            
            return asMatrix(m_shape[0], m_shape[1]).row(i);
        }

        auto col(int j) {
            if (m_shape.size() != 2) 
                throw std::runtime_error("col() can only be called on 2D tensors (Matrices).");
           
            return asMatrix(m_shape[0], m_shape[1]).col(j);
        }

        int rank() const { return m_shape.size(); }
        int dim(int index) const { return m_shape.at(index); }
        const std::vector<int>& shape() const { return m_shape; }
        double* data() { return m_data.data(); }
        const double* data() const { return m_data.data(); }
        size_t size() const { return m_data.size(); }
        
        double& operator[](size_t index) { return m_data[index]; }

        template<typename... Args>
        double& operator()(Args... indices) 
        {
            std::vector<int> idx = { static_cast<int>(indices)... };
            return m_data[calculateIndex(idx)];
        }

        Tensor operator*(const Tensor& other) const 
        {
            int a_rows = this->dim(0);
            int a_cols = (this->rank() == 1) ? 1 : this->dim(1);

            int b_rows = other.dim(0);
            int b_cols = (other.rank() == 1) ? 1 : other.dim(1);

            if (a_cols != b_rows) 
                throw std::runtime_error("Matrix-Vector multiplication dimension mismatch.");

            std::vector<int> resShape;
            if (other.rank() == 1) resShape = { a_rows };
            else resShape = { a_rows, b_cols };

            Tensor result(resShape);

            ConstMatrixMap matA(this->data(), a_rows, a_cols);
            ConstMatrixMap matB(other.data(), b_rows, b_cols);

            result.asMatrix(a_rows, b_cols) = matA * matB;

            return result;
        }

        Tensor clampedMin(double threshold) const 
        {
            Tensor result(m_shape);
            result.asVector() = asVector().cwiseMax(threshold);

            return result;
        }

        Tensor step() const 
        {
            Tensor result(m_shape);
            result.asVector() = (asVector().array() > 0.0).cast<double>();

            return result;
        }

    private:
        int calculateIndex(const std::vector<int>& indices) const 
        {
            int flatIndex = 0;
            int stride = 1;

            for (int i = m_shape.size() - 1; i >= 0; --i) {
                flatIndex += indices[i] * stride;
                stride *= m_shape[i];
            }

            return flatIndex;
        }
    };
}
