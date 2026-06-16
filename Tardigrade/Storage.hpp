#pragma once
#include <vector>
#include <memory>

namespace tardigrade::autograd
{
    /**
     * @brief A shared data storage class holding the actual 1D double vector.
     *
     * This class manages the raw data memory using a shared pointer, allowing
     * multiple Tensor instances to share the same underlying storage (e.g., for views).
     */
    class Storage
    {
    private:
        std::shared_ptr<std::vector<double>> m_data;

    public:
        /**
         * @brief Constructs storage with a specific size initialized to 0.0.
         */
        Storage(size_t size)
        {
            m_data = std::make_shared<std::vector<double>>(size, 0.0);
        }

        /**
         * @brief Default constructor creating empty storage.
         */
        Storage()
        {
            m_data = std::make_shared<std::vector<double>>();
        }

        /**
         * @brief Accesses the raw double pointer of the storage.
         */
        double* GetData()
        {
            return m_data->data();
        }

        /**
         * @brief Accesses the constant raw double pointer of the storage.
         */
        const double* GetData() const
        {
            return m_data->data();
        }

        /**
         * @brief Returns the total number of elements in the storage.
         */
        size_t GetSize() const
        {
            return m_data->size();
        }

        /**
         * @brief Resizes the storage in place.
         */
        void Resize(size_t newSize)
        {
            m_data->resize(newSize, 0.0);
        }

        /**
         * @brief Element-wise access operator.
         */
        double& operator[](size_t index)
        {
            return (*m_data)[index];
        }

        /**
         * @brief Constant element-wise access operator.
         */
        const double& operator[](size_t index) const
        {
            return (*m_data)[index];
        }
    };
}
