#include "Layer.hpp"

namespace tardigrade::layer
{
    Dense::Dense(int inputSize, int outputSize, int batchSize, activation::ACTIVATION activation)
    {
        m_inputSize = inputSize;
        m_outputSize = outputSize;
        m_batchSize = batchSize;
        m_enumAct = activation;

        // Weights and bias require gradients
        m_weight = Tensor({m_inputSize, m_outputSize}, true);
        m_bias = Tensor({1, m_outputSize}, true);
        InitWeight();
    }

    Tensor Dense::Forward(const Tensor& input)
    {
        int rows = input.dim(0);
        int cols = (input.rank() == 1) ? 1 : input.dim(1);

        if (rows != m_inputSize)
        {
            throw std::runtime_error("Input dimension mismatch in autograd Dense::Forward.");
        }

        if (cols != m_batchSize)
        {
            SetBatchSize(cols);
        }

        /*
         * Forward linear activation calculation:
         *
         * \( Y = W^T X + b^T \cdot \mathbf{1} \)
         *
         * Mathematical breakdown:
         *  - \( X \): Input tensor of shape \( (D_{in}, N) \) where \( N = \text{m\_batchSize} \)
         *  - \( W \): Feature weight matrix of shape \( (D_{in}, D_{out}) \)
         *  - \( b \): Bias row vector of shape \( (1, D_{out}) \)
         *  - \( \mathbf{1} \): Row vector of ones of shape \( (1, N) \) for bias broadcasting
         *  - \( Y \): Logits output tensor of shape \( (D_{out}, N) \)
         */
        Tensor Y_feature = matmul(transpose(m_weight), input);

        // Broadcast bias vector by multiplying m_bias^T with a constant row of ones.
        // ones: shape (1, m_batchSize) initialized to 1.0
        Tensor ones({1, m_batchSize}, false);
        std::fill(ones.data(), ones.data() + ones.size(), 1.0);

        Tensor Y_bias = matmul(transpose(m_bias), ones);

        // Add feature and bias predictions: Y = Y_feature + Y_bias
        Tensor logits = add(Y_feature, Y_bias);

        // Apply activation functions using pure autograd ops
        Tensor output;
        if (m_enumAct == activation::ACTIVATION::ReLU)
        {
            output = relu(logits);
        }
        else if (m_enumAct == activation::ACTIVATION::Softmax)
        {
            output = softmax(logits);
        }
        else
        {
            output = logits;
        }

        return output;
    }

    std::vector<Tensor> Dense::GetParameters()
    {
        return { m_weight, m_bias };
    }

    void Dense::SetBatchSize(int batchSize)
    {
        if (m_batchSize == batchSize)
        {
            return;
        }
        m_batchSize = batchSize;
    }

    int Dense::GetBatchSize() const
    {
        return m_batchSize;
    }

    void Dense::InitWeight()
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        // He (Kaiming) initialization standard deviation: sqrt(2.0 / fan_in)
        double stddev = std::sqrt(2.0 / static_cast<double>(m_inputSize));
        std::normal_distribution<double> dist(0.0, stddev);

        for (int i = 0; i < m_inputSize; ++i)
        {
            for (int j = 0; j < m_outputSize; ++j)
            {
                m_weight(i, j) = dist(gen);
            }
        }

        // Initialize bias vector to zero
        for (int j = 0; j < m_outputSize; ++j)
        {
            m_bias(0, j) = 0.0;
        }
    }
}