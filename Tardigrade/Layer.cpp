#include "Layer.hpp"

namespace tardigrade::layer
{
    Dense::Dense(int inputSize, int outputSize, int batchSize, activation::ACTIVATION activation)
    {
        m_inputSize = inputSize + 1; // augmented input size including bias
        m_outputSize = outputSize;
        m_batchSize = batchSize;
        m_enumAct = activation;

        // Weights require gradients
        m_weight = Tensor({m_inputSize, m_outputSize}, true);
        InitWeight();
    }

    Tensor Dense::Forward(const Tensor& input)
    {
        int rows = input.dim(0);
        int cols = (input.rank() == 1) ? 1 : input.dim(1);

        if (rows != m_inputSize - 1)
        {
            throw std::runtime_error("Input dimension mismatch in autograd Dense::Forward.");
        }

        if (cols != m_batchSize)
        {
            SetBatchSize(cols);
        }

        // Split m_weight into feature weights (bottom rows) and bias weights (top row)
        // feature_W: shape (m_inputSize - 1, m_outputSize)
        // bias_W: shape (1, m_outputSize)
        Tensor feature_W = slice(m_weight, 1, m_inputSize);
        Tensor bias_W = slice(m_weight, 0, 1);

        // Linear activation calculation: Y_feature = W_feature^T * X
        Tensor Y_feature = matmul(transpose(feature_W), input);

        // Broadcast bias vector by multiplying bias_W^T with a constant row of ones.
        // ones: shape (1, m_batchSize) initialized to 1.0
        Tensor ones({1, m_batchSize}, false);
        std::fill(ones.data(), ones.data() + ones.size(), 1.0);

        Tensor Y_bias = matmul(transpose(bias_W), ones);

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
        return { m_weight };
    }

    void Dense::SetBatchSize(int batchSize)
    {
        if (m_batchSize == batchSize)
        {
            return;
        }
        m_batchSize = batchSize;
    }

    void Dense::InitWeight()
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        // He initialization standard deviation: sqrt(2.0 / fan_in)
        double stddev = std::sqrt(2.0 / static_cast<double>(m_inputSize - 1));
        std::normal_distribution<double> dist(0.0, stddev);

        for (int i = 0; i < m_inputSize; ++i)
        {
            for (int j = 0; j < m_outputSize; ++j)
            {
                m_weight(i, j) = dist(gen);
            }
        }

        // Initialize bias row (row 0) to zero
        for (int j = 0; j < m_outputSize; ++j)
        {
            m_weight(0, j) = 0.0;
        }
    }
}