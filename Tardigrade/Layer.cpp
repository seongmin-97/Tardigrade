#include "Layer.hpp"
#include "Activation.hpp"

using namespace tardigrade;
using namespace tardigrade::layer;
using namespace tardigrade::activation;

Dense::Dense(int inputSize, int outputSize, int batchSize, ACTIVATION activation)
{
    m_batchSize = batchSize;
	
	m_inputSize = inputSize + 1; // Always use bias (augmented input)
	m_outputSize = outputSize;

    m_weight = Tensor({ m_inputSize, m_outputSize });
    m_gradient = Tensor({ m_inputSize, m_outputSize });

    m_inputMat = Tensor({ m_inputSize, m_batchSize });
    m_outputMat = Tensor({ m_outputSize, m_batchSize });

    m_enumAct = activation;

    switch (m_enumAct)
    {
    case ACTIVATION::NONE :
        m_activation = std::make_unique<None>(m_outputSize, m_batchSize);
        break;
    case ACTIVATION::ReLU :
        m_activation = std::make_unique<ReLU>(m_outputSize, m_batchSize);
        break;
    }
}

Tensor Dense::Forward(const Tensor& input)
{
    int rows = input.dim(0);
    int cols = (input.rank() == 1) ? 1 : input.dim(1);

    if (rows != m_inputSize - 1)
        throw std::runtime_error("Input dimension mismatch in Dense::Forward. Expected " + std::to_string(m_inputSize - 1) + " but got " + std::to_string(rows));

    m_inputMat.row(0).setConstant(1.0);
    m_inputMat.asMatrix(rows, cols).bottomRows(rows) = input.asMatrix(rows, cols);

    m_outputMat = m_activation->Forward(m_weight.transpose() * m_inputMat);

	return m_outputMat;
}

Tensor Dense::Backward(const Tensor& input)
{
    Tensor dZ = m_activation->Backward(input);

    m_gradient = m_inputMat * dZ.transpose();

    Tensor dX_aug = m_weight * dZ;

    Tensor result({ m_inputSize - 1, m_batchSize });
    int rows = m_inputSize - 1;
    result.asMatrix(rows, m_batchSize) = dX_aug.asMatrix(m_inputSize, m_batchSize).bottomRows(rows);
    
    return result;
}

void Dense::SetInputSize(int inputSize)
{
    if (inputSize + 1 == m_inputSize)
        return;

    m_inputSize = inputSize + 1; // Always use bias

    m_weight.reshape({ m_inputSize, m_outputSize });
    m_gradient.reshape({ m_inputSize, m_outputSize });
    m_inputMat.reshape({ m_inputSize, m_batchSize });

    InitWeight();
}

void Dense::SetOutputSize(int outputSize)
{
    if (outputSize == m_outputSize)
        return;

    m_outputSize = outputSize;

    m_weight.reshape({ m_inputSize, m_outputSize });
    m_gradient.reshape({ m_inputSize, m_outputSize });
    m_outputMat.reshape({ m_outputSize, m_batchSize });

    InitWeight();
}

void Dense::SetBatchSize(int batchSize)
{
    if (batchSize == m_batchSize)
        return;

    m_batchSize = batchSize;

    m_inputMat.reshape({ m_inputSize, m_batchSize });
    m_outputMat.reshape({ m_outputSize, m_batchSize });
}

void Dense::SetActivation(ACTIVATION activation)
{
    if (m_enumAct == activation)
        return;

    m_enumAct = activation;

    switch (m_enumAct)
    {
    case ACTIVATION::NONE:
        m_activation = std::make_unique<None>(m_outputSize, m_batchSize);
        break;
    case ACTIVATION::ReLU:
        m_activation = std::make_unique<ReLU>(m_outputSize, m_batchSize);
        break;
    }
}

void Dense::InitWeight()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // Bias 항을 제외한 실제 입력 뉴런 수(m_inputSize - 1)를 기준으로 계산
    double stddev = std::sqrt(2.0 / static_cast<double>(m_inputSize - 1));

    std::normal_distribution<double> dist(0.0, stddev);
    
    int row = m_weight.dim(0);
    int col = m_weight.dim(1);

    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            m_weight(i, j) = dist(gen);

    // 상숫값(1.0)과 곱해지는 Bias 항은 0으로 초기화
    m_weight.row(0).setZero();
}