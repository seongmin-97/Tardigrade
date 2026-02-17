#include "Tensor.hpp"
#include "Layer.hpp"

using namespace tardigrade;
using namespace tardigrade::layer;

Dense::Dense(int inputSize, int outputSize, int batchSize, bool useBias)
{
	m_useBias = useBias;
    m_batchSize = batchSize;
	
	m_inputSize = inputSize;
	m_outputSize = outputSize;

	if (m_useBias)
		++m_inputSize;

    m_weight = Tensor({ m_inputSize, outputSize });
    m_gradient = Tensor({ m_inputSize, outputSize });

    m_inputMat = Tensor({ m_inputSize, m_batchSize });
    m_outputMat = Tensor({ m_outputSize, m_batchSize });
}

Tensor Dense::Forward(const Tensor& input)
{
    if (m_useBias)
    {
        int rows = input.dim(0);
        int cols = (input.rank() == 1) ? 1 : input.dim(1);

        m_inputMat.row(0).setConstant(1.0);
        m_inputMat.asMatrix(rows, cols).bottomRows(rows) = input.asMatrix(rows, cols);
    }
    else
    {
        m_inputMat = input;
    }

    m_outputMat = m_weight.transpose() * m_inputMat;
	
	return m_outputMat;
}

void Dense::SetInputSize(int inputSize)
{
    if (inputSize == m_inputSize)
        return;

    m_inputSize = m_useBias ? (inputSize + 1) : inputSize;

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

void Dense::SetUseBias(bool useBias)
{
    if (m_useBias == useBias) 
        return;

    if (useBias)
        m_inputSize++;
    else
        m_inputSize--;

    m_useBias = useBias;

    m_weight.reshape({ m_inputSize, m_outputSize });
    m_gradient.reshape({ m_inputSize, m_outputSize });
    m_inputMat.reshape({ m_inputSize, m_batchSize });

    InitWeight();
}

void Dense::InitWeight()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    double stddev = std::sqrt(2.0 / static_cast<double>(m_inputSize));

    std::normal_distribution<double> dist(0.0, stddev);
    
    int row = m_weight.dim(0);
    int col = m_weight.dim(1);

    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            m_weight(i, j) = dist(gen);

    if (m_useBias)
    {
        m_weight.row(0).setZero();
    }
}