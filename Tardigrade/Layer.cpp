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

	m_weight = Matrix(m_inputSize, outputSize);
	m_gradient = Matrix(m_inputSize, outputSize);

	m_inputMat = Matrix(m_inputSize, m_batchSize);
	m_outputMat = Matrix(m_outputSize, m_batchSize);
}

Matrix Dense::Forward(const Matrix& input)
{
    if (m_useBias) 
    {
        m_inputMat.resize(input.rows() + 1, m_batchSize);
        m_inputMat.row(0).setConstant(1.0);
        m_inputMat.block(1, 0, input.rows(), m_batchSize) = input;
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

    m_weight.resize(m_inputSize, m_outputSize);
    m_gradient.resize(m_inputSize, m_outputSize);
    m_inputMat.resize(m_inputSize, m_batchSize);

    InitWeight();
}

void Dense::SetOutputSize(int outputSize)
{
    if (outputSize == m_outputSize)
        return;

    m_outputSize = outputSize;

    m_weight.resize(m_inputSize, m_outputSize);
    m_gradient.resize(m_inputSize, m_outputSize);
    m_outputMat.resize(m_outputSize, m_batchSize);

    InitWeight();
}

void Dense::SetBatchSize(int batchSize)
{
    if (batchSize == m_batchSize)
        return;

    m_batchSize = batchSize;

    m_inputMat.resize(m_inputSize, m_batchSize);
    m_outputMat.resize(m_outputSize, m_batchSize);
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

    m_weight.resize(m_inputSize, m_outputSize);
    m_gradient.resize(m_inputSize, m_outputSize);
    m_inputMat.resize(m_inputSize, m_batchSize);

    InitWeight();
}

void Dense::InitWeight()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    double stddev = std::sqrt(2.0 / static_cast<double>(m_inputSize));

    std::normal_distribution<double> dist(0.0, stddev);

    for (int i = 0; i < m_weight.rows(); ++i)
    {
        for (int j = 0; j < m_weight.cols(); ++j)
        {
            m_weight(i, j) = dist(gen);
        }
    }

    if (m_useBias)
    {
        m_weight.row(0).setZero();
    }
}