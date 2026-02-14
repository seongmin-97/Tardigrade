#pragma once
#include <random>
#include <cmath>

#include <Eigen/Dense>

#include "Types.hpp"

namespace tardigrade::layer
{
	class Layer
	{
	public :
		virtual ~Layer() = default;

		virtual Vector Forward(const Vector& input) = 0;
		virtual Vector Backward(const Vector& input) = 0;

		virtual void Update(double learningRate) {}
		virtual void SetInputSize(int inputSize) {}
		virtual void SetOutputSize(int outputSize) {}
		virtual void SetBatchSize(int batchSize) {}
		virtual void SetUseBias(bool useBias) {}
	};

	class Dense : Layer
	{
	public :
		Dense(int inputSize, int outputSize, int batchSize = 1, bool useBias = true);
		
		Matrix Forward(const Matrix& input);
		Matrix Backward(const Matrix& input);

		void SetInputSize(int inputSize);
		void SetOutputSize(int outputSize);
		void SetBatchSize(int batchSize);
		void SetUseBias(bool useBias);

		void InitWeight();

	public:
		int m_inputSize;
		int m_outputSize;
		int m_batchSize;
		bool m_useBias;

		Matrix m_weight;
		Matrix m_gradient;

		Matrix m_inputMat;
		Matrix m_outputMat;
	};
}