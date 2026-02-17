#pragma once
#include <random>
#include <cmath>

#include <Eigen/Dense>

#include "Tensor.hpp"

namespace tardigrade::layer
{
	class Layer
	{
	public :
		virtual ~Layer() = default;

		virtual Tensor Forward(const Tensor& input) = 0;
		virtual Tensor Backward(const Tensor& input) = 0;

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
		
		Tensor Forward(const Tensor& input);
		Tensor Backward(const Tensor& input) { return Tensor({ 0, 0 }); }

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

		Tensor m_weight;
		Tensor m_gradient;

		Tensor m_inputMat;
		Tensor m_outputMat;
	};
}