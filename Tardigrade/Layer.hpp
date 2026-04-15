#pragma once
#include <cmath>
#include <random>
#include <memory>

#include <Eigen/Dense>

#include "Tensor.hpp"
#include "Activation.hpp"

using namespace tardigrade::activation;

namespace tardigrade::layer
{
	class Layer
	{
	public :
		virtual ~Layer() = default;

		virtual Tensor Forward(const Tensor& input) = 0;
		virtual Tensor Backward(const Tensor& input) = 0;

		virtual std::vector<std::pair<Tensor*, Tensor*>> GetParameters() { return {}; }

		virtual void SetInputSize(int inputSize) {}
		virtual void SetOutputSize(int outputSize) {}
		virtual void SetBatchSize(int batchSize) {}
	};

	class Dense : public Layer
	{
	public :
		Dense(int inputSize, int outputSize, int batchSize = 1, ACTIVATION activation = ACTIVATION::NONE);
		
		Tensor Forward(const Tensor& input);
		Tensor Backward(const Tensor& input);

		void SetInputSize(int inputSize);
		void SetOutputSize(int outputSize);
		void SetBatchSize(int batchSize);
		void SetActivation(ACTIVATION activation);

		void InitWeight();
        std::vector<std::pair<Tensor*, Tensor*>> GetParameters() override;

	public:
		int m_inputSize;
		int m_outputSize;
		int m_batchSize;

		Tensor m_weight;
		Tensor m_gradient;

		Tensor m_inputMat;
		Tensor m_outputMat;

		ACTIVATION m_enumAct;
		std::unique_ptr<Activation> m_activation;
	};
}