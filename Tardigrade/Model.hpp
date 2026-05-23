#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>

#include "Tensor.hpp"
#include "Layer.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"

namespace tardigrade::model
{
    /**
     * @brief Model — 학습 파이프라인 오케스트레이터
     *
     * Layer, Optimizer, Loss를 통합 관리하며,
     * Forward/Backward/TrainStep 등 학습에 필요한 워크플로우를 제공한다.
     *
     * 사용 예시:
     *   Model model;
     *   model.AddLayer(std::make_unique<Dense>(784, 256, 1, ACTIVATION::ReLU));
     *   model.SetLossFunction(std::make_unique<SoftmaxCrossEntropy>(10, 1));
     *   model.SetOptimizer(std::make_unique<SGD>(0.01));
     *   model.InitWeights();
     *   double loss = model.TrainStep(input, label, predicted);
     */
    class Model
    {
    public:
        Model() = default;

        void AddLayer(std::unique_ptr<layer::Layer> layer);

        void SetOptimizer(std::unique_ptr<optimizer::Optimizer> opt);

        void SetLossFunction(std::unique_ptr<loss::Loss> lossFunc);

        void InitWeights();

        Tensor Forward(const Tensor& input);

        void Backward(const Tensor& gradOutput);

        double TrainStep(const Tensor& input, int label, int& predicted);

        Tensor Predict(const Tensor& input);

        // Getters
        const std::vector<std::unique_ptr<layer::Layer>>& GetLayers() const;
        optimizer::Optimizer* GetOptimizer() const;
        loss::Loss* GetLossFunction() const;

    private:
        std::vector<std::unique_ptr<layer::Layer>> m_layers;
        std::unique_ptr<optimizer::Optimizer> m_optimizer;
        std::unique_ptr<loss::Loss> m_lossFunction;
    };
}
