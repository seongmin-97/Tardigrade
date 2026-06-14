#pragma once
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>

#include "Tensor.hpp"
#include "Layer.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"
#include "Metric.hpp"

namespace tardigrade::model
{
    /**
     * @brief Orchestrator for the training and inference pipeline.
     * 
     * Integrates and manages Layer, Optimizer, and Loss objects, 
     * providing high-level workflows like Forward, Backward, and TrainStep.
     */
    class Model
    {
    public:
        Model() = default;

        void AddLayer(std::unique_ptr<layer::Layer> layer);

        void SetOptimizer(std::unique_ptr<optimizer::Optimizer> opt);

        void SetLossFunction(std::unique_ptr<loss::Loss> lossFunc);

        void SetMetric(std::unique_ptr<metric::Metric> metric);

        void InitWeights();

        Tensor Forward(const Tensor& input);

        void Backward(const Tensor& gradOutput);

        std::pair<double, double> TrainStep(const Tensor& input, const Tensor& target);

        Tensor Predict(const Tensor& input);

        // Getters
        const std::vector<std::unique_ptr<layer::Layer>>& GetLayers() const;
        optimizer::Optimizer* GetOptimizer() const;
        loss::Loss* GetLossFunction() const;
        metric::Metric* GetMetric() const;

    private:
        std::vector<std::unique_ptr<layer::Layer>> m_layers;
        std::unique_ptr<optimizer::Optimizer> m_optimizer;
        std::unique_ptr<loss::Loss> m_lossFunction;
        std::unique_ptr<metric::Metric> m_metric;
    };
}
