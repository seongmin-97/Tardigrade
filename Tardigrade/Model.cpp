#include "Model.hpp"

using namespace tardigrade;
using namespace tardigrade::model;

// ------------------------------------------------------------
// Model Setup: Layer, Optimizer and Loss Function configuration
// ------------------------------------------------------------
void Model::AddLayer(std::unique_ptr<layer::Layer> layer)
{
    m_layers.push_back(std::move(layer));
}

void Model::SetOptimizer(std::unique_ptr<optimizer::Optimizer> opt)
{
    m_optimizer = std::move(opt);
}

void Model::SetLossFunction(std::unique_ptr<loss::Loss> lossFunc)
{
    m_lossFunction = std::move(lossFunc);
}

void Model::SetMetric(std::unique_ptr<metric::Metric> metric)
{
    m_metric = std::move(metric);
}

// ------------------------------------------------------------
// InitWeights: Initialize weights for all layers and register parameters
// ------------------------------------------------------------
void Model::InitWeights()
{
    if (!m_optimizer)
    {
        throw std::runtime_error("Model: Optimizer must be set before InitWeights()");
    }

    for (auto& layer : m_layers)
    {
        layer->InitWeight();
        m_optimizer->AddParameters(layer->GetParameters());
    }
}

// ------------------------------------------------------------
// Forward: Propagates input through all layers sequentially
// ------------------------------------------------------------
Tensor Model::Forward(const Tensor& input)
{
    Tensor current = input;

    for (auto& layer : m_layers)
    {
        current = layer->Forward(current);
    }

    return current;
}

// ------------------------------------------------------------
// Backward: Auto-differentiates starting from the Loss Function
// ------------------------------------------------------------
void Model::Backward(const Tensor& gradOutput)
{
    // Autograd manages backward graph natively, so we just run loss backward.
    if (m_lossFunction)
    {
        m_lossFunction->Backward();
    }
}

// ------------------------------------------------------------
// TrainStep: Performs a single training step
// ------------------------------------------------------------
std::pair<double, double> Model::TrainStep(const Tensor& input, const Tensor& target)
{
    if (!m_optimizer || !m_lossFunction)
    {
        throw std::runtime_error("Model: Optimizer and LossFunction must be set before training");
    }

    // 1. ZeroGrad
    m_optimizer->ZeroGrad();

    // 2. Forward
    Tensor logits = Forward(input);

    // 3. Loss computation
    double lossValue = m_lossFunction->Forward(logits, target);

    // 4. Metric computation
    double metricValue = 0.0;
    if (m_metric)
    {
        metricValue = m_metric->Evaluate(m_lossFunction->GetProbs(), target);
    }

    // 5. Backward pass via Autograd
    m_lossFunction->Backward();

    // 6. Parameter update
    m_optimizer->Step();

    return { lossValue, metricValue };
}

Tensor Model::Predict(const Tensor& input)
{
    return Forward(input);
}

const std::vector<std::unique_ptr<layer::Layer>>& Model::GetLayers() const
{
    return m_layers;
}

optimizer::Optimizer* Model::GetOptimizer() const
{
    return m_optimizer.get();
}

loss::Loss* Model::GetLossFunction() const
{
    return m_lossFunction.get();
}

metric::Metric* Model::GetMetric() const
{
    return m_metric.get();
}
