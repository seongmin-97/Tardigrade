#include <iostream>

#include "Model.hpp"

using namespace tardigrade;
using namespace tardigrade::model;

// ------------------------------------------------------------
// Model Setup: Layer, Optimizer and Loss Function configuration
// ------------------------------------------------------------
void Model::AddLayer(std::unique_ptr<layer::Layer> layer)
{
    if (!layer)
    {
        throw std::invalid_argument("Model::AddLayer - Layer pointer cannot be nullptr.");
    }

    int layerBatchSize = layer->GetBatchSize();
    if (m_batchSize == -1)
    {
        m_batchSize = layerBatchSize;
    }
    else if (m_batchSize != layerBatchSize)
    {
        throw std::invalid_argument("Model::AddLayer - Batch size mismatch. Expected: " + std::to_string(m_batchSize) +
                                    ", Got: " + std::to_string(layerBatchSize));
    }

    m_layers.push_back(std::move(layer));
}

void Model::SetOptimizer(std::unique_ptr<optimizer::Optimizer> opt) { m_optimizer = std::move(opt); }

void Model::SetLossFunction(std::unique_ptr<loss::Loss> lossFunc) { m_lossFunction = std::move(lossFunc); }

void Model::SetMetric(std::unique_ptr<metric::Metric> metric) { m_metric = std::move(metric); }

// ------------------------------------------------------------
// InitWeights: Initialize weights for all layers and register parameters
// ------------------------------------------------------------
void Model::InitWeights()
{
    if (!m_optimizer)
    {
        throw std::runtime_error("Model: Optimizer must be set before InitWeights()");
    }

    for (auto &layer : m_layers)
    {
        layer->InitWeight();
        m_optimizer->AddParameters(layer->GetParameters());
    }
}

// ------------------------------------------------------------
// Forward: Propagates input through all layers sequentially
// ------------------------------------------------------------
Tensor Model::Forward(const Tensor &input)
{
    Tensor current = input;

    for (auto &layer : m_layers)
    {
        current = layer->Forward(current);
    }

    return current;
}

// ------------------------------------------------------------
// Backward: Auto-differentiates starting from the Loss Function
// ------------------------------------------------------------
void Model::Backward(const Tensor &gradOutput)
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
std::pair<double, double> Model::TrainStep(const Tensor &input, const Tensor &target)
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

    // 7. Automatic accumulation of loss & metric
    size_t currentBatchSize = static_cast<size_t>((input.rank() == 1) ? 1 : input.dim(1));
    m_totalLoss += lossValue * currentBatchSize;
    m_totalMetric += metricValue * currentBatchSize;
    m_processedSamples += currentBatchSize;

    return {lossValue, metricValue};
}

Tensor Model::Predict(const Tensor &input) { return Forward(input); }

// ------------------------------------------------------------
// Metric Tracking & Progress Logging
// ------------------------------------------------------------
void Model::ResetMetrics()
{
    m_totalLoss = 0.0;
    m_totalMetric = 0.0;
    m_processedSamples = 0;
}

std::pair<double, double> Model::GetCurrentMetrics() const
{
    double avgLoss = (m_processedSamples > 0) ? (m_totalLoss / m_processedSamples) : 0.0;
    double acc = (m_processedSamples > 0) ? ((m_totalMetric / m_processedSamples) * 100.0) : 0.0;
    return {avgLoss, acc};
}

void Model::PrintProgress(size_t totalDataSize, int epoch, int totalEpochs, size_t stepInterval)
{
    if (m_batchSize <= 0)
    {
        return;
    }

    size_t batch = static_cast<size_t>(m_batchSize);

    if (stepInterval > 0 && (m_processedSamples / batch) % stepInterval == 0)
    {
        auto [avgLoss, acc] = GetCurrentMetrics();
        std::string metricName = m_metric ? m_metric->GetName() : "Metric";
        std::cout << "  [Epoch " << epoch << "/" << totalEpochs << " | Step " << m_processedSamples << "/"
                  << totalDataSize << "] Loss=" << avgLoss << " | " << metricName << "=" << acc << "%\n";
    }

    if (m_processedSamples >= totalDataSize)
    {
        auto [avgLoss, acc] = GetCurrentMetrics();
        std::string metricName = m_metric ? m_metric->GetName() : "Metric";

        std::cout << "========================================\n";
        std::cout << "[Epoch " << epoch << "/" << totalEpochs << "] "
                  << "Loss=" << avgLoss << " | " << metricName << "=" << acc << "%"
                  << " | Samples=" << m_processedSamples << "\n";
        std::cout << "========================================\n\n";
    }
}

const std::vector<std::unique_ptr<layer::Layer>> &Model::GetLayers() const { return m_layers; }

optimizer::Optimizer *Model::GetOptimizer() const { return m_optimizer.get(); }

loss::Loss *Model::GetLossFunction() const { return m_lossFunction.get(); }

metric::Metric *Model::GetMetric() const { return m_metric.get(); }

int Model::GetBatchSize() const { return m_batchSize; }
