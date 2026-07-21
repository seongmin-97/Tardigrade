#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Layer.hpp"
#include "Loss.hpp"
#include "Metric.hpp"
#include "Optimizer.hpp"
#include "Tensor.hpp"

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

    Tensor Forward(const Tensor &input);

    void Backward(const Tensor &gradOutput);

    std::pair<double, double> TrainStep(const Tensor &input, const Tensor &target);

    Tensor Predict(const Tensor &input);

    /**
     * @brief Resets accumulated loss, metric, and processed sample counts.
     */
    void ResetMetrics();

    /**
     * @brief Prints step progress log and epoch summary to stdout.
     *
     * @param totalDataSize Total number of samples in dataset.
     * @param epoch Current epoch index (1-based).
     * @param totalEpochs Total number of epochs.
     * @param stepInterval Number of steps between progress logs (default: 10).
     */
    void PrintProgress(size_t totalDataSize, int epoch, int totalEpochs, size_t stepInterval = 10);

    /**
     * @brief Retrieves current average loss and accuracy.
     * @return std::pair<double, double> Pair of (average loss, accuracy percentage).
     */
    std::pair<double, double> GetCurrentMetrics() const;

    // Getters
    const std::vector<std::unique_ptr<layer::Layer>> &GetLayers() const;
    optimizer::Optimizer *GetOptimizer() const;
    loss::Loss *GetLossFunction() const;
    metric::Metric *GetMetric() const;
    int GetBatchSize() const;

private:
    std::vector<std::unique_ptr<layer::Layer>> m_layers;
    std::unique_ptr<optimizer::Optimizer> m_optimizer;
    std::unique_ptr<loss::Loss> m_lossFunction;
    std::unique_ptr<metric::Metric> m_metric;

    int m_batchSize{-1};

    // Accumulated metrics for tracking
    double m_totalLoss{0.0};
    double m_totalMetric{0.0};
    size_t m_processedSamples{0};
};
} // namespace tardigrade::model
