# Model API Reference

## Overview
The `tardigrade::model` namespace provides the `Model` class, which serves as the central orchestrator for the Tardigrade framework. It manages the collection of layers, the objective loss function, and the optimizer, handling the entire lifecycle of training and inference.

## Classes

### `Model`
A Sequential-style neural network model that chains layers together.

#### Member Management
- `void AddLayer(std::unique_ptr<layer::Layer> layer)`: Appends a layer to the network sequence.
- `void SetOptimizer(std::unique_ptr<optimizer::Optimizer> optimizer)`: Sets the optimization algorithm (e.g., SGD, Adam). It automatically registers all currently added layers with the optimizer.
- `void SetLossFunction(std::unique_ptr<loss::Loss> loss)`: Sets the objective loss function (e.g., SoftmaxCrossEntropy, MSE).
- `void SetMetric(std::unique_ptr<metric::Metric> metric)`: Registers an evaluation metric (e.g., `Accuracy`) to assess training performance.

#### Forward and Backward Chaining
- `Tensor Forward(const Tensor& input)`: 
  Executes the forward pass through the network. It sequentially passes the output of layer $i$ as the input to layer $i+1$.
- `void Backward(const Tensor& gradOutput)`: 
  Executes the backpropagation pass. It traverses the layers in reverse order, passing the gradient of layer $i$ as the `gradOutput` for layer $i-1$.

#### High-Level Operations
- `std::pair<double, double> TrainStep(const Tensor& input, const Tensor& target)`: 
  Performs a complete training iteration on a single batch:
  1. Clears previous gradients via `m_optimizer->ZeroGrad()`.
  2. Executes `Forward(input)` to get prediction logits.
  3. Computes the loss using `m_lossFunction->Forward(predictions, target)`.
  4. Evaluates the evaluation metric if registered: `m_metric->Evaluate(predictions, target)`.
  5. Obtains the initial gradient from `m_lossFunction->Backward()`.
  6. Executes `Backward(grad)` through the layers.
  7. Updates weights via `m_optimizer->Step()`.
  8. Returns a pair of `{lossValue, metricValue}`.

- `Tensor Predict(const Tensor& input)`: 
  Alias for `Forward(input)`. Used during inference.

#### Accessors
- `const std::vector<std::unique_ptr<layer::Layer>>& GetLayers() const`
- `optimizer::Optimizer* GetOptimizer() const`
- `loss::Loss* GetLossFunction() const`
- `metric::Metric* GetMetric() const`

## Usage Example
```cpp
#include <iostream>
#include <memory>
#include "Model.hpp"
#include "Layer.hpp"
#include "Activation.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Metric.hpp"

using namespace tardigrade;
using namespace tardigrade::model;
using namespace tardigrade::layer;
using namespace tardigrade::activation;
using namespace tardigrade::loss;
using namespace tardigrade::optimizer;
using namespace tardigrade::metric;

int main()
{
    constexpr int batchSize = 16;
    Model myModel;

    // 1. Add Layers
    myModel.AddLayer(std::make_unique<Dense>(784, 256, batchSize, ACTIVATION::ReLU));
    myModel.AddLayer(std::make_unique<Dense>(256, 10, batchSize, ACTIVATION::NONE)); // Softmax is handled by Loss
    
    // 2. Set Loss, Optimizer and Metric
    myModel.SetLossFunction(std::make_unique<SoftmaxCrossEntropy>(10, batchSize));
    myModel.SetOptimizer(std::make_unique<Adam>(0.001));
    myModel.SetMetric(std::make_unique<Accuracy>());
    myModel.InitWeights();

    // 3. Mock Data
    Tensor batchImages({784, batchSize});
    Tensor batchLabels({1, batchSize});
    batchLabels[0] = 0.0;

    // 4. Train Step
    auto [loss, metricVal] = myModel.TrainStep(batchImages, batchLabels);
    std::cout << "Loss: " << loss << " | Accuracy: " << metricVal * 100.0 << "%\n";

    return 0;
}
```
