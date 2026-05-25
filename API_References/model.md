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

#### Forward and Backward Chaining
- `Tensor Forward(const Tensor& input)`: 
  Executes the forward pass through the network. It sequentially passes the output of layer $i$ as the input to layer $i+1$.
- `void Backward(const Tensor& gradOutput)`: 
  Executes the backpropagation pass. It traverses the layers in reverse order, passing the gradient of layer $i$ as the `gradOutput` for layer $i-1$.

#### High-Level Operations
- `double TrainStep(const Tensor& input, const Tensor& target)`: 
  Performs a complete training iteration on a single batch:
  1. Clears previous gradients via `m_optimizer->ZeroGrad()`.
  2. Executes `Forward(input)` to get predictions.
  3. Computes the loss using `m_lossFunction->Forward(predictions, target)`.
  4. Obtains the initial gradient from `m_lossFunction->Backward()`.
  5. Executes `Backward(grad)` through the layers.
  6. Updates weights via `m_optimizer->Step()`.
  7. Returns the scalar loss value.

- `Tensor Predict(const Tensor& input)`: 
  Alias for `Forward(input)`. Used during inference.

#### Accessors
- `const std::vector<std::unique_ptr<layer::Layer>>& GetLayers() const`
- `optimizer::Optimizer* GetOptimizer() const`
- `loss::Loss* GetLossFunction() const`

## Usage Example
```cpp
#include "Model.hpp"
#include "Layer.hpp"
#include "Activation.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

using namespace tardigrade;

model::Model myModel;

// 1. Add Layers
auto dense1 = std::make_unique<layer::Dense>(784, 256);
dense1->SetActivation(std::make_unique<activation::ReLU>());
myModel.AddLayer(std::move(dense1));

auto dense2 = std::make_unique<layer::Dense>(256, 10);
myModel.AddLayer(std::move(dense2)); // Softmax is handled by Loss

// 2. Set Loss and Optimizer
myModel.SetLossFunction(std::make_unique<loss::SoftmaxCrossEntropy>());
myModel.SetOptimizer(std::make_unique<optimizer::Adam>(0.001));

// 3. Train Step
double loss = myModel.TrainStep(batchImages, batchLabels);
```
