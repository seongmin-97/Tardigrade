# Layer API Reference

## Overview
The `tardigrade::layer` namespace provides the neural network layer primitives. A layer is responsible for defining how data propagates forward (inference) and backward (gradient computation).

## Interfaces

### `Layer` (Abstract Base Class)
Defines the contract for all neural network layers.

#### Pure Virtual Methods
- `virtual Tensor Forward(const Tensor& input) = 0;`: Computes the forward pass.
- `virtual Tensor Backward(const Tensor& gradOutput) = 0;`: Computes the backward pass and calculates parameter gradients.

#### Virtual Methods (with default no-op implementations)
- `virtual std::vector<std::pair<Tensor*, Tensor*>> GetParameters()`: Returns a vector of pairs, each containing pointers to (weight, gradient) tensors for optimization.
- `virtual void SetInputSize(int inputSize)`: Set the input feature size of the layer.
- `virtual void SetOutputSize(int outputSize)`: Set the output feature size of the layer.
- `virtual void SetBatchSize(int batchSize)`: Sets the batch size for calculations, triggering internal shape reallocations or reshaping.

---

## Implementations

### `Dense`
A fully connected (Dense) layer implementation. To optimize computation, the `Dense` layer uses the **Augmented Input Strategy**, where the bias term is folded directly into the weight matrix.

#### Constructor
- `Dense(int inputSize, int outputSize, int batchSize = 1, ACTIVATION activation = ACTIVATION::NONE)`
  - `inputSize`: Feature dimension of incoming data (excluding bias).
  - `outputSize`: Feature dimension of outgoing data.
  - `batchSize`: Expected mini-batch size (defaults to 1).
  - `activation`: An enum `ACTIVATION` value specifying the layer's activation function.

#### Initialization
Weights are initialized using **He Initialization**, suitable for layers followed by ReLU:
$$ W_{ij} \sim \mathcal{N}\left(0, \frac{2}{\text{fan\_in}}\right) $$
*(Bias is initialized to 0.0)*

#### Forward Pass
1. Compares the input batch size (number of columns) with `m_batchSize`. If there is a mismatch (e.g. remainder batch at the end of an epoch), it dynamically invokes `SetBatchSize(cols)` to reallocate buffers safely.
2. Augments input `X` to `X'`.
3. Computes Matrix Multiplication $Z = X' W'$.
4. If an activation function is set, returns $\sigma(Z)$, otherwise returns $Z$.

#### Backward Pass
1. Computes the gradient w.r.t Activation: $\delta = \text{gradOutput} \odot \sigma'(Z)$.
2. Computes the Weight gradients: $\nabla W' = (X')^T \delta$.
3. Computes the Input gradients for the previous layer: $\nabla X = \delta (W)^T$ (where $W$ excludes the bias row).
4. Returns $\nabla X$.

## Usage Example
```cpp
#include <iostream>
#include <memory>
#include "Layer.hpp"
#include "Activation.hpp"

using namespace tardigrade;
using namespace tardigrade::layer;
using namespace tardigrade::activation;

int main()
{
    constexpr int inputSize = 784;
    constexpr int outputSize = 256;
    constexpr int batchSize = 16;

    // Create a Dense layer with ReLU activation and batch size 16
    Dense layer1(inputSize, outputSize, batchSize, ACTIVATION::ReLU);
    layer1.InitWeight();

    // Forward Pass with matching batch size
    Tensor input({inputSize, batchSize});
    Tensor output = layer1.Forward(input);

    return 0;
}
```
