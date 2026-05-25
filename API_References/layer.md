# Layer API Reference

## Overview
The `tardigrade::layer` namespace provides the neural network layer primitives. A layer is responsible for defining how data propagates forward (inference) and backward (gradient computation).

## Interfaces

### `Layer` (Abstract Base Class)
Defines the contract for all neural network layers.

#### Pure Virtual Methods
- `virtual Tensor Forward(const Tensor& input) = 0;`: Computes the forward pass.
- `virtual Tensor Backward(const Tensor& gradOutput) = 0;`: Computes the backward pass and calculates weight gradients.
- `virtual Tensor& GetWeights() = 0;`: Returns a reference to the layer's parameters.
- `virtual Tensor& GetGradients() = 0;`: Returns a reference to the parameter gradients.
- `virtual void SetActivation(std::unique_ptr<activation::Activation> activation) = 0;`: Sets the activation function for the layer.

---

## Implementations

### `Dense`
A fully connected (Dense) layer implementation. To optimize computation, the `Dense` layer uses the **Augmented Input Strategy**, where the bias term is folded directly into the weight matrix.

#### Mathematical Foundation
Instead of maintaining separate Weights ($W$) and Bias ($b$):
$$ Z = XW + b $$

The augmented strategy prepends a `1.0` to every input vector $X$:
$$ X' = [1, X_1, X_2, \dots, X_N] $$
$$ W' = \begin{bmatrix} b \\ W \end{bmatrix} $$
Resulting in a single matrix multiplication:
$$ Z = X' W' $$

#### Initialization
Weights are initialized using **He Initialization**, suitable for layers followed by ReLU:
$$ W_{ij} \sim \mathcal{N}\left(0, \frac{2}{\text{fan\_in}}\right) $$
*(Bias is initialized to 0.01)*

#### Forward Pass
1. Augments input `X` to `X'`.
2. Computes Matrix Multiplication $Z = X' W'$.
3. If an activation function is set, returns $\sigma(Z)$, otherwise returns $Z$.

#### Backward Pass
1. Computes the gradient w.r.t Activation: $\delta = \text{gradOutput} \odot \sigma'(Z)$.
2. Computes the Weight gradients: $\nabla W' = (X')^T \delta$.
3. Computes the Input gradients for the previous layer: $\nabla X = \delta (W)^T$ (where $W$ excludes the bias row).
4. Returns $\nabla X$.

## Usage Example
```cpp
#include "Layer.hpp"
#include "Activation.hpp"
using namespace tardigrade::layer;
using namespace tardigrade::activation;

// Create a Dense layer: Input=784, Output=256
Dense layer1(784, 256);

// Set ReLU activation
layer1.SetActivation(std::make_unique<ReLU>());

// Forward Pass
Tensor input({1, 784});
Tensor output = layer1.Forward(input);
```
