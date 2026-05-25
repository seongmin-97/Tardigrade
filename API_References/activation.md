# Activation API Reference

## Overview
The `tardigrade::activation` namespace contains non-linear activation functions applied to the outputs of neural network layers. 

## Interfaces

### `Activation` (Abstract Base Class)
Defines the standard interface for an activation function.

#### Pure Virtual Methods
- `virtual Tensor Forward(const Tensor& input) = 0;`: Applies the activation function to the input.
- `virtual Tensor Backward(const Tensor& gradOutput, const Tensor& input) = 0;`: Computes the gradient of the loss with respect to the input of the activation.

---

## Implementations

### `None`
A pass-through activation function.
- **Forward**: $f(x) = x$
- **Backward**: $f'(x) = 1 \implies \text{gradInput} = \text{gradOutput}$

---

### `ReLU` (Rectified Linear Unit)
The standard activation function for deep neural networks.

#### Mathematical Foundation
- **Forward Pass**: 
  $$ f(x) = \max(0, x) $$
- **Backward Pass**: 
  $$ f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases} $$
  The gradient is computed via Hadamard product: $\text{gradOutput} \odot f'(x)$.

---

### `Softmax`
Used primarily in the output layer for multi-class classification to yield a probability distribution.

#### Mathematical Foundation
To prevent numerical overflow, the implementation subtracts the maximum value in the input vector ($C = \max(x_i)$):
$$ \sigma(x)_i = \frac{e^{x_i - C}}{\sum_j e^{x_j - C}} $$

> **Note on Softmax Backward**: 
> In Tardigrade, when Softmax is combined with Cross-Entropy Loss, computing their independent gradients is computationally inefficient and numerically unstable. Therefore, `Softmax::Backward` acts as a pass-through in this implementation, and the true combined gradient $P - Y$ is computed inside `SoftmaxCrossEntropy::Backward`.

## Usage Example
```cpp
#include "Activation.hpp"
using namespace tardigrade::activation;

ReLU relu;
Tensor input({1, 5}); 
// input = [-2, -1, 0, 1, 2]
Tensor activated = relu.Forward(input); 
// activated = [0, 0, 0, 1, 2]
```
