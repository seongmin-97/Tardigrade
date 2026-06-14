# Activation API Reference

## Overview
The `tardigrade::activation` namespace contains non-linear activation functions applied to the outputs of neural network layers. 

## Interfaces

### `Activation` (Abstract Base Class)
Defines the standard interface for an activation function.

#### Pure Virtual Methods
- `virtual Tensor Forward(const Tensor& input) = 0;`: Applies the activation function to the input.
- `virtual Tensor Backward(const Tensor& gradOutput) = 0;`: Computes the gradient of the loss with respect to the input of the activation using the cached forward states.

#### Virtual Methods
- `virtual void SetBatchSize(int batchSize);`: Dynamically updates the batch size, resizing intermediate cached tensors (`m_inputVector`, `m_outputVector`, `m_gradient`).

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
Used primarily in the output layer for multi-class classification to yield a probability distribution. The implementation performs numerically stable softmax per sample column in the batch.

#### Mathematical Foundation
To prevent numerical overflow, the implementation subtracts the maximum value in each sample column ($C_i = \max_j(Z_{j, i})$):

- **Forward Pass**:
  $$ \sigma(Z)_{k, i} = \frac{e^{Z_{k, i} - C_i}}{\sum_j e^{Z_{j, i} - C_i}} $$
  Where $k$ is the class/feature index and $i$ is the batch index.

- **Backward Pass**:
  Computes the backward pass using the Softmax Jacobian matrix for each sample column:
  $$ \frac{\partial L}{\partial Z_{k, i}} = \sigma_{k, i} \left( \frac{\partial L}{\partial \sigma_{k, i}} - \sum_j \frac{\partial L}{\partial \sigma_{j, i}} \sigma_{j, i} \right) $$

> **Note on Softmax Backward with Loss**: 
> While `Softmax::Backward` implements the general Jacobian multiplication, in practice when Softmax is combined with Cross-Entropy Loss, computing independent gradients is inefficient and numerically unstable. Therefore, Tardigrade provides a dedicated `SoftmaxCrossEntropy` loss layer that computes the combined gradient $P - Y$ directly, bypassing this individual activation backward step during training.

## Usage Example
```cpp
#include <iostream>
#include "Activation.hpp"

using namespace tardigrade;
using namespace tardigrade::activation;

int main()
{
    constexpr int featureSize = 5;
    constexpr int batchSize = 1;
    
    ReLU relu(featureSize, batchSize);
    Tensor input({featureSize, batchSize}); 
    
    input[0] = -2.0;
    input[1] = -1.0;
    input[2] = 0.0;
    input[3] = 1.0;
    input[4] = 2.0;

    Tensor activated = relu.Forward(input); 
    // activated will be [0.0, 0.0, 0.0, 1.0, 2.0]
    
    return 0;
}
```
