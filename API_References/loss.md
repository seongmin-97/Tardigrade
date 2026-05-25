# Loss API Reference

## Overview
The `tardigrade::loss` namespace provides objective functions used to evaluate the error between the model's predictions and the true labels.

## Interfaces

### `Loss` (Abstract Base Class)
Defines the standard interface for a Loss function.

#### Pure Virtual Methods
- `virtual double Forward(const Tensor& predictions, const Tensor& targets) = 0;`: Computes the scalar loss value given the predictions and the ground truth targets.
- `virtual Tensor Backward() = 0;`: Computes the gradient of the loss with respect to the predictions.

---

## Implementations

### `MSE` (Mean Squared Error)
Typically used for regression tasks.

#### Mathematical Foundation
- **Forward Pass**:
  $$ L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$
- **Backward Pass**:
  $$ \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i) $$

---

### `SoftmaxCrossEntropy`
The standard loss function for multi-class classification. It mathematically combines the Softmax activation and Categorical Cross-Entropy into a single layer to avoid numerical instability and simplify gradient calculation.

#### Mathematical Foundation
Let $Z$ be the raw logits, $P$ be the softmax probabilities ($P = \text{softmax}(Z)$), and $Y$ be the one-hot encoded ground truth targets.

- **Forward Pass**:
  $$ L = -\frac{1}{N} \sum_{i=1}^N \sum_{j} Y_{ij} \log(P_{ij}) $$
  *(Note: A small epsilon $10^{-7}$ is added to $P$ before the logarithm to prevent $\log(0)$).*

- **Backward Pass**:
  When computing the gradient of the Loss with respect to the logits $Z$, the mathematical derivation yields a beautifully simple formula:
  $$ \frac{\partial L}{\partial Z} = P - Y $$
  This gradient is directly returned by the `Backward()` method, effectively bypassing the `Softmax::Backward()` calculation in the computational graph.

## Usage Example
```cpp
#include "Loss.hpp"
using namespace tardigrade::loss;

SoftmaxCrossEntropy lossFunc;
Tensor preds({1, 3});   // e.g., Output from Dense layer (logits)
Tensor targets({1, 3}); // One-hot labels

double lossVal = lossFunc.Forward(preds, targets);
Tensor gradients = lossFunc.Backward();
```
