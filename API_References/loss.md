# Loss API Reference

## Overview
The `tardigrade::loss` namespace provides objective functions used to evaluate the error between the model's predictions and the true labels.

## Interfaces

### `Loss` (Abstract Base Class)
Defines the standard interface for a Loss function.

#### Pure Virtual Methods
- `virtual double Forward(const Tensor& prediction, const Tensor& target) = 0;`: Computes the scalar loss value given the prediction logits of shape `(inputSize, batchSize)` and the ground truth targets of shape `(1, batchSize)`.
- `virtual Tensor Backward() = 0;`: Computes the gradient of the loss with respect to the prediction inputs.

---

## Implementations

### `MSE` (Mean Squared Error)
Typically used for regression tasks.

#### Constructor
- `MSE(int inputSize, int batchSize)`

#### Mathematical Foundation
- **Forward Pass**:
  $$ L = \frac{1}{B \cdot C} \sum_{i=1}^B \sum_{j=1}^C (\hat{Y}_{j, i} - Y_{j, i})^2 $$
  Where $B$ is the batch size, $C$ is the feature size (`inputSize`), $\hat{Y}$ is the prediction tensor, and $Y$ is the target tensor.
- **Backward Pass**:
  $$ \frac{\partial L}{\partial \hat{Y}_{j, i}} = \frac{2}{B \cdot C} (\hat{Y}_{j, i} - Y_{j, i}) $$

---

### `SoftmaxCrossEntropy`
The standard loss function for multi-class classification. It mathematically combines the Softmax activation and Categorical Cross-Entropy into a single layer to avoid numerical instability and simplify gradient calculation.

#### Constructor
- `SoftmaxCrossEntropy(int inputSize, int batchSize)`

#### Target Format
The `target` tensor for `SoftmaxCrossEntropy` must be a 1D tensor of shape `(1, batchSize)` containing the integer class labels (represented as double) for each sample.

#### Mathematical Foundation
Let $Z$ be the raw logits, $P$ be the softmax probabilities per sample column, and $Y_i$ be the target integer class index for sample $i$.

- **Forward Pass**:
  $$ L = -\frac{1}{B} \sum_{i=1}^B \log(P_{Y_i, i} + \epsilon) $$
  *(Note: A small epsilon $10^{-12}$ is added to $P$ before the logarithm to prevent $\log(0)$).*

- **Backward Pass**:
  When computing the gradient of the Loss with respect to the logits $Z$, the mathematical derivation yields:
  $$ \frac{\partial L}{\partial Z_{j, i}} = \frac{1}{B} (P_{j, i} - Y'_{j, i}) $$
  where $Y'_{j, i} = 1.0$ if $j == Y_i$, else $0.0$.
  This gradient is directly returned by the `Backward()` method, effectively bypassing the `Softmax::Backward()` calculation in the computational graph.

## Usage Example
```cpp
#include <iostream>
#include "Loss.hpp"

using namespace tardigrade;
using namespace tardigrade::loss;

int main()
{
    constexpr int numClasses = 3;
    constexpr int batchSize = 2;

    SoftmaxCrossEntropy lossFunc(numClasses, batchSize);

    // Create mock prediction logits of shape (numClasses, batchSize)
    Tensor preds({numClasses, batchSize});
    preds(0, 0) = 2.0; preds(1, 0) = 0.5; preds(2, 0) = 0.1;
    preds(0, 1) = 0.2; preds(1, 1) = 1.8; preds(2, 1) = 0.5;

    // Create target labels for the batch
    Tensor targets({1, batchSize});
    targets[0] = 0.0; // Class 0
    targets[1] = 1.0; // Class 1

    double lossVal = lossFunc.Forward(preds, targets);
    Tensor gradients = lossFunc.Backward();

    std::cout << "Loss: " << lossVal << "\n";

    return 0;
}
```
