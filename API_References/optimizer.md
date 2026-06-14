# Optimizer API Reference

## Overview
The `tardigrade::optimizer` namespace defines algorithms that update the parameters (weights) of neural network layers based on the gradients computed during backpropagation.

## Interfaces

### `Optimizer` (Abstract Base Class)
Defines the standard interface for an optimization algorithm.

#### Abstract Methods
- `virtual void Step() = 0;`: Executes a single parameter update step.

#### Member Management
- `void AddParameters(const std::vector<std::pair<Tensor*, Tensor*>>& params)`: Registers a vector of parameter-gradient pairs with the optimizer. (Typically retrieved from layers via `layer.GetParameters()`).
- `virtual void ZeroGrad()`: Resets the gradients of all registered parameters to zero. The implementation leverages Eigen's `.setZero()` vectorized method for optimal memory zero-out performance.

---

## Implementations

### `SGD` (Stochastic Gradient Descent)
A basic optimizer that updates weights by moving them in the opposite direction of the gradient.

#### Mathematical Foundation
$$ W_{t+1} = W_t - \eta \cdot \nabla W_t $$
Where $\eta$ is the `learningRate`.

---

### `Adam` (Adaptive Moment Estimation)
An advanced optimizer that computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

#### Mathematical Foundation
Maintains a moving average of both the gradients (first moment $m$) and the squared gradients (second moment $v$).

1. **Update biased first moment estimate**:
   $$ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla W_t $$
2. **Update biased second raw moment estimate**:
   $$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla W_t)^2 $$
3. **Compute bias-corrected first moment estimate**:
   $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$
4. **Compute bias-corrected second raw moment estimate**:
   $$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
5. **Update parameters**:
   $$ W_{t+1} = W_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

*Where:*
- $\eta$: Learning rate
- $\beta_1$: Decay rate for first moment (default: 0.9)
- $\beta_2$: Decay rate for second moment (default: 0.999)
- $\epsilon$: Small constant for numerical stability (default: $10^{-8}$)
- $t$: Time step (iteration number)

## Usage Example
```cpp
#include <iostream>
#include "Optimizer.hpp"
#include "Layer.hpp"

using namespace tardigrade;
using namespace tardigrade::optimizer;
using namespace tardigrade::layer;

int main()
{
    constexpr double lr = 0.001;
    constexpr int batchSize = 16;
    
    Adam optimizer(lr); 

    Dense layer1(784, 256, batchSize, ACTIVATION::ReLU);
    layer1.InitWeight();
    
    // Register the layer parameters to the optimizer
    optimizer.AddParameters(layer1.GetParameters());

    // ... inside training loop ...
    optimizer.ZeroGrad();
    // (Compute Forward and Backward passes)
    optimizer.Step();

    return 0;
}
```
