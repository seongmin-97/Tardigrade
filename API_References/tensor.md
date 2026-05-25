# Tensor API Reference

## Overview
`tardigrade::Tensor` is the foundational data structure of the Tardigrade framework. It wraps a flat 1D `std::vector<double>` and manages multi-dimensional views. By leveraging `Eigen::Map`, `Tensor` achieves zero-copy, highly optimized matrix and vector operations without sacrificing standard C++ vector safety.

## Class Details

### `tardigrade::Tensor`

#### Memory Layout
The tensor stores all elements contiguously.
$$ \text{Size} = \prod_{i=0}^{N-1} \text{shape}[i] $$

#### Constructors
- `Tensor(const std::vector<int>& shape)`: Allocates zero-initialized memory for the given shape.
- `Tensor() = default;`: Creates an empty tensor.

#### Memory Mapping (Zero-Copy)
- `MatrixMap asMatrix(int rows, int cols)` / `ConstMatrixMap asMatrix(int rows, int cols) const`: Returns an `Eigen::Map<Eigen::MatrixXd>` representation.
- `VectorMap asVector()` / `ConstVectorMap asVector() const`: Returns an `Eigen::Map<Eigen::VectorXd>`.

#### Shape Operations
- `void reshape(const std::vector<int>& newShape)`: Changes the tensor shape in-place without reallocation.
- `Tensor transpose() const`: Returns a new tensor matching $(A^T)_{ij} = A_{ji}$ for 2D tensors.

#### Mathematical Operations
Overloaded operators utilize Eigen's vectorized instructions for optimal performance.
- `Tensor operator*(const Tensor& other) const`: Matrix multiplication.
  $$ C_{ij} = \sum_k A_{ik} B_{kj} $$
- `Tensor operator+(const Tensor& other) const`: Element-wise addition.
- `Tensor operator-(const Tensor& other) const`: Element-wise subtraction.
- `Tensor operator*(double scalar) const`: Scalar multiplication.
- `Tensor operator/(double scalar) const`: Scalar division.
- `Tensor cwiseMul(const Tensor& other) const`: Hadamard (Element-wise) multiplication. $C_{ij} = A_{ij} \times B_{ij}$
- `Tensor cwiseDiv(const Tensor& other) const`: Element-wise division.

#### Activation Helpers
- `Tensor clampedMin(double threshold) const`: Computes $f(x) = \max(x, \text{threshold})$. (Used in ReLU).
- `Tensor step() const`: Computes the step function (derivative of ReLU). $f(x) = 1$ if $x > 0$ else $0$.

## Usage Example
```cpp
#include "Tensor.hpp"
using namespace tardigrade;

Tensor A({2, 3});
Tensor B({3, 2});

// A and B can be accessed and manipulated...
Tensor C = A * B; // Matrix Multiplication resulting in {2, 2} shape
Tensor D = C + C; // Element-wise addition
```
