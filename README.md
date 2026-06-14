# Tardigrade Framework

![Tardigrade](https://img.shields.io/badge/Status-Educational-blue) ![C++17](https://img.shields.io/badge/C++-17-blue) ![macOS](https://img.shields.io/badge/OS-macOS-lightgrey)

Tardigrade is a lightweight, educational Deep Learning and 3D Vision framework written from scratch in modern C++17. The project aims to provide clear, understandable implementations of core Neural Network architectures without abstracting away the fundamental mathematics.

## Features

- **Zero-Dependency Core Math**: Built entirely on top of `Eigen 3` for robust and high-performance linear algebra and `OpenCV` for dataset loading and image processing.
- **Clear Mathematical Foundations**: Equations for Forward Propagation, Backpropagation, and Optimizers are heavily documented directly in the codebase using LaTeX/ASCII syntax.
- **Modular Architecture**: 
  - `Tensor`: Zero-copy memory mapped arrays (`Eigen::Map`) with Row-Major matrix bindings.
  - `Model`: Sequential network orchestrator supporting dynamic batch sizing.
  - `Layer`: Abstract layer designs (e.g., `Dense` with He Initialization) with automatic remainder batch adjustment.
  - `Activation`: Nonlinearities (`ReLU`, `Softmax`) running column-wise over mini-batches.
  - `Loss`: Objective functions (`MSE`, `SoftmaxCrossEntropy`) with mean batch loss scaling.
  - `Optimizer`: Parameter update algorithms (`SGD`, `Adam`) with optimized zero-gradients initialization via Eigen.
  - `Metric`: Evaluation indicators (e.g., `Accuracy`) to compute performance metrics on the fly.
  - `DataLoader`: High-performance batched dataset loading.
- **Doxygen Commented**: All headers are fully documented in English Doxygen format.
- **API Reference**: Extensive markdown documentation available in the `API_References/` directory.

---

## Environment & Requirements

- **OS**: macOS (Linux compatible)
- **Compiler**: Clang or GCC supporting C++17
- **Build System**: CMake (>= 3.10)
- **Dependencies**: 
  - `Eigen3` (Usually installed via `brew install eigen`)
  - `OpenCV` (Usually installed via `brew install opencv`)

---

## Documentation

Full architectural concepts and individual API component references can be found in the `API_References/` directory:
- [Concept & Architecture](API_References/concept.md)
- [Tensor API](API_References/tensor.md)
- [Layer API](API_References/layer.md)
- [Activation API](API_References/activation.md)
- [Loss API](API_References/loss.md)
- [Optimizer API](API_References/optimizer.md)
- [Metric API](API_References/metric.md)
- [DataLoader API](API_References/dataloader.md)
- [Model API](API_References/model.md)

---

## Build & Execute

We provide a convenient build script `./build.sh` for easy compilation.

### Step 1: Clone the repository
```bash
git clone https://github.com/seongmin-97/Tardigrade.git
cd Tardigrade
```

### Step 2: Ensure Dataset is ready
The `main.cpp` executes a MNIST Multi-Layer Perceptron (MLP) training session. Make sure your dataset path in `main.cpp` points to your local MNIST dataset directory.

```cpp
// main.cpp
const std::string datasetRoot = "/Users/home/Main/01_Dev/99_Dataset/MNIST/train";
```

### Step 3: Build
Execute the provided build script:
```bash
./build.sh
```

### Step 4: Run
The build script outputs the binary in the `build/` directory:
```bash
./build/Tardigrade
```

You should see the network training on MNIST and printing loss iteratively.
