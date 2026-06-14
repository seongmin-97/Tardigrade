# DataLoader API Reference

## Overview
The `tardigrade::data` namespace provides utilities for loading and batching datasets, specifically designed around standard formats like the MNIST dataset. It uses `OpenCV` (`cv::Mat`) to read image files and converts them into normalized `tardigrade::Tensor` structures.

## Enums and Structures

### `LoadStrategy`
Defines how data is loaded into memory.
- `EAGER`: Loads the entire dataset into RAM during initialization.
- `LAZY`: (Not fully implemented) Intended for dynamic loading from disk during training to save memory.

### `MatSize`
A simple struct holding target dimensions for resizing images.
- `int row;`
- `int col;`

---

## Classes

### `DataLoader`
Handles reading images from directories, normalizing pixel values, and generating minibatches for training.

#### Constructor
- `DataLoader(LoadStrategy strategy = LoadStrategy::EAGER)`: Initializes the loader with the given strategy.

#### Methods
- `void LoadImageDataset(const std::string& rootDir, MatSize target = {0, 0}, int flag = cv::IMREAD_GRAYSCALE)`: 
  Iterates through subdirectories in `rootDir` (treating directory names as labels). Reads images using `cv::imread`, resizes them to `target`, flattens them, and normalizes pixel values to the $[0, 1]$ range.
  $$ X_{\text{norm}} = \frac{X_{\text{raw}}}{255.0} $$

- `size_t GetDataSize() const`: Returns the total number of samples loaded.

- `Tensor GetData(size_t index) const`: Retrieves a single normalized image tensor of shape `(totalPixels, 1)`.

- `int GetLabel(size_t index) const`: Retrieves a single integer label.

- `Tensor GetBatch(size_t startIdx, size_t batchSize) const`: 
  Retrieves a batch of image tensors stacked column-wise.
  - Returns a Tensor of shape `(featureSize, actualBatchSize)`.

- `std::vector<int> GetLabelBatch(size_t startIdx, size_t batchSize) const`: 
  Retrieves a batch of ground-truth integer labels as a vector of size `actualBatchSize`.

- `void Shuffle(std::mt19937 &rng)`: Shuffles the indices of the dataset in-place using the provided random engine.

## Usage Example
```cpp
#include <iostream>
#include <random>
#include "DataLoader.hpp"

using namespace tardigrade;
using namespace tardigrade::data;

int main()
{
    std::random_device rd;
    std::mt19937 rng(rd());

    DataLoader loader(LoadStrategy::EAGER);
    loader.LoadImageDataset("/path/to/MNIST/train", {28, 28}, cv::IMREAD_GRAYSCALE);

    // Shuffle each epoch
    loader.Shuffle(rng);

    constexpr size_t batchSize = 16;
    for (size_t i = 0; i < loader.GetDataSize(); i += batchSize)
    {
        size_t currentBatchSize = std::min(batchSize, loader.GetDataSize() - i);
        
        Tensor batchImages = loader.GetBatch(i, currentBatchSize);
        std::vector<int> batchLabels = loader.GetLabelBatch(i, currentBatchSize);

        // ... process batch ...
    }

    return 0;
}
```
