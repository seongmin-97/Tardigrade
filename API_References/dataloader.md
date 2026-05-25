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
- `int rows;`
- `int cols;`

---

## Classes

### `DataLoader`
Handles reading images from directories, normalizing pixel values, and generating minibatches for training.

#### Constructor
- `DataLoader(LoadStrategy strategy)`: Initializes the loader with the given strategy.

#### Methods
- `void LoadImageDataset(const std::string& rootDir, MatSize targetSize, int imreadFlag)`: 
  Iterates through subdirectories in `rootDir` (treating directory names as labels). Reads images using `cv::imread`, resizes them to `targetSize`, flattens them, and normalizes pixel values to the $[0, 1]$ range.
  $$ X_{\text{norm}} = \frac{X_{\text{raw}}}{255.0} $$

- `std::pair<Tensor, Tensor> GetBatch(int batchSize)`: 
  Returns a random minibatch of images and their corresponding one-hot encoded labels.
  - Returns a tuple `(ImagesTensor, LabelsTensor)`
  - Images Tensor Shape: `{batchSize, Rows * Cols}`
  - Labels Tensor Shape: `{batchSize, NumClasses}`

- `size_t GetDataSize() const`: Returns the total number of samples loaded.

## Usage Example
```cpp
#include "DataLoader.hpp"
using namespace tardigrade::data;

DataLoader loader(LoadStrategy::EAGER);
loader.LoadImageDataset("/path/to/MNIST/train", {28, 28}, cv::IMREAD_GRAYSCALE);

// Get a batch of 32 images
auto [images, labels] = loader.GetBatch(32);
```
