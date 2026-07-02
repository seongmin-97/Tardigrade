#pragma once
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Tensor.hpp"

namespace tardigrade::data {
/**
 * @brief Data loading strategy.
 * 
 * EAGER: Loads the entire dataset into RAM (faster training, suitable for small/medium datasets).
 * LAZY: Loads images from disk on-the-fly (memory-efficient, suitable for large datasets).
 */
enum class LoadStrategy { EAGER, LAZY };

/**
 * @brief Target size for image resizing (0 indicates preserving original size).
 */
struct MatSize {
  int row;
  int col;
};

namespace fs = std::filesystem;

/**
 * @brief DataLoader - Handles image loading, preprocessing, and batch generation.
 * 
 * It scans the directory structure (assumes rootDir/{label}/<files>) and manages data access.
 */
class DataLoader {
public:
  /**
   * @brief Construct a new DataLoader object.
   * @param strategy The data loading strategy (EAGER or LAZY).
   */
  DataLoader(LoadStrategy strategy = LoadStrategy::EAGER);

  /**
   * @brief Load image dataset from a root directory.
   * @param rootDir Root directory path containing label subdirectories.
   * @param target Target size for resizing images ({0,0} keeps original size).
   * @param flag OpenCV imread flag.
   */
  void LoadImageDataset(const std::string &rootDir, MatSize target = {0, 0},
                        int flag = cv::IMREAD_GRAYSCALE);

  /**
   * @brief Get the total size of the dataset.
   * @return The number of samples.
   */
  size_t GetDataSize() const;

  /**
   * @brief Get a single data sample by index.
   * @param index The sample index.
   * @return Tensor representing the image.
   */
  Tensor GetData(size_t index) const;

  /**
   * @brief Get a single label by index.
   * @param index The sample index.
   * @return The integer label.
   */
  int GetLabel(size_t index) const;

  /**
   * @brief Retrieve a batch of image tensors stacked together.
   * @param startIdx The starting index of the batch.
   * @param batchSize The size of the batch.
   * @return Combined batch Tensor of shape (featureSize, actualBatchSize).
   */
  Tensor GetBatch(size_t startIdx, size_t batchSize) const;

  /**
   * @brief Retrieve a batch of labels as a Tensor.
   * @param startIdx The starting index of the batch.
   * @param batchSize The size of the batch.
   * @return Tensor representing the labels.
   */
  Tensor GetLabelBatch(size_t startIdx, size_t batchSize) const;

  /**
   * @brief Shuffle the dataset.
   * @param rng Random number generator engine for reproducibility.
   */
  void Shuffle(std::mt19937 &rng);

private:
  /**
   * @brief Reads an image from disk and converts it to a normalized Tensor [0.0, 1.0].
   * @param path File path of the image.
   * @param target Target size for resizing.
   * @param flag OpenCV imread flag.
   * @return Normalized Tensor of shape (totalPixels, 1).
   */
  Tensor ReadImage(const std::string &path, MatSize target, int flag) const;

  LoadStrategy m_strategy;
  std::vector<Tensor> m_data;
  std::vector<std::string> m_paths;
  std::vector<int> m_labels;
  MatSize m_targetSize;
  int m_readFlag;

  static inline const std::unordered_set<std::string> IMAGE_EXTENSIONS = {
      ".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG"};
};
} // namespace tardigrade::data