/**
 * @file main.cpp
 * @brief MNIST image classification training with SGD + 3-layer Dense MLP
 *
 * Network architecture:
 *   Input (28*28 = 784) -> Dense(784->256, ReLU)
 *                       -> Dense(256->128, ReLU)
 *                       -> Dense(128->10,  None)
 *                       -> Softmax -> Cross-Entropy Loss
 *
 * Softmax:
 *   sigma(z)_k = exp(z_k) / sum_j exp(z_j)
 *
 * Cross-Entropy Loss (single sample):
 *   L = -sum_k y_k * log(p_k)
 *   where y_k : one-hot label, p_k : softmax output
 *
 * Combined Softmax + CE backward gradient:
 *   dL/dz_k = p_k - y_k
 */

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "Layer.hpp"
#include "Optimizer.hpp"

namespace fs = std::filesystem;

using namespace tardigrade;
using namespace tardigrade::layer;
using namespace tardigrade::optimizer;

// ============================================================
// Data sample struct
// ============================================================
struct Sample
{
    std::string imagePath;
    int label;
};

// ============================================================
// Collect all samples from folder structure:
//   rootDir/{0~9}/*.jpg
// ============================================================
std::vector<Sample> LoadDataset(const std::string& rootDir)
{
    std::vector<Sample> dataset;

    for (int label = 0; label <= 9; ++label)
    {
        fs::path labelDir = fs::path(rootDir) / std::to_string(label);

        if (!fs::exists(labelDir) || !fs::is_directory(labelDir))
        {
            std::cerr << "[WARNING] Label directory not found: " << labelDir << "\n";
            continue;
        }

        for (const auto& entry : fs::directory_iterator(labelDir))
        {
            if (!entry.is_regular_file())
                continue;

            const auto ext = entry.path().extension().string();
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png")
                continue;

            dataset.push_back({ entry.path().string(), label });
        }
    }

    return dataset;
}

// ============================================================
// Read image -> normalized Tensor of shape {784, 1}
// Pixels scaled from [0, 255] to [0.0, 1.0]
// ============================================================
Tensor ImageToTensor(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if (img.empty())
        throw std::runtime_error("Cannot read image: " + path);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(28, 28));

    Tensor input({ 784, 1 });

    for (int i = 0; i < 784; ++i)
    {
        input[i] = static_cast<double>(resized.data[i]) / 255.0;
    }

    return input;
}

// ============================================================
// Softmax (numerically stable via max-subtraction)
//
// sigma(z)_k = exp(z_k - max_z) / sum_j exp(z_j - max_z)
// ============================================================
Tensor Softmax(const Tensor& logits)
{
    int n = static_cast<int>(logits.size());

    double maxVal = *std::max_element(logits.data(), logits.data() + n);

    Tensor result({ n, 1 });
    double sumExp = 0.0;

    for (int i = 0; i < n; ++i)
    {
        result[i] = std::exp(logits[i] - maxVal);
        sumExp += result[i];
    }

    for (int i = 0; i < n; ++i)
    {
        result[i] /= sumExp;
    }

    return result;
}

// ============================================================
// Cross-Entropy Loss (single sample)
//
// L = -log(p_label + eps)
// ============================================================
double CrossEntropyLoss(const Tensor& probs, int label)
{
    constexpr double eps = 1e-12;
    return -std::log(probs[label] + eps);
}

// ============================================================
// Combined Softmax + Cross-Entropy gradient
//
// dL/dz_k = p_k - y_k
// (y_k = 1 if k == label, else 0)
//
// Output shape: {10, 1}
// ============================================================
Tensor SoftmaxCEGradient(const Tensor& probs, int label)
{
    int n = static_cast<int>(probs.size());
    Tensor grad({ n, 1 });

    for (int i = 0; i < n; ++i)
    {
        grad[i] = probs[i] - (i == label ? 1.0 : 0.0);
    }

    return grad;
}

// ============================================================
// Main
// ============================================================
int main()
{
    // Hyperparameters
    const std::string datasetRoot = "/Users/home/Main/01_Dev/99_Dataset/MNIST/train";
    constexpr double learningRate  = 0.01;
    constexpr int    numEpochs     = 10;
    constexpr int    batchSize     = 1;  // SGD: one sample at a time

    // Load dataset
    std::cout << "[INFO] Loading dataset...\n";
    std::vector<Sample> dataset = LoadDataset(datasetRoot);
    std::cout << "[INFO] Total samples: " << dataset.size() << "\n";

    if (dataset.empty())
    {
        std::cerr << "[ERROR] Dataset is empty.\n";
        return 1;
    }

    // Build network:
    //   fc1: Dense(784 -> 256, ReLU)
    //   fc2: Dense(256 -> 128, ReLU)
    //   fc3: Dense(128 ->  10, None)   <- softmax applied externally
    Dense fc1(784, 256, batchSize, ACTIVATION::ReLU);
    Dense fc2(256, 128, batchSize, ACTIVATION::ReLU);
    Dense fc3(128,  10, batchSize, ACTIVATION::NONE);

    fc1.InitWeight();
    fc2.InitWeight();
    fc3.InitWeight();

    // SGD optimizer
    SGD sgd(learningRate);
    sgd.AddParameters(fc1.GetParameters());
    sgd.AddParameters(fc2.GetParameters());
    sgd.AddParameters(fc3.GetParameters());

    // Training loop
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        // Shuffle dataset each epoch
        std::shuffle(dataset.begin(), dataset.end(), rng);

        double totalLoss    = 0.0;
        int    correctCount = 0;
        int    processed    = 0;

        for (const auto& sample : dataset)
        {
            // Image -> Tensor
            Tensor input;
            try
            {
                input = ImageToTensor(sample.imagePath);
            }
            catch (const std::exception& e)
            {
                std::cerr << "[WARNING] Failed to load (" << sample.imagePath << "): " << e.what() << "\n";
                continue;
            }

            // Forward Pass
            //
            // fc1: input(784,1) -> z1(256,1)    [ReLU]
            // fc2: z1(256,1)    -> z2(128,1)    [ReLU]
            // fc3: z2(128,1)    -> logits(10,1) [None]
            // softmax: logits   -> probs(10,1)
            Tensor z1     = fc1.Forward(input);
            Tensor z2     = fc2.Forward(z1);
            Tensor logits = fc3.Forward(z2);
            Tensor probs  = Softmax(logits);

            // Loss: L = -log(p_label)
            double loss = CrossEntropyLoss(probs, sample.label);
            totalLoss += loss;

            // Accuracy
            int predicted = static_cast<int>(
                std::max_element(probs.data(), probs.data() + 10) - probs.data()
            );
            if (predicted == sample.label)
                ++correctCount;

            // Backward Pass
            //
            // dZ3 = p - y  (Softmax + CE combined gradient)
            // fc3.Backward(dZ3): updates fc3.m_gradient, returns dA2 = dL/dA2
            // fc2.Backward(dA2): updates fc2.m_gradient, returns dA1 = dL/dA1
            // fc1.Backward(dA1): updates fc1.m_gradient
            sgd.ZeroGrad();

            Tensor dZ3 = SoftmaxCEGradient(probs, sample.label);
            Tensor dA2 = fc3.Backward(dZ3);
            Tensor dA1 = fc2.Backward(dA2);
            fc1.Backward(dA1);

            // Parameter update: W = W - lr * grad
            sgd.Step();

            ++processed;

            // Log every 500 steps
            if (processed % 500 == 0)
            {
                double avgLoss = totalLoss / processed;
                double acc     = static_cast<double>(correctCount) / processed * 100.0;

                std::cout << "  [Epoch " << (epoch + 1) << "/" << numEpochs
                          << " | Step " << processed << "/" << dataset.size()
                          << "] Loss=" << avgLoss
                          << " | Acc=" << acc << "%\n";
            }
        }

        // Epoch summary
        double avgLoss = (processed > 0) ? (totalLoss / processed) : 0.0;
        double acc     = (processed > 0) ? (static_cast<double>(correctCount) / processed * 100.0) : 0.0;

        std::cout << "========================================\n";
        std::cout << "[Epoch " << (epoch + 1) << "/" << numEpochs << "] "
                  << "Loss=" << avgLoss
                  << " | Acc=" << acc << "%"
                  << " | Samples=" << processed << "\n";
        std::cout << "========================================\n\n";
    }

    std::cout << "[INFO] Training complete.\n";
    return 0;
}
