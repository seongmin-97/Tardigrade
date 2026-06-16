/**
 * @file main.cpp
 * @brief MNIST image classification training with SGD + 3-layer Dense MLP
 * Includes 1:1 numerical verification test comparing legacy vs autograd Dense.
 */

#include <iostream>
#include <random>
#include <cassert>
#include <cmath>

#include "DataLoader.hpp"
#include "Model.hpp"
#include "Metric.hpp"

// Autograd components
#include "Autograd.hpp"
#include "AutogradLayer.hpp"

using namespace tardigrade;
using namespace tardigrade::data;
using namespace tardigrade::layer;
using namespace tardigrade::loss;
using namespace tardigrade::optimizer;
using namespace tardigrade::model;
using namespace tardigrade::activation;
using namespace tardigrade::metric;

/**
 * @brief Run a 1:1 numerical verification test comparing the legacy Dense layer
 * (with manual backprop) against the new autograd Dense layer.
 */
void RunAutogradVerificationTest()
{
    std::cout << "[VERIFICATION] Starting Autograd vs Legacy Dense 1:1 test...\n";

    constexpr int inputSize = 4;
    constexpr int outputSize = 2;
    constexpr int batchSize = 3;

    // 1. Create layers
    Dense legacyLayer(inputSize, outputSize, batchSize, ACTIVATION::ReLU);
    autograd::layer::Dense autogradLayer(inputSize, outputSize, batchSize, activation::ACTIVATION::ReLU);

    // 2. Align weights
    // Copy weights from legacyLayer to autogradLayer
    for (int i = 0; i < legacyLayer.m_weight.dim(0); ++i)
    {
        for (int j = 0; j < legacyLayer.m_weight.dim(1); ++j)
        {
            autogradLayer.m_weight(i, j) = legacyLayer.m_weight(i, j);
        }
    }

    // 3. Create identical inputs and targets
    Tensor legacyInput({inputSize, batchSize});
    autograd::Tensor autogradInput({inputSize, batchSize}, false); // input doesn't require gradients

    // Fill with dummy data
    for (size_t i = 0; i < legacyInput.size(); ++i)
    {
        double val = static_cast<double>(i) * 0.25 + 0.1;
        legacyInput[i] = val;
        autogradInput[i] = val;
    }

    Tensor legacyTarget({outputSize, batchSize});
    autograd::Tensor autogradTarget({outputSize, batchSize}, false);
    for (size_t i = 0; i < legacyTarget.size(); ++i)
    {
        double val = (i % 2 == 0) ? 1.0 : 0.0;
        legacyTarget[i] = val;
        autogradTarget[i] = val;
    }

    // 4. Forward pass
    Tensor legacyOut = legacyLayer.Forward(legacyInput);
    autograd::Tensor autogradOut = autogradLayer.Forward(autogradInput);

    // Verify forward output
    double forwardDiffSum = 0.0;
    for (size_t i = 0; i < legacyOut.size(); ++i)
    {
        forwardDiffSum += std::abs(legacyOut[i] - autogradOut[i]);
    }
    std::cout << "  - Forward pass output L1 difference: " << forwardDiffSum << "\n";
    assert(forwardDiffSum < 1e-7 && "Forward pass numerical mismatch!");

    // 5. Loss computation
    // Autograd loss
    autograd::Tensor loss = autograd::mse_loss(autogradOut, autogradTarget);

    // Legacy manual loss gradient (dLoss/dOut)
    // MSE Loss: L = (1/N) * sum((pred - target)^2)
    // dL/dPred = (2/N) * (pred - target)
    Tensor dLoss_dOut(legacyOut.shape());
    double N = static_cast<double>(legacyOut.size());
    for (size_t i = 0; i < legacyOut.size(); ++i)
    {
        dLoss_dOut[i] = (2.0 / N) * (legacyOut[i] - legacyTarget[i]);
    }

    // 6. Backward pass
    // Legacy backward
    legacyLayer.Backward(dLoss_dOut); // updates legacyLayer.m_gradient

    // Autograd backward
    loss.Backward(); // propagates back and updates autogradLayer.m_weight.grad()

    // 7. Verify gradients
    autograd::Tensor autogradGrad = autogradLayer.m_weight.grad();
    double gradDiffSum = 0.0;
    
    for (size_t i = 0; i < legacyLayer.m_gradient.size(); ++i)
    {
        gradDiffSum += std::abs(legacyLayer.m_gradient[i] - autogradGrad[i]);
    }
    
    std::cout << "  - Gradient L1 difference: " << gradDiffSum << "\n";
    assert(gradDiffSum < 1e-7 && "Backward pass gradient numerical mismatch!");

    std::cout << "[VERIFICATION] SUCCESS! Autograd matches Legacy results exactly.\n\n";
}

int main()
{
    // Run verification first
    RunAutogradVerificationTest();

    // Hyperparameters
    const std::string datasetRoot = "/Users/home/Main/01_Dev/99_Dataset/MNIST/train";
    constexpr double learningRate = 0.01;
    constexpr int numEpochs = 10;
    constexpr int batchSize = 16;

    // --------------------------------------------------------
    // 1. Data Loading (Eager — 전체 데이터를 RAM에 적재)
    // --------------------------------------------------------
    std::cout << "[INFO] Loading dataset...\n";
    DataLoader loader(LoadStrategy::EAGER);
    loader.LoadImageDataset(datasetRoot, {28, 28}, cv::IMREAD_GRAYSCALE);

    if (loader.GetDataSize() == 0)
    {
        std::cerr << "[ERROR] Dataset is empty.\n";
        return 1;
    }

    // --------------------------------------------------------
    // 2. Model Construction
    // --------------------------------------------------------
    Model model;
    model.AddLayer(std::make_unique<Dense>(784, 256, batchSize, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(256, 128, batchSize, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(128, 10, batchSize, ACTIVATION::NONE));
    model.SetLossFunction(std::make_unique<SoftmaxCrossEntropy>(10, batchSize));
    model.SetOptimizer(std::make_unique<Adam>(learningRate));
    model.InitWeights();

    // Metric configuration
    model.SetMetric(std::make_unique<Accuracy>());

    // --------------------------------------------------------
    // 3. Training Loop
    // --------------------------------------------------------
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        loader.Shuffle(rng);

        double totalLoss = 0.0;
        double totalMetric = 0.0;
        int processed = 0;

        for (size_t i = 0; i < loader.GetDataSize(); i += batchSize)
        {
            size_t currentBatchSize = std::min(static_cast<size_t>(batchSize), loader.GetDataSize() - i);

            Tensor batchInput = loader.GetBatch(i, currentBatchSize);
            std::vector<int> batchLabels = loader.GetLabelBatch(i, currentBatchSize);

            Tensor batchTarget({1, static_cast<int>(currentBatchSize)});
            for (size_t b = 0; b < currentBatchSize; ++b)
            {
                batchTarget[b] = static_cast<double>(batchLabels[b]);
            }

            auto [loss, metricVal] = model.TrainStep(batchInput, batchTarget);

            totalLoss += loss * currentBatchSize;
            totalMetric += metricVal * currentBatchSize;

            processed += currentBatchSize;

            if ((processed / batchSize) % 10 == 0 || processed == loader.GetDataSize())
            {
                double avgLoss = totalLoss / processed;
                double acc = (totalMetric / processed) * 100.0;

                std::cout << "  [Epoch " << (epoch + 1) << "/" << numEpochs << " | Step " << processed << "/"
                          << loader.GetDataSize() << "] Loss=" << avgLoss << " | " << model.GetMetric()->GetName() << "=" << acc << "%\n";
            }
        }

        double avgLoss = (processed > 0) ? (totalLoss / processed) : 0.0;
        double acc = (processed > 0) ? ((totalMetric / processed) * 100.0) : 0.0;

        std::cout << "========================================\n";
        std::cout << "[Epoch " << (epoch + 1) << "/" << numEpochs << "] "
                  << "Loss=" << avgLoss << " | " << model.GetMetric()->GetName() << "=" << acc << "%"
                  << " | Samples=" << processed << "\n";
        std::cout << "========================================\n\n";
    }

    std::cout << "[INFO] Training complete.\n";
    return 0;
}
