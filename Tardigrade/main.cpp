/**
 * @file main.cpp
 * @brief MNIST image classification training with SGD + 3-layer Dense MLP
 *
 * Network architecture:
 *   Input (28*28 = 784) -> Dense(784->256, ReLU)
 *                       -> Dense(256->128, ReLU)
 *                       -> Dense(128->10,  None)
 *                       -> SoftmaxCrossEntropy Loss
 *
 * 이전 버전에서 main.cpp에 직접 구현되어 있던 로직들이
 * 각 프레임워크 컴포넌트로 분리되었다:
 *   - LoadDataset(), ImageToTensor() → DataLoader
 *   - Softmax(), CrossEntropyLoss(), SoftmaxCEGradient() → Loss (SoftmaxCrossEntropy)
 *   - 레이어 관리, Forward/Backward 체인 → Model
 */

#include <iostream>
#include <random>

#include "DataLoader.hpp"
#include "Model.hpp"

using namespace tardigrade;
using namespace tardigrade::data;
using namespace tardigrade::layer;
using namespace tardigrade::loss;
using namespace tardigrade::optimizer;
using namespace tardigrade::model;
using namespace tardigrade::activation;

// ============================================================
// Main
// ============================================================
int main()
{
    // Hyperparameters
    const std::string datasetRoot = "/Users/home/Main/01_Dev/99_Dataset/MNIST/train";
    constexpr double learningRate  = 0.01;
    constexpr int    numEpochs     = 10;

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
    //
    //   fc1: Dense(784 -> 256, ReLU)
    //   fc2: Dense(256 -> 128, ReLU)
    //   fc3: Dense(128 ->  10, None)  <- softmax은 Loss에서 수행
    //
    //   Loss: SoftmaxCrossEntropy
    //   Optimizer: SGD
    // --------------------------------------------------------
    Model model;
    model.AddLayer(std::make_unique<Dense>(784, 256, 1, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(256, 128, 1, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(128,  10, 1, ACTIVATION::NONE));
    model.SetLossFunction(std::make_unique<SoftmaxCrossEntropy>(10, 1));
    model.SetOptimizer(std::make_unique<SGD>(learningRate));
    model.InitWeights();

    // --------------------------------------------------------
    // 3. Training Loop
    // --------------------------------------------------------
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        // Shuffle dataset each epoch
        loader.Shuffle(rng);

        double totalLoss    = 0.0;
        int    correctCount = 0;
        int    processed    = 0;

        for (size_t i = 0; i < loader.GetDataSize(); ++i)
        {
            // TrainStep: ZeroGrad → Forward → Loss → Backward → Step
            int predicted;
            double loss = model.TrainStep(loader.GetData(i), loader.GetLabel(i), predicted);

            totalLoss += loss;

            if (predicted == loader.GetLabel(i))
            {
                ++correctCount;
            }

            ++processed;

            // Log every 500 steps
            if (processed % 500 == 0)
            {
                double avgLoss = totalLoss / processed;
                double acc     = static_cast<double>(correctCount) / processed * 100.0;

                std::cout << "  [Epoch " << (epoch + 1) << "/" << numEpochs
                          << " | Step " << processed << "/" << loader.GetDataSize()
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
