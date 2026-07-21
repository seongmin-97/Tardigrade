/**
 * @file main.cpp
 * @brief MNIST image classification training with Adam + 3-layer Dense MLP using Autograd
 */

#include <cmath>
#include <iostream>
#include <random>

#include "Autograd.hpp"
#include "DataLoader.hpp"
#include "Metric.hpp"
#include "Model.hpp"

using namespace tardigrade;
using namespace tardigrade::data;
using namespace tardigrade::layer;
using namespace tardigrade::loss;
using namespace tardigrade::optimizer;
using namespace tardigrade::model;
using namespace tardigrade::activation;
using namespace tardigrade::metric;

int main()
{

    // Hyperparameters
    const std::string datasetRoot = "/Users/home/Main/01_Dev/99_Dataset/MNIST/train";
    constexpr double learningRate = 0.002;
    constexpr int numEpochs = 100;
    constexpr int batchSize = 16;

    // --------------------------------------------------------
    // 1. Data Loading (Eager — 전체 데이터를 RAM에 적재)
    // --------------------------------------------------------
    std::cout << "[INFO] Loading dataset...\n";
    DataLoader loader(LoadStrategy::EAGER);
    loader.SetBatchSize(batchSize);
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
    model.AddLayer(std::make_unique<Dense>(784, 200, batchSize, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(200, 150, batchSize, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(150, 150, batchSize, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(150, 100, batchSize, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(100, 50, batchSize, ACTIVATION::ReLU));
    model.AddLayer(std::make_unique<Dense>(50, 10, batchSize, ACTIVATION::NONE));
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
        model.ResetMetrics();

        for (size_t i = 0; i < loader.GetDataSize(); i += batchSize)
        {
            Tensor batchInput = loader.GetBatch(i);
            Tensor batchTarget = loader.GetLabelBatch(i);

            model.TrainStep(batchInput, batchTarget);
            model.PrintProgress(loader.GetDataSize(), epoch + 1, numEpochs);
        }
    }

    std::cout << "[INFO] Training complete.\n";
    return 0;
}
