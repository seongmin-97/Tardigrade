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

void testNDTensorAndConv()
{
    std::cout << "[TEST] 1. Testing N-D Sub-Tensor [][] indexing...\n";
    Tensor t3d({2, 3, 4});
    t3d(0, 1, 2) = 42.0;

    // PyTorch style [][][] access check via operator double()
    Tensor sub2d = t3d[0];      // Shape: [3, 4]
    Tensor sub1d = t3d[0][1];   // Shape: [4]
    double val = t3d[0][1][2];  // Triple indexing chaining to scalar double!

    std::cout << " -> t3d[0][1][2] = " << val << " (Expected: 42)\n";

    std::cout << "[TEST] 2. Testing Broadcasting Add & Scalar Operators...\n";
    Tensor A({2, 3, 4}, true);
    A.fill(2.0);
    Tensor B({3, 1}, true);
    B.fill(3.0);

    Tensor C = A + B + 5.0; // Broadcasting shape [2, 3, 4]
    std::cout << " -> C shape: [" << C.dim(0) << ", " << C.dim(1) << ", " << C.dim(2) << "], C(0,0,0)=" << C(0,0,0) << " (Expected: 10)\n";

    Tensor loss = sum(C);
    loss.Backward();
    std::cout << " -> A grad shape: [" << A.grad().dim(0) << ", " << A.grad().dim(1) << ", " << A.grad().dim(2) << "], val=" << A.grad().data()[0] << " (Expected: 1)\n";
    std::cout << " -> B grad shape: [" << B.grad().dim(0) << ", " << B.grad().dim(1) << "], val=" << B.grad().data()[0] << " (Expected: 8)\n";


    std::cout << "[TEST] 3. Testing Conv2d & MaxPool2d Forward & Backward...\n";
    Tensor img({1, 1, 5, 5}, true); // [N, C, H, W]
    img.fill(1.0);
    Tensor weight({1, 1, 3, 3}, true); // [C_out, C_in, Kh, Kw]
    weight.fill(0.5);

    Tensor convOut = conv2d(img, weight, Tensor(), 1, 0); // Output: [1, 1, 3, 3]
    std::cout << " -> Conv2d output shape: [" << convOut.dim(0) << ", " << convOut.dim(1) << ", " << convOut.dim(2) << ", " << convOut.dim(3) << "], val=" << convOut(0,0,0,0) << " (Expected: 4.5)\n";

    Tensor poolOut = maxPool2d(convOut, 2, 1, 0); // Output: [1, 1, 2, 2]
    std::cout << " -> MaxPool2d output shape: [" << poolOut.dim(0) << ", " << poolOut.dim(1) << ", " << poolOut.dim(2) << ", " << poolOut.dim(3) << "]\n";

    Tensor convLoss = sum(poolOut);
    convLoss.Backward();
    std::cout << " -> Conv2d img grad shape: [" << img.grad().dim(0) << ", " << img.grad().dim(1) << ", " << img.grad().dim(2) << ", " << img.grad().dim(3) << "]\n";
    std::cout << "[TEST] All N-D Tensor & Conv Tests Passed Successfully!\n\n";
}

void testCNNNetwork()
{
    std::cout << "[TEST] 4. Testing CNN Network Construction (Conv2D -> MaxPool2D -> AvgPool2D -> Flatten -> Dense)...\n";

    Model cnnModel;
    // Layer 0: Conv2D (1 in_channel, 4 out_channels, 3x3 kernel, stride 1, padding 1) + ReLU
    cnnModel.AddLayer(std::make_unique<Conv2D>(1, 4, 3, 1, 1, ACTIVATION::ReLU));
    // Layer 1: MaxPool2D (2x2 kernel, stride 2)
    cnnModel.AddLayer(std::make_unique<MaxPool2D>(2, 2, 0));
    // Layer 2: Conv2D (4 in_channels, 8 out_channels, 3x3 kernel, stride 1, padding 1) + ReLU
    cnnModel.AddLayer(std::make_unique<Conv2D>(4, 8, 3, 1, 1, ACTIVATION::ReLU));
    // Layer 3: AvgPool2D (2x2 kernel, stride 2)
    cnnModel.AddLayer(std::make_unique<AvgPool2D>(2, 2, 0));
    // Layer 4: Flatten (8 x 7 x 7 = 392 features)
    cnnModel.AddLayer(std::make_unique<Flatten>());
    // Layer 5: Dense (392 -> 10)
    cnnModel.AddLayer(std::make_unique<Dense>(392, 10, 2, ACTIVATION::NONE));

    cnnModel.SetLossFunction(std::make_unique<SoftmaxCrossEntropy>(10, 2));
    cnnModel.SetOptimizer(std::make_unique<Adam>(0.001));

    // Input shape: [N=2, C=1, H=28, W=28]
    Tensor inputImg({2, 1, 28, 28}, true);
    inputImg.fill(0.5);

    Tensor targetLabel({2});
    targetLabel(0) = 0.0; // class 0 for batch 0
    targetLabel(1) = 1.0; // class 1 for batch 1

    auto [lossVal, accVal] = cnnModel.TrainStep(inputImg, targetLabel);

    std::cout << " -> CNN TrainStep Loss: " << lossVal << ", Accuracy: " << accVal * 100.0 << "%\n";
    std::cout << "[TEST] CNN Network Test Passed Successfully!\n\n";
}




int main()
{
    testNDTensorAndConv();
    testCNNNetwork();


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
