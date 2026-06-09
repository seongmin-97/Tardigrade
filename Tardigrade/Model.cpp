#include "Model.hpp"


using namespace tardigrade;
using namespace tardigrade::model;

// ------------------------------------------------------------
// Model Setup: Layer, Optimizer and Loss Function configuration
// ------------------------------------------------------------
void Model::AddLayer(std::unique_ptr<layer::Layer> layer)
{
    m_layers.push_back(std::move(layer));
}

void Model::SetOptimizer(std::unique_ptr<optimizer::Optimizer> opt)
{
    m_optimizer = std::move(opt);
}

void Model::SetLossFunction(std::unique_ptr<loss::Loss> lossFunc)
{
    m_lossFunction = std::move(lossFunc);
}

// ------------------------------------------------------------
// InitWeights: Initialize weights for all layers and register parameters
// ------------------------------------------------------------
void Model::InitWeights()
{
    if (!m_optimizer)
    {
        throw std::runtime_error("Model: Optimizer must be set before InitWeights()");
    }

    for (auto& layer : m_layers)
    {
        // If the layer is Dense, initialize weights and register parameters with optimizer
        auto* dense = dynamic_cast<layer::Dense*>(layer.get());
        if (dense)
        {
            dense->InitWeight();
            m_optimizer->AddParameters(dense->GetParameters());
        }
    }
}

// ------------------------------------------------------------
// Forward: Propagates input through all layers sequentially
// ------------------------------------------------------------
Tensor Model::Forward(const Tensor& input)
{
    Tensor current = input;

    for (auto& layer : m_layers)
    {
        current = layer->Forward(current);
    }

    return current;
}

// ------------------------------------------------------------
// Backward: Propagates gradients backwards through layers in reverse order
// ------------------------------------------------------------
void Model::Backward(const Tensor& gradOutput)
{
    Tensor current = gradOutput;

    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
    {
        current = (*it)->Backward(current);
    }
}

// ------------------------------------------------------------
// TrainStep: Performs a single training step
//
// 1. ZeroGrad     - Resets gradients of parameters
// 2. Forward      - Propagates input to logits
// 3. Loss.Forward - Computes loss value from logits and label
// 4. Prediction   - Predicts class label
// 5. Backward     - Backpropagates loss gradient
// 6. Step         - Updates parameters using the optimizer
// ------------------------------------------------------------
double Model::TrainStep(const Tensor& input, const Tensor& target, Tensor& predicted)
{
    if (!m_optimizer || !m_lossFunction)
    {
        throw std::runtime_error("Model: Optimizer and LossFunction must be set before training");
    }

    // 1. ZeroGrad
    m_optimizer->ZeroGrad();

    // 2. Forward
    Tensor logits = Forward(input);

    // 3. Loss computation
    double lossValue = m_lossFunction->Forward(logits, target);

    // 4. Class prediction determination (Argmax per column)
    int C = logits.dim(0);
    int B = (logits.rank() == 1) ? 1 : logits.dim(1);

    if (predicted.shape() != std::vector<int>{ 1, B })
    {
        predicted = Tensor({ 1, B });
    }

    auto* sce = dynamic_cast<loss::SoftmaxCrossEntropy*>(m_lossFunction.get());
    const Tensor& scoreTensor = (sce != nullptr) ? sce->GetProbs() : logits;

    for (int i = 0; i < B; ++i)
    {
        double maxVal = scoreTensor(0, i);
        int argMax = 0;
        for (int j = 1; j < C; ++j)
        {
            if (scoreTensor(j, i) > maxVal)
            {
                maxVal = scoreTensor(j, i);
                argMax = j;
            }
        }
        predicted[i] = static_cast<double>(argMax);
    }

    // 5. Backward pass
    Tensor grad = m_lossFunction->Backward();
    Backward(grad);

    // 6. Parameter update
    m_optimizer->Step();

    return lossValue;
}

Tensor Model::Predict(const Tensor& input)
{
    return Forward(input);
}

const std::vector<std::unique_ptr<layer::Layer>>& Model::GetLayers() const
{
    return m_layers;
}

optimizer::Optimizer* Model::GetOptimizer() const
{
    return m_optimizer.get();
}

loss::Loss* Model::GetLossFunction() const
{
    return m_lossFunction.get();
}
