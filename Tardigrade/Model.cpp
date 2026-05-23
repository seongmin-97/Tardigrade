#include "Model.hpp"


using namespace tardigrade;
using namespace tardigrade::model;

// ============================================================
// AddLayer / SetOptimizer / SetLossFunction
// ============================================================
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

// ============================================================
// InitWeights
//
// 각 레이어의 가중치를 초기화하고,
// Optimizer에 (weight, gradient) 쌍을 자동 등록한다.
// ============================================================
void Model::InitWeights()
{
    if (!m_optimizer)
    {
        throw std::runtime_error("Model: Optimizer must be set before InitWeights()");
    }

    for (auto& layer : m_layers)
    {
        // Dense 레이어인 경우 InitWeight + 파라미터 등록
        auto* dense = dynamic_cast<layer::Dense*>(layer.get());
        if (dense)
        {
            dense->InitWeight();
            m_optimizer->AddParameters(dense->GetParameters());
        }
    }
}

// ============================================================
// Forward: 입력 → 모든 레이어 순서대로 → 출력
// ============================================================
Tensor Model::Forward(const Tensor& input)
{
    Tensor current = input;

    for (auto& layer : m_layers)
    {
        current = layer->Forward(current);
    }

    return current;
}

// ============================================================
// Backward: gradient를 역순으로 전파
// ============================================================
void Model::Backward(const Tensor& gradOutput)
{
    Tensor current = gradOutput;

    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
    {
        current = (*it)->Backward(current);
    }
}

// ============================================================
// TrainStep: 한 step 학습 통합 수행
//
// 1. ZeroGrad    — 이전 gradient 초기화
// 2. Forward     — 입력 → 모든 레이어 → logits
// 3. Loss.Forward — logits + label → 손실값
// 4. Loss.Backward — dL/d(logits)
// 5. Layer Backward — gradient 역전파
// 6. Optimizer.Step — 파라미터 업데이트
// ============================================================
double Model::TrainStep(const Tensor& input, int label, int& predicted)
{
    if (!m_optimizer || !m_lossFunction)
    {
        throw std::runtime_error("Model: Optimizer and LossFunction must be set before training");
    }

    // 1. ZeroGrad
    m_optimizer->ZeroGrad();

    // 2. Forward
    Tensor logits = Forward(input);

    // 3. Loss
    double lossValue = m_lossFunction->Forward(logits, label);

    // 4. 예측 클래 스 판별
    auto* sce = dynamic_cast<loss::SoftmaxCrossEntropy*>(m_lossFunction.get());
    if (sce)
    {
        const Tensor& probs = sce->GetProbs();
        predicted = static_cast<int>(
            std::max_element(probs.data(), probs.data() + probs.size()) - probs.data()
        );
    }
    else
    {
        predicted = static_cast<int>(
            std::max_element(logits.data(), logits.data() + logits.size()) - logits.data()
        );
    }

    // 5. Backward
    Tensor grad = m_lossFunction->Backward();
    Backward(grad);

    // 6. Step
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
