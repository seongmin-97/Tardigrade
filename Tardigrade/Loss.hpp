#pragma once
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "Tensor.hpp"

namespace tardigrade::loss
{
    /**
     * @brief Loss 추상 클래스
     *
     * 모든 손실 함수의 기반 클래스.
     * Forward()로 손실값을 계산하고, Backward()로 gradient를 반환한다.
     * Activation 클래스와 구조적으로 유사하지만, 역할이 근본적으로 다르다:
     *   - Activation: 비선형 변환 (레이어 내부)
     *   - Loss: 예측과 정답 사이의 오차 측정 (학습 목적)
     */
    class Loss
    {
    public:
        Loss(int inputSize, int batchSize);
        virtual ~Loss() = default;

        /**
         * @brief 순전파: 예측값과 라벨로부터 손실값 계산
         * @param prediction 모델의 출력 (logits 또는 확률)
         * @param label 정답 라벨 (정수)
         * @return 스칼라 손실값
         */
        virtual double Forward(const Tensor& prediction, int label) = 0;

        /**
         * @brief 역전파: 예측값에 대한 gradient 계산
         * @return dL/d(prediction) gradient Tensor
         */
        virtual Tensor Backward() = 0;

    protected:
        int m_inputSize;
        int m_batchSize;

        Tensor m_prediction;   // Forward에서 캐시
        Tensor m_gradient;     // Backward에서 계산
    };

    /**
     * @brief Softmax + Cross-Entropy Loss 결합 손실 함수
     *
     * Softmax와 Cross-Entropy를 결합하면 backward gradient가
     * 매우 깔끔하게 떨어지는 수학적 장점이 있다.
     *
     * Forward (Softmax):
     *   σ(z)_k = exp(z_k - max(z)) / Σ_j exp(z_j - max(z))
     *
     * Forward (Cross-Entropy Loss):
     *   L = -log(σ(z)_label + ε)
     *
     * Backward (Combined gradient):
     *   dL/dz_k = σ(z)_k - y_k
     *   where y_k = 1 if k == label, else 0 (one-hot)
     */
    class SoftmaxCrossEntropy : public Loss
    {
    public:
        SoftmaxCrossEntropy(int inputSize, int batchSize);

        double Forward(const Tensor& logits, int label) override;
        Tensor Backward() override;

        /// 마지막 Forward의 softmax 확률 접근 (예측 클래스 판별용)
        const Tensor& GetProbs() const;

    private:
        Tensor m_probs;   // softmax 출력 캐시
        int m_label;      // Forward에서 사용한 라벨 캐시
    };

    /**
     * @brief Mean Squared Error (MSE) 손실 함수
     *
     * 회귀 문제를 위한 기본 손실 함수.
     *
     * Forward:
     *   L = (1/n) Σ_i (y_pred_i - y_true_i)²
     *
     * Backward:
     *   dL/dy_pred_i = (2/n)(y_pred_i - y_true_i)
     */
    class MSE : public Loss
    {
    public:
        MSE(int inputSize, int batchSize);

        /// 회귀용 Forward: 예측값과 타겟 Tensor 비교
        double Forward(const Tensor& prediction, const Tensor& target);

        /// 분류용 Forward: int label을 one-hot 변환 후 계산
        double Forward(const Tensor& prediction, int label) override;

        Tensor Backward() override;

    private:
        Tensor m_target;  // Forward에서 사용한 타겟 캐시
    };
}
