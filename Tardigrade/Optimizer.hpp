#pragma once
#include <vector>
#include <utility>
#include <memory>
#include <cmath>

#include "Tensor.hpp"

namespace tardigrade 
{
    namespace optimizer 
    {
        // @brief Pair of pointers to a weight tensor and its gradient tensor.
        // @param first Pointer to the weight tensor.
        // @param second Pointer to the gradient tensor.
        using ParamGradPair = std::pair<Tensor*, Tensor*>;

        /**
         * @brief Base class for Optimizers.
        * 
        * Optimizers manage updating the parameters of a neural network given their gradients.
        */
        class Optimizer 
        {
        protected:
            std::vector<ParamGradPair> m_parameters;
            double m_learningRate;

        public:
            Optimizer(double learningRate) : m_learningRate(learningRate) {}
            virtual ~Optimizer() = default;

            /**
            * @brief Adds parameter and gradient tensor pairs to the optimizer.
            */
            void AddParameters(const std::vector<ParamGradPair>& params) 
            {
                for (const auto& pair : params)
                {
                    if (pair.first->shape() != pair.second->shape())
                        throw std::runtime_error("Parameter and gradient shape mismatch.");
                }
                m_parameters.insert(m_parameters.end(), params.begin(), params.end());
            }

            virtual void Step() = 0;
            virtual void ZeroGrad() = 0;
        };

        /**
         * @brief Stochastic Gradient Descent (SGD) Optimizer.
         * 
         * Mathematical Formula:
         * W_t = W_{t-1} - \eta \nabla L(W_{t-1})
         * (Where \eta is learning rate, \nabla L is the gradient)
       */
        class SGD : public Optimizer 
        {
        public:
            SGD(double learningRate);

            void Step() override;
            void ZeroGrad() override;
        };

        /**
        * @brief Adam Optimizer.
        * 
        * Mathematical Formulas:
        * m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
        * v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
        * \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
        * \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
        * W_t = W_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
        */
        class Adam : public Optimizer 
        {
        private:
            double m_beta1;
            double m_beta2;
            double m_epsilon;
            int m_t;                         // time step

            std::vector<Tensor> m_m;         // 1st moment vector
            std::vector<Tensor> m_v;         // 2nd moment vector
            bool m_initialized;

            void InitializeMoments();

        public:
            Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

            void Step() override;
            void ZeroGrad() override;
        };

    } // namespace optimizer
} // namespace tardigrade
