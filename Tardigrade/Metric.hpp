#pragma once
#include <string>
#include "Tensor.hpp"

namespace tardigrade::metric
{
    /**
     * @brief Abstract base class for evaluation metrics.
     */
    class Metric
    {
    public:
        virtual ~Metric() = default;

        /**
         * @brief Evaluate the metric using prediction and target tensors.
         * @param prediction Predicted output from the model (typically logits of shape [C, B]).
         * @param target Ground truth target labels (typically shape [1, B]).
         * @return Scalar evaluation metric value (e.g. accuracy ratio).
         */
        virtual double Evaluate(const Tensor& prediction, const Tensor& target) = 0;

        /**
         * @brief Get the metric's name.
         * @return Name of the metric.
         */
        virtual std::string GetName() const = 0;
    };

    /**
     * @brief Classification accuracy metric.
     * Computes the ratio of correct predictions to total predictions on a batch.
     */
    class Accuracy : public Metric
    {
    public:
        Accuracy() = default;

        double Evaluate(const Tensor& prediction, const Tensor& target) override;
        std::string GetName() const override;
    };
}
