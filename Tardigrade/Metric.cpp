#include "Metric.hpp"
#include <stdexcept>
#include <vector>

using namespace tardigrade;
using namespace tardigrade::metric;

// ------------------------------------------------------------
// Accuracy Implementation
// ------------------------------------------------------------
double Accuracy::Evaluate(const Tensor& prediction, const Tensor& target)
{
    // B: Batch size, C: Channels/Classes
    int C = prediction.dim(0);
    int B = (prediction.rank() == 1) ? 1 : prediction.dim(1);

    int targetSize = target.size();
    if (targetSize != B)
    {
        throw std::invalid_argument("Accuracy: Target size does not match prediction batch size.");
    }

    int correctCount = 0;
    for (int b = 0; b < B; ++b)
    {
        double maxVal = prediction(0, b);
        int argMax = 0;
        for (int c = 1; c < C; ++c)
        {
            if (prediction(c, b) > maxVal)
            {
                maxVal = prediction(c, b);
                argMax = c;
            }
        }

        if (argMax == static_cast<int>(target[b]))
        {
            correctCount++;
        }
    }

    if (B == 0)
    {
        return 0.0;
    }

    return static_cast<double>(correctCount) / B;
}

std::string Accuracy::GetName() const
{
    return "Accuracy";
}
