#include "Layer.hpp"

namespace tardigrade::layer
{
Dense::Dense(int inputSize, int outputSize, activation::ACTIVATION activation)
{
    m_inputSize = inputSize;
    m_outputSize = outputSize;

    switch (activation)
    {
        case activation::ACTIVATION::ReLU:
            m_activation = std::make_unique<activation::ReLU>(m_outputSize);
            break;
        case activation::ACTIVATION::Softmax:
            m_activation = std::make_unique<activation::Softmax>(m_outputSize);
            break;
        case activation::ACTIVATION::NONE:
        default:
            m_activation = std::make_unique<activation::None>(m_outputSize);
            break;
    }

    // Weights and bias require gradients
    m_weight = Tensor({m_inputSize, m_outputSize}, true);
    m_bias = Tensor({1, m_outputSize}, true);

    InitWeight();
}


Tensor Dense::Forward(const Tensor &input)
{
    int rows = input.dim(0);
    int batchSize = (input.rank() == 1) ? 1 : input.dim(1);

    if (rows != m_inputSize)
    {
        throw std::runtime_error("Input dimension mismatch in autograd Dense::Forward.");
    }

    /*
     * Forward linear activation calculation:
     *
     * \( Y = W^T X + b^T \cdot \mathbf{1} \)
     *
     * Mathematical breakdown:
     *  - \( X \): Input tensor of shape \( (D_{in}, N) \) where \( N = \text{batchSize} \) dynamically inferred
     *  - \( W \): Feature weight matrix of shape \( (D_{in}, D_{out}) \)
     *  - \( b \): Bias row vector of shape \( (1, D_{out}) \)
     *  - \( \mathbf{1} \): Row vector of ones of shape \( (1, N) \) for bias broadcasting
     *  - \( Y \): Logits output tensor of shape \( (D_{out}, N) \)
     */
    Tensor Y_feature = m_weight.transpose() % input;

    // Broadcast bias vector by multiplying m_bias^T with a constant row of ones.
    // ones: shape (1, batchSize) initialized to 1.0
    Tensor ones = Tensor::ones({1, batchSize}, false);

    Tensor Y_bias = m_bias.transpose() % ones;

    // Add feature and bias predictions: Y = Y_feature + Y_bias
    Tensor logits = Y_feature + Y_bias;

    // Apply polymorphic activation object Forward pass
    return m_activation->Forward(logits);
}


std::vector<Tensor> Dense::GetParameters() { return {m_weight, m_bias}; }

void Dense::InitWeight()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // He (Kaiming) initialization standard deviation: sqrt(2.0 / fan_in)
    double stddev = std::sqrt(2.0 / static_cast<double>(m_inputSize));
    std::normal_distribution<double> dist(0.0, stddev);

    for (int i = 0; i < m_inputSize; ++i)
    {
        for (int j = 0; j < m_outputSize; ++j)
        {
            m_weight(i, j) = dist(gen);
        }
    }

    // Initialize bias vector to zero
    for (int j = 0; j < m_outputSize; ++j)
    {
        m_bias(0, j) = 0.0;
    }
}

// ------------------------------------------------------------
// Conv2D Layer Implementation
// ------------------------------------------------------------

Conv2D::Conv2D(int inChannels, int outChannels, int kernelSize, int stride, int padding,
               activation::ACTIVATION activation)
{
    m_inChannels = inChannels;
    m_outChannels = outChannels;
    m_kernelSize = kernelSize;
    m_stride = stride;
    m_padding = padding;

    switch (activation)
    {
        case activation::ACTIVATION::ReLU:
            m_activation = std::make_unique<activation::ReLU>();
            break;
        case activation::ACTIVATION::Softmax:
            m_activation = std::make_unique<activation::Softmax>();
            break;
        case activation::ACTIVATION::NONE:
        default:
            m_activation = std::make_unique<activation::None>();
            break;
    }

    m_weight = Tensor({m_outChannels, m_inChannels, m_kernelSize, m_kernelSize}, true);
    m_bias = Tensor({m_outChannels}, true);
    InitWeight();
}


Tensor Conv2D::Forward(const Tensor& input)
{
    if (input.rank() != 4)
    {
        throw std::runtime_error("Conv2D::Forward expects a 4D input tensor [N, C, H, W].");
    }

    if (input.dim(1) != m_inChannels)
    {
        throw std::runtime_error("Conv2D::Forward input channel dimension mismatch.");
    }

    /*
     * Forward pass of Conv2D Layer (composition via primitives):
     *
     * \( Y = \text{convolve}(X, W) + b \)
     *
     * Bias broadcasting: reshape b from [C_out] to [1, C_out, 1, 1],
     * then add to Y [N, C_out, H_out, W_out] via broadcasting.
     *
     * Autograd graph: Im2colNode -> ReshapeNode -> MatMulNode -> ReshapeNode
     *                 -> PermuteNode -> AddNode -> ReshapeNode(bias) -> AddNode
     */
    Tensor Y = convolve(input, m_weight, m_stride, m_padding);

    // Add bias: reshape [C_out] -> [1, C_out, 1, 1] for broadcasting
    Tensor biasReshaped = m_bias.reshape({1, m_outChannels, 1, 1});
    Y = Y + biasReshaped;

    // Dynamic Polymorphic Activation Forward Pass
    return m_activation->Forward(Y);
}


std::vector<Tensor> Conv2D::GetParameters() { return {m_weight, m_bias}; }

void Conv2D::InitWeight()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // He (Kaiming) initialization for Conv2D: stddev = sqrt(2.0 / (inChannels * Kh * Kw))
    double fanIn = static_cast<double>(m_inChannels * m_kernelSize * m_kernelSize);
    double stddev = std::sqrt(2.0 / fanIn);
    std::normal_distribution<double> dist(0.0, stddev);

    double *wData = m_weight.data();
    size_t totalW = m_weight.size();
    for (size_t i = 0; i < totalW; ++i)
    {
        wData[i] = dist(gen);
    }

    double *bData = m_bias.data();
    size_t totalB = m_bias.size();
    for (size_t i = 0; i < totalB; ++i)
    {
        bData[i] = 0.0;
    }
}

// ------------------------------------------------------------
// MaxPool2D Layer Implementation
// ------------------------------------------------------------

MaxPool2D::MaxPool2D(int kernelSize, int stride, int padding)
{
    m_kernelSize = kernelSize;
    m_stride = (stride <= 0) ? kernelSize : stride;
    m_padding = padding;
}

Tensor MaxPool2D::Forward(const Tensor& input)
{
    if (input.rank() != 4)
    {
        throw std::runtime_error("MaxPool2D::Forward expects a 4D input tensor [N, C, H, W].");
    }

    int N = input.dim(0);
    int C = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);
    int k = m_kernelSize;
    int s = m_stride;
    int p = m_padding;

    int outH = (H + 2 * p - k) / s + 1;
    int outW = (W + 2 * p - k) / s + 1;

    /*
     * MaxPool2D as composition of primitive ops:
     *
     * \( \text{col} = \text{im2col}(X, k, k, s, s, p, p) \quad [C \cdot k^2,\; N \cdot H_{out} \cdot W_{out}] \)
     * \( \text{grouped} = \text{reshape}(\text{col}, [C, k^2, N \cdot H_{out} \cdot W_{out}]) \)
     * \( \text{maxVals} = \max_{\text{axis}=1}(\text{grouped}) \quad [C, N \cdot H_{out} \cdot W_{out}] \)
     * \( Y = \text{permute}(\text{reshape}(\text{maxVals}, [C, N, H_{out}, W_{out}]), \{1,0,2,3\}) \)
     *
     * Autograd: Im2colNode -> ReshapeNode -> ReduceMaxNode -> ReshapeNode -> PermuteNode
     * Backward of ReduceMaxNode: scatter-add gradient to argmax positions.
     */
    Tensor col = im2col(input, k, k, s, s, p, p);                 // [C*k*k, N*outH*outW]
    Tensor grouped = col.reshape({C, k * k, N * outH * outW});    // [C, k*k, N*outH*outW]
    Tensor maxVals = reduce_max(grouped, 1, false);                // [C, N*outH*outW]  ReduceMaxNode
    return maxVals.reshape({C, N, outH, outW}).permute({1, 0, 2, 3}); // [N, C, outH, outW]
}

// ------------------------------------------------------------
// AvgPool2D Layer Implementation
// ------------------------------------------------------------

AvgPool2D::AvgPool2D(int kernelSize, int stride, int padding)
{
    m_kernelSize = kernelSize;
    m_stride = (stride <= 0) ? kernelSize : stride;
    m_padding = padding;
}

Tensor AvgPool2D::Forward(const Tensor& input)
{
    if (input.rank() != 4)
    {
        throw std::runtime_error("AvgPool2D::Forward expects a 4D input tensor [N, C, H, W].");
    }

    int N = input.dim(0);
    int C = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);
    int k = m_kernelSize;
    int s = m_stride;
    int p = m_padding;

    int outH = (H + 2 * p - k) / s + 1;
    int outW = (W + 2 * p - k) / s + 1;

    /*
     * AvgPool2D as composition of primitive ops:
     *
     * \( \text{col} = \text{im2col}(X, k, k, s, s, p, p) \quad [C \cdot k^2,\; N \cdot H_{out} \cdot W_{out}] \)
     * \( \text{grouped} = \text{reshape}(\text{col}, [C, k^2, N \cdot H_{out} \cdot W_{out}]) \)
     * \( \text{colSum} = \sum_{\text{axis}=1}(\text{grouped}) \quad [C, N \cdot H_{out} \cdot W_{out}] \)
     * \( \text{avg} = \text{colSum} / k^2 \quad [C, N \cdot H_{out} \cdot W_{out}] \)
     * \( Y = \text{permute}(\text{reshape}(\text{avg}, [C, N, H_{out}, W_{out}]), \{1,0,2,3\}) \)
     *
     * Autograd: Im2colNode -> ReshapeNode -> SumNode -> DivNode -> ReshapeNode -> PermuteNode
     */
    Tensor col = im2col(input, k, k, s, s, p, p);                  // [C*k*k, N*outH*outW]
    Tensor grouped = col.reshape({C, k * k, N * outH * outW});     // [C, k*k, N*outH*outW]
    Tensor colSum = sum(grouped, 1, false);                         // [C, N*outH*outW]   SumNode
    Tensor avg = colSum / static_cast<double>(k * k);              // [C, N*outH*outW]   DivNode
    return avg.reshape({C, N, outH, outW}).permute({1, 0, 2, 3});  // [N, C, outH, outW]
}

// ------------------------------------------------------------
// Flatten Layer Implementation
// ------------------------------------------------------------

Tensor Flatten::Forward(const Tensor &input)
{
    if (input.rank() == 2)
    {
        return input;
    }

    if (input.rank() != 4)
    {
        throw std::runtime_error("Flatten::Forward expects 4D input tensor [N, C, H, W].");
    }

    int N = input.dim(0);
    int C = input.dim(1);
    int H = input.dim(2);
    int W = input.dim(3);

    int featureDim = C * H * W;

    /*
     * Flattening 4D tensor [N, C, H, W] to 2D matrix [C * H * W, N]:
     *
     * \( Y_{(c \cdot H \cdot W + h \cdot W + w), n} = X_{n, c, h, w} \)
     */
    Tensor perm = input.permute({1, 2, 3, 0});
    Tensor flat = perm.reshape({featureDim, N});

    return flat;
}
} // namespace tardigrade::layer