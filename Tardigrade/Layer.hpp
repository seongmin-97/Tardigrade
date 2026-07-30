#pragma once
#include <vector>
#include <memory>
#include <random>
#include <cmath>

#include "Tensor.hpp"
#include "Autograd.hpp"
#include "Activation.hpp" // For ACTIVATION enum

namespace tardigrade::layer
{
    /**
     * @brief Abstract base class representing a layer in the autograd network.
     */
    class Layer
    {
    public:
        virtual ~Layer() = default;

        /**
         * @brief Computes the forward pass of the layer.
         * @param input The input tensor to this layer.
         * @return The output tensor after transformation.
         */
        virtual Tensor Forward(const Tensor& input) = 0;

        /**
         * @brief Retrieves references to the parameter tensors of the layer.
         * @return Vector of parameter Tensors.
         */
        virtual std::vector<Tensor> GetParameters()
        {
            return {};
        }

        /**
         * @brief Initializes weights dynamically.
         */
        virtual void InitWeight() {}
    };

    /**
     * @brief Dense (Fully Connected) layer using pure Tensor Autograd operations.
     */
    class Dense : public Layer
    {
    public:
        /**
         * @brief Construct a new Dense object.
         * @param inputSize Number of input features (excluding bias).
         * @param outputSize Number of output features.
         * @param activation Type of activation function to apply.
         */
        Dense(int inputSize, int outputSize, activation::ACTIVATION activation = activation::ACTIVATION::NONE);

        Tensor Forward(const Tensor& input) override;

        std::vector<Tensor> GetParameters() override;

        /**
         * @brief Initializes weight matrix using He (Kaiming) normal initialization.
         */
        void InitWeight() override;

    public:
        int m_inputSize;                  ///< Input feature dimension (excluding bias)
        int m_outputSize;                 ///< Output size (number of neurons)

        Tensor m_weight;                                      ///< Weight matrix of shape (inputSize, outputSize)
        Tensor m_bias;                                        ///< Bias vector of shape (1, outputSize)
        std::unique_ptr<activation::Activation> m_activation; ///< Polymorphic activation object
    };

    /**
     * @brief 2D Convolutional Layer using pure Tensor Autograd operations.
     *
     * Mathematical Formula:
     * Forward:
     *   \( Y = \text{conv2d}(X, W, b, S, P) \)
     * where:
     *   - \( X \in \mathbb{R}^{N \times C_{in} \times H \times W} \)
     *   - \( W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w} \)
     *   - \( b \in \mathbb{R}^{C_{out}} \)
     */
    class Conv2D : public Layer
    {
    public:
        Conv2D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, activation::ACTIVATION activation = activation::ACTIVATION::NONE);

        Tensor Forward(const Tensor& input) override;

        std::vector<Tensor> GetParameters() override;

        void InitWeight() override;

    public:
        int m_inChannels;
        int m_outChannels;
        int m_kernelSize;
        int m_stride;
        int m_padding;

        Tensor m_weight;
        Tensor m_bias;
        std::unique_ptr<activation::Activation> m_activation;
    };


    /**
     * @brief 2D Max Pooling Layer.
     *
     * Mathematical Formula:
     * Forward:
     *   \( Y_{n, c, oh, ow} = \max_{kh, kw} X_{n, c, oh \cdot S + kh, ow \cdot S + kw} \)
     */
    class MaxPool2D : public Layer
    {
    public:
        MaxPool2D(int kernelSize, int stride = -1, int padding = 0);

        Tensor Forward(const Tensor& input) override;

    public:
        int m_kernelSize;
        int m_stride;
        int m_padding;
    };

    /**
     * @brief 2D Average Pooling Layer.
     *
     * Mathematical Formula:
     * Forward:
     *   \( Y_{n, c, oh, ow} = \frac{1}{K_h \times K_w} \sum_{kh=0}^{K_h-1} \sum_{kw=0}^{K_w-1} X_{n, c, oh \cdot S + kh, ow \cdot S + kw} \)
     */
    class AvgPool2D : public Layer
    {
    public:
        AvgPool2D(int kernelSize, int stride = -1, int padding = 0);

        Tensor Forward(const Tensor& input) override;

    public:
        int m_kernelSize;
        int m_stride;
        int m_padding;
    };

    /**
     * @brief Flatten Layer converting N-D (e.g. 4D image) Tensors into 2D matrices for Dense layers.
     *
     * Reshapes and transposes 4D tensor \( [N, C, H, W] \) into 2D tensor \( [C \cdot H \cdot W, N] \)
     * matching Dense layer input convention \( (D_{in}, N) \).
     */
    class Flatten : public Layer
    {
    public:
        Flatten() = default;

        Tensor Forward(const Tensor& input) override;
    };
}