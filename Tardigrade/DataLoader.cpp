#include "DataLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb/stb_image_resize2.h"

using namespace tardigrade;
using namespace tardigrade::data;

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
DataLoader::DataLoader(LoadStrategy strategy)
    : m_strategy(strategy),
      m_targetSize({0, 0}),
      m_readMode(ImageReadMode::GRAYSCALE)
{
}

// ------------------------------------------------------------
// ReadImage: Reads image file and returns a normalized Tensor [0.0, 1.0]
// ------------------------------------------------------------
Tensor DataLoader::ReadImage(const std::string& path, MatSize target, ImageReadMode mode) const
{
    int reqChannels = static_cast<int>(mode);
    int width = 0;
    int height = 0;
    int actualChannels = 0;

    unsigned char* imgData = stbi_load(path.c_str(), &width, &height, &actualChannels, reqChannels);
    if (!imgData)
    {
        throw std::runtime_error("Cannot read image: " + path + " (stb_image failure)");
    }

    int outWidth = (target.col > 0) ? target.col : width;
    int outHeight = (target.row > 0) ? target.row : height;

    std::vector<unsigned char> resizedBuffer;
    unsigned char* finalPixels = imgData;

    if (outWidth != width || outHeight != height)
    {
        resizedBuffer.resize(outWidth * outHeight * reqChannels);
        unsigned char* res = stbir_resize_uint8_linear(
            imgData, width, height, 0,
            resizedBuffer.data(), outWidth, outHeight, 0,
            static_cast<stbir_pixel_layout>(reqChannels)
        );

        if (!res)
        {
            stbi_image_free(imgData);
            throw std::runtime_error("Failed to resize image: " + path);
        }
        finalPixels = resizedBuffer.data();
    }

    int totalPixels = outHeight * outWidth * reqChannels;
    Tensor result({ totalPixels, 1 });
    double* rawPtr = result.data();

    for (int i = 0; i < totalPixels; ++i)
    {
        rawPtr[i] = static_cast<double>(finalPixels[i]) / 255.0;
    }

    stbi_image_free(imgData);
    return result;
}

// ------------------------------------------------------------
// LoadImageDataset: Loads image dataset from directory structure
//
// Assumes directory structure: rootDir/{0~9}/*.{jpg,png,...}
// Folder names are used as integer labels.
// ------------------------------------------------------------
void DataLoader::LoadImageDataset(const std::string& rootDir, MatSize target, ImageReadMode mode)
{
    m_targetSize = target;
    m_readMode = mode;

    m_data.clear();
    m_paths.clear();
    m_labels.clear();

    for (int label = 0; label <= 9; ++label)
    {
        fs::path labelDir = fs::path(rootDir) / std::to_string(label);

        if (!fs::exists(labelDir) || !fs::is_directory(labelDir))
        {
            std::cerr << "[WARNING] Label directory not found: " << labelDir << "\n";
            continue;
        }

        for (const auto& entry : fs::directory_iterator(labelDir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            const auto ext = entry.path().extension().string();
            if (IMAGE_EXTENSIONS.find(ext) == IMAGE_EXTENSIONS.end())
            {
                continue;
            }

            std::string filePath = entry.path().string();

            if (m_strategy == LoadStrategy::EAGER)
            {
                try
                {
                    m_data.push_back(ReadImage(filePath, target, mode));
                    m_labels.push_back(label);
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[WARNING] Failed to load: " << e.what() << "\n";
                }
            }
            else // LAZY
            {
                m_paths.push_back(filePath);
                m_labels.push_back(label);
            }
        }
    }

    std::cout << "[INFO] Dataset loaded: " << GetDataSize() << " samples"
              << " (strategy: " << (m_strategy == LoadStrategy::EAGER ? "EAGER" : "LAZY") << ")\n";
}

// ------------------------------------------------------------
// Getters & Setters
// ------------------------------------------------------------
void DataLoader::SetBatchSize(size_t batchSize)
{
    if (batchSize == 0)
    {
        throw std::invalid_argument("DataLoader: batchSize cannot be 0");
    }
    m_batchSize = batchSize;
}

size_t DataLoader::GetBatchSize() const
{
    return m_batchSize;
}

size_t DataLoader::GetDataSize() const
{
    return m_labels.size();
}

Tensor DataLoader::GetData(size_t index) const
{
    if (index >= GetDataSize())
    {
        throw std::runtime_error("DataLoader: index out of range");
    }

    if (m_strategy == LoadStrategy::EAGER)
    {
        return m_data[index];
    }
    else
    {
        return ReadImage(m_paths[index], m_targetSize, m_readMode);
    }
}

int DataLoader::GetLabel(size_t index) const
{
    if (index >= GetDataSize())
    {
        throw std::runtime_error("DataLoader: label index out of range");
    }

    return m_labels[index];
}

// ------------------------------------------------------------
// Batch Retrieval
// ------------------------------------------------------------
Tensor DataLoader::GetBatch(size_t startIdx, size_t batchSize) const
{
    size_t targetBatchSize = (batchSize > 0) ? batchSize : m_batchSize;

    if (startIdx >= GetDataSize() || targetBatchSize == 0)
    {
        throw std::runtime_error("DataLoader: invalid batch parameters");
    }

    size_t actualSize = std::min(targetBatchSize, GetDataSize() - startIdx);
    Tensor first = GetData(startIdx);
    int featureSize = static_cast<int>(first.size());

    Tensor batch({ featureSize, static_cast<int>(actualSize) });
    batch.setSelect(1, 0, first);

    for (size_t i = 1; i < actualSize; ++i)
    {
        Tensor sample = GetData(startIdx + i);
        batch.setSelect(1, static_cast<int>(i), sample);
    }

    return batch;
}

Tensor DataLoader::GetLabelBatch(size_t startIdx, size_t batchSize) const
{
    size_t targetBatchSize = (batchSize > 0) ? batchSize : m_batchSize;

    if (startIdx >= GetDataSize() || targetBatchSize == 0)
    {
        throw std::runtime_error("DataLoader: invalid label batch parameters");
    }

    size_t actualSize = std::min(targetBatchSize, GetDataSize() - startIdx);
    Tensor batchTarget({ 1, static_cast<int>(actualSize) });

    for (size_t i = 0; i < actualSize; ++i)
    {
        batchTarget.data()[i] = static_cast<double>(m_labels[startIdx + i]);
    }

    return batchTarget;
}

// ------------------------------------------------------------
// Shuffle: Shuffle index arrays (Works for Eager/Lazy)
// ------------------------------------------------------------
void DataLoader::Shuffle(std::mt19937& rng)
{
    size_t n = GetDataSize();
    if (n <= 1)
    {
        return;
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    if (m_strategy == LoadStrategy::EAGER)
    {
        std::vector<Tensor> shuffledData(n);
        std::vector<int> shuffledLabels(n);

        for (size_t i = 0; i < n; ++i)
        {
            shuffledData[i] = std::move(m_data[indices[i]]);
            shuffledLabels[i] = m_labels[indices[i]];
        }

        m_data = std::move(shuffledData);
        m_labels = std::move(shuffledLabels);
    }
    else // LAZY
    {
        std::vector<std::string> shuffledPaths(n);
        std::vector<int> shuffledLabels(n);

        for (size_t i = 0; i < n; ++i)
        {
            shuffledPaths[i] = std::move(m_paths[indices[i]]);
            shuffledLabels[i] = m_labels[indices[i]];
        }

        m_paths = std::move(shuffledPaths);
        m_labels = std::move(shuffledLabels);
    }
}