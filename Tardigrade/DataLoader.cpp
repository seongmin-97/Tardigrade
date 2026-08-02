#include "DataLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb/stb_image_resize2.h"

using namespace tardigrade;
using namespace tardigrade::data;

// ------------------------------------------------------------
// ReadImage: Reads image file and returns a normalized Tensor [0.0, 1.0]
// ------------------------------------------------------------
Tensor DataLoader::ReadImage(const std::string &path, MatSize target, ImageReadMode mode)
{
    int reqChannels = static_cast<int>(mode);
    int width = 0;
    int height = 0;
    int actualChannels = 0;

    unsigned char *imgData = stbi_load(path.c_str(), &width, &height, &actualChannels, reqChannels);
    if (!imgData)
    {
        throw std::runtime_error("Cannot read image: " + path + " (stb_image failure)");
    }

    int outWidth = (target.col > 0) ? target.col : width;
    int outHeight = (target.row > 0) ? target.row : height;

    std::vector<unsigned char> resizedBuffer;
    unsigned char *finalPixels = imgData;

    if (outWidth != width || outHeight != height)
    {
        resizedBuffer.resize(outWidth * outHeight * reqChannels);
        unsigned char *res = stbir_resize_uint8_linear(
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
    Tensor result({totalPixels, 1});
    double *rawPtr = result.data();

    for (int i = 0; i < totalPixels; ++i)
    {
        rawPtr[i] = static_cast<double>(finalPixels[i]) / 255.0;
    }

    stbi_image_free(imgData);
    return result;
}

// ------------------------------------------------------------
// Factory Method: FromImageFolder
// ------------------------------------------------------------
DataLoader DataLoader::FromImageFolder(const std::string &rootDir,
                                       MatSize target,
                                       ImageReadMode mode,
                                       LoadStrategy strategy)
{
    DataLoader loader;
    loader.m_strategy = strategy;
    loader.LoadImageDataset(rootDir, target, mode);
    return loader;
}

// ------------------------------------------------------------
// Factory Method: FromTensor
// ------------------------------------------------------------
DataLoader DataLoader::FromTensor(const Tensor &inputs, const Tensor &targets)
{
    size_t numSamples = 0;
    bool selectAlongCol = true;

    if (inputs.rank() >= 2)
    {
        if (inputs.dim(1) > 0)
        {
            numSamples = static_cast<size_t>(inputs.dim(1));
            selectAlongCol = true;
        }
        else
        {
            numSamples = static_cast<size_t>(inputs.dim(0));
            selectAlongCol = false;
        }
    }
    else
    {
        numSamples = inputs.size();
    }

    auto fetcher = [inputs, targets, selectAlongCol](size_t index) -> std::pair<Tensor, Tensor>
    {
        Tensor feat;
        if (inputs.rank() >= 2)
        {
            feat = selectAlongCol ? inputs.select(1, static_cast<int>(index))
                                  : inputs.select(0, static_cast<int>(index));
        }
        else
        {
            feat = Tensor({1, 1});
            feat.data()[0] = inputs.data()[index];
        }

        Tensor tgt;
        if (targets.rank() >= 2)
        {
            tgt = (targets.dim(1) > static_cast<int>(index)) ? targets.select(1, static_cast<int>(index))
                                                             : targets.select(0, static_cast<int>(index));
        }
        else if (targets.size() > index)
        {
            tgt = Tensor({1, 1});
            tgt.data()[0] = targets.data()[index];
        }
        else
        {
            tgt = Tensor({1, 1});
            tgt.data()[0] = 0.0;
        }

        return {feat, tgt};
    };

    return FromCustom(numSamples, fetcher);
}

// ------------------------------------------------------------
// Factory Method: FromCustom
// ------------------------------------------------------------
DataLoader DataLoader::FromCustom(size_t totalSamples,
                                 std::function<std::pair<Tensor, Tensor>(size_t index)> sampleFetcher)
{
    DataLoader loader;
    loader.m_dataSize = totalSamples;
    loader.m_sampleFetcher = std::move(sampleFetcher);
    loader.m_indices.resize(totalSamples);
    std::iota(loader.m_indices.begin(), loader.m_indices.end(), 0);
    return loader;
}

// ------------------------------------------------------------
// LoadImageDataset: Loads image dataset from directory structure
// ------------------------------------------------------------
void DataLoader::LoadImageDataset(const std::string &rootDir, MatSize target, ImageReadMode mode)
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

        for (const auto &entry : fs::directory_iterator(labelDir))
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
                catch (const std::exception &e)
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

    m_dataSize = m_labels.size();
    m_indices.resize(m_dataSize);
    std::iota(m_indices.begin(), m_indices.end(), 0);

    m_sampleFetcher = [this](size_t index) -> std::pair<Tensor, Tensor>
    {
        Tensor feat;
        if (m_strategy == LoadStrategy::EAGER)
        {
            feat = m_data[index];
        }
        else
        {
            feat = ReadImage(m_paths[index], m_targetSize, m_readMode);
        }

        Tensor tgt({1, 1});
        tgt.data()[0] = static_cast<double>(m_labels[index]);

        return {feat, tgt};
    };

    std::cout << "[INFO] Image dataset loaded: " << GetDataSize() << " samples"
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
    return m_dataSize;
}

std::pair<Tensor, Tensor> DataLoader::GetSample(size_t index) const
{
    if (index >= m_dataSize)
    {
        throw std::out_of_range("DataLoader: index out of range");
    }
    if (!m_sampleFetcher)
    {
        throw std::runtime_error("DataLoader: sample fetcher is not initialized");
    }

    size_t permutedIdx = m_indices[index];
    return m_sampleFetcher(permutedIdx);
}

Tensor DataLoader::GetData(size_t index) const
{
    return GetSample(index).first;
}

int DataLoader::GetLabel(size_t index) const
{
    Tensor tgt = GetSample(index).second;
    return static_cast<int>(tgt.data()[0]);
}

// ------------------------------------------------------------
// Batch Retrieval
// ------------------------------------------------------------
Tensor DataLoader::GetBatch(size_t startIdx, size_t batchSize) const
{
    size_t targetBatchSize = (batchSize > 0) ? batchSize : m_batchSize;

    if (startIdx >= m_dataSize || targetBatchSize == 0)
    {
        throw std::runtime_error("DataLoader: invalid batch parameters");
    }

    size_t actualSize = std::min(targetBatchSize, m_dataSize - startIdx);
    Tensor first = GetData(startIdx);
    int featureSize = static_cast<int>(first.size());

    Tensor batch({featureSize, static_cast<int>(actualSize)});
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

    if (startIdx >= m_dataSize || targetBatchSize == 0)
    {
        throw std::runtime_error("DataLoader: invalid label batch parameters");
    }

    size_t actualSize = std::min(targetBatchSize, m_dataSize - startIdx);
    Tensor firstTarget = GetSample(startIdx).second;
    int targetSize = static_cast<int>(firstTarget.size());

    Tensor batchTarget;
    if (targetSize == 1)
    {
        batchTarget = Tensor({1, static_cast<int>(actualSize)});
        batchTarget.data()[0] = firstTarget.data()[0];

        for (size_t i = 1; i < actualSize; ++i)
        {
            batchTarget.data()[i] = GetSample(startIdx + i).second.data()[0];
        }
    }
    else
    {
        batchTarget = Tensor({targetSize, static_cast<int>(actualSize)});
        batchTarget.setSelect(1, 0, firstTarget);

        for (size_t i = 1; i < actualSize; ++i)
        {
            Tensor sampleTarget = GetSample(startIdx + i).second;
            batchTarget.setSelect(1, static_cast<int>(i), sampleTarget);
        }
    }

    return batchTarget;
}

std::pair<Tensor, Tensor> DataLoader::GetBatchPair(size_t startIdx, size_t batchSize) const
{
    return {GetBatch(startIdx, batchSize), GetLabelBatch(startIdx, batchSize)};
}

// ------------------------------------------------------------
// Shuffle: Efficient O(N) index permutation shuffle
// ------------------------------------------------------------
void DataLoader::Shuffle(std::mt19937 &rng)
{
    if (m_indices.size() <= 1)
    {
        return;
    }
    std::shuffle(m_indices.begin(), m_indices.end(), rng);
}