#include "DataLoader.hpp"

using namespace tardigrade;
using namespace tardigrade::data;

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
DataLoader::DataLoader(LoadStrategy strategy)
    : m_strategy(strategy),
      m_targetSize({0, 0}),
      m_readFlag(cv::IMREAD_GRAYSCALE)
{
}

// ------------------------------------------------------------
// ReadImage: Reads image file and returns a normalized Tensor [0.0, 1.0]
// ------------------------------------------------------------
Tensor DataLoader::ReadImage(const std::string& path, MatSize target, int flag) const
{
    cv::Mat img = cv::imread(path, flag);

    if (img.empty())
    {
        throw std::runtime_error("Cannot read image: " + path);
    }

    // 리사이즈 (target이 유효한 경우)
    if (target.row > 0 && target.col > 0)
    {
        cv::resize(img, img, cv::Size(target.col, target.row));
    }

    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();
    int totalPixels = rows * cols * channels;

    Tensor result({ totalPixels, 1 });
    double* rawPtr = result.data();

    if (channels == 1)
    {
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                *rawPtr++ = static_cast<double>(img.at<uchar>(r, c)) / 255.0;
            }
        }
    }
    else if (channels == 3)
    {
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
                *rawPtr++ = static_cast<double>(pixel[0]) / 255.0;
                *rawPtr++ = static_cast<double>(pixel[1]) / 255.0;
                *rawPtr++ = static_cast<double>(pixel[2]) / 255.0;
            }
        }
    }

    return result;
}

// ------------------------------------------------------------
// LoadImageDataset: Loads image dataset from directory structure
//
// Assumes directory structure: rootDir/{0~9}/*.{jpg,png,...}
// Folder names are used as integer labels.
// ------------------------------------------------------------
void DataLoader::LoadImageDataset(const std::string& rootDir, MatSize target, int flag)
{
    m_targetSize = target;
    m_readFlag = flag;

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
                    m_data.push_back(ReadImage(filePath, target, flag));
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
// Getters
// ------------------------------------------------------------
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
        return ReadImage(m_paths[index], m_targetSize, m_readFlag);
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
    if (startIdx >= GetDataSize() || batchSize == 0)
    {
        throw std::runtime_error("DataLoader: invalid batch parameters");
    }

    size_t actualSize = std::min(batchSize, GetDataSize() - startIdx);
    Tensor first = GetData(startIdx);
    int featureSize = static_cast<int>(first.size());

    Tensor batch({ featureSize, static_cast<int>(actualSize) });
    batch.asMatrix(featureSize, static_cast<int>(actualSize)).col(0) = first.asVector();

    for (size_t i = 1; i < actualSize; ++i)
    {
        Tensor sample = GetData(startIdx + i);
        batch.asMatrix(featureSize, static_cast<int>(actualSize)).col(static_cast<int>(i)) = sample.asVector();
    }

    return batch;
}

std::vector<int> DataLoader::GetLabelBatch(size_t startIdx, size_t batchSize) const
{
    if (startIdx >= GetDataSize() || batchSize == 0)
    {
        throw std::runtime_error("DataLoader: invalid label batch parameters");
    }

    size_t actualSize = std::min(batchSize, GetDataSize() - startIdx);
    return std::vector<int>(
        m_labels.begin() + static_cast<long>(startIdx),
        m_labels.begin() + static_cast<long>(startIdx + actualSize)
    );
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