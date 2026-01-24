#include "DataLoader.hpp"

namespace tardigrade::data 
{
    DataLoader::DataLoader(DataType dataType)
    {
        m_dataSize = 0;
        m_dataType = dataType;
    }

    Data DataLoader::ReadImage(const std::string& filePath, int flag, bool showImg)
    {
        cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

        Data tmpData = Eigen::MatrixXd(0, 0);
        
        if (img.empty()) 
        {
            std::cerr << "Error: Image not found at " << filePath << std::endl;
            return tmpData;
        }

        int rows = img.rows;
        int cols = img.cols;

        tmpData.resize(rows, cols);

        for (int r = 0; r < rows; ++r) 
            for (int c = 0; c < cols; ++c) 
                tmpData(r, c) = static_cast<double>(img.at<uchar>(r, c)) / 255.0;

        if (showImg)
        {
            cv::namedWindow("Debug: Loaded Image", cv::WINDOW_AUTOSIZE);
            cv::imshow("Debug: Loaded Image", img);
            std::cout << "[Debug] Visualization active. Press any key to continue..." << std::endl;
            cv::waitKey(0); 
            cv::destroyWindow("Debug: Loaded Image");
        }

        return tmpData;
    }

    void DataLoader::ReadDataset(const std::string& dirPath, int flag, bool dirIsLabel)
    {
        try 
        {
            if (!fs::exists(dirPath) || !fs::is_directory(dirPath))
            {
                std::cerr << "Invalid path: " << dirPath << std::endl;
                return;
            }

            int startIdx = m_dataSize;

            std::vector<fs::path> paths;
            for (const auto& entry : fs::directory_iterator(dirPath)) 
            {
                if (fs::is_regular_file(entry) && IMAGE_EXTENSIONS.count(entry.path().extension().string())) 
                {
                    paths.push_back(entry.path());
                }
            }

            if (paths.empty())
                return;

            m_dataset.resize(startIdx + paths.size());
            std::vector<std::future<void>> futures;

            for (size_t i = 0; i < paths.size(); i++)
            {
                futures.push_back(std::async(std::launch::async, [this, i, startIdx, &paths]()
                    {
                        Data imgData = ReadImage(paths[i].string(), 0, false);
                        m_dataset[startIdx + i] = std::make_unique<Data>(std::move(imgData));
                    }));
            }

            for (auto& f : futures) 
                f.wait();

            m_dataSize += paths.size();

            if (dirIsLabel)
            {
                m_labelset.resize(m_dataSize);

                double value;
                fs::path p(dirPath);

                if (p.has_filename())
                {
                    value = std::stod(p.filename().string());
                }
                else
                {
                    value = std::stod(p.parent_path().filename().string());
                }

                for (int i = startIdx; i < m_dataSize; i++)
                {
                    Label label = Label(1, 1);
                    label(0, 0) = value;
                    m_labelset[i] = label;
                }
            }
        }
        catch (const fs::filesystem_error& e) 
        {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
    }

    void DataLoader::ReadLabelset(const std::string& dirPath, MatSize size)
    {
        try
        {
            if (!fs::exists(dirPath) || !fs::is_directory(dirPath))
            {
                std::cerr << "Invalid path: " << dirPath << std::endl;
            }

        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
    }

    int DataLoader::GetDataSize()
    {
        return m_dataSize;
    }
}