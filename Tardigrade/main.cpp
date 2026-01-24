#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "DataLoader.hpp"

using namespace tardigrade::data;

int main() 
{
    std::string trainDir = "E:\\Dataset\\MNIST\\train\\";
    DataLoader dataLoader = DataLoader(DataType::IMAGE);

    for (int i = 0; i < 10; i++)
    {
        std::string path = trainDir + std::to_string(i);
        dataLoader.ReadDataset(path, cv::IMREAD_GRAYSCALE);
        std::cout << i << "완료 / 데이터 수 : " << dataLoader.GetDataSize() << std::endl;
    }

    return 0;
}