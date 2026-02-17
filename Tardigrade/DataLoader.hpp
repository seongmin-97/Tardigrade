#pragma once
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <algorithm>
#include <random>
#include <numeric>
#include <filesystem>
#include <unordered_set>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "Tensor.hpp"

namespace tardigrade::data
{
	using Dataset = std::vector<std::unique_ptr<Tensor>>;
	using Labelset = std::vector<Tensor>;

	enum class DataType
	{
		IMAGE
	};

	struct MatSize
	{
		int row;
		int col;
	};

	namespace fs = std::filesystem;

	static inline const std::unordered_set<std::string> IMAGE_EXTENSIONS = { ".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG" };

	class DataLoader
	{
	public :
		DataLoader(DataType dataType);
		Tensor ReadImage(const std::string& filePath, int flag = cv::IMREAD_COLOR_RGB, bool showImg = false);
		void ReadDataset(const std::string& dirPath, int flag = cv::IMREAD_COLOR_RGB, bool dirIsLabel = true);
		void ReadLabelset(const std::string& dirPath, MatSize size = { 1, 1 });

		int GetDataSize();
		void Shuffle();

	private :
		int m_dataSize;

		std::unordered_map<std::string, int> m_idx2path; //dataIdx to dataPath
		Dataset m_dataset;
		Labelset m_labelset;
		DataType m_dataType;
	};
}