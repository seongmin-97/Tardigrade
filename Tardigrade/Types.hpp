#pragma once
#include <vector>
#include <memory>

#include <Eigen/Dense>

namespace tardigrade::data
{
	typedef Eigen::MatrixXd Data;
	typedef Eigen::MatrixXd Label;
	
	typedef std::vector<std::unique_ptr<Data>> Dataset;
	typedef std::vector<Label> Labelset;

	enum class DataType
	{
		IMAGE
	};

	struct MatSize
	{
		int row;
		int col;
	};
}
