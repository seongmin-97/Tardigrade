#pragma once
#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tardigrade
{
	template <int N>
	using Tensor = Eigen::Tensor<double, N>;
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;
	using Tensor3D = Tensor<3>;
	using Tensor4D = Tensor<4>;
}

namespace tardigrade::data
{
	using Dataset = std::vector<std::unique_ptr< Eigen::MatrixXd>>;
	using Labelset = std::vector<Eigen::MatrixXd>;

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