/**
 * @file test_tensor_runner.cpp
 * @brief Comprehensive C++ Test Runner designed for 100% Function Coverage across all headers & sources.
 *
 * Mathematical Formulations:
 * 1. Chained Computational Graph Autograd (Chain Rule):
 *    \[ Z = Y \cdot A = (A + B) \cdot A \implies \frac{\partial Z}{\partial A} = \frac{\partial Z}{\partial Y} \frac{\partial Y}{\partial A} + \frac{\partial Z}{\partial A}_{direct} \]
 * 2. Concat & Slice Backward Passes:
 *    \[ \frac{\partial L}{\partial X_i} = \text{slice}\left(\frac{\partial L}{\partial Y}, \text{axis}, \text{start}_i, \text{end}_i\right) \]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <iomanip>
#include <stdexcept>

#include "Tensor.hpp"
#include "Autograd.hpp"

namespace
{

void printJsonVector(std::ostream &os, const std::vector<double> &vec)
{
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        os << std::setprecision(15) << vec[i];
        if (i + 1 < vec.size())
        {
            os << ", ";
        }
    }
    os << "]";
}

void printJsonShape(std::ostream &os, const tardigrade::Shape &shape)
{
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        os << shape[i];
        if (i + 1 < shape.size())
        {
            os << ", ";
        }
    }
    os << "]";
}

void printTensorResult(const tardigrade::Tensor &tensor, const std::string &opName)
{
    std::cout << "{\n";
    std::cout << "  \"op\": \"" << opName << "\",\n";
    std::cout << "  \"rank\": " << tensor.rank() << ",\n";
    std::cout << "  \"size\": " << tensor.size() << ",\n";
    std::cout << "  \"shape\": ";
    printJsonShape(std::cout, tensor.shape());
    std::cout << ",\n  \"strides\": ";
    printJsonShape(std::cout, tensor.strides());
    std::cout << ",\n  \"data\": ";
    std::vector<double> dataVec(tensor.data(), tensor.data() + tensor.size());
    printJsonVector(std::cout, dataVec);

    if (tensor.requiresGrad() && tensor.grad().m_impl != nullptr)
    {
        std::cout << ",\n  \"grad\": ";
        const auto &g = tensor.grad();
        std::vector<double> gradVec(g.data(), g.data() + g.size());
        printJsonVector(std::cout, gradVec);
    }

    std::cout << "\n}\n";
}

tardigrade::Tensor readTensorFromCin(bool requiresGrad = false)
{
    int rank = 0;
    if (!(std::cin >> rank))
    {
        throw std::runtime_error("Failed to read tensor rank from std::cin");
    }
    tardigrade::Shape shape(rank);
    size_t totalElements = 1;
    for (int i = 0; i < rank; ++i)
    {
        std::cin >> shape[i];
        totalElements *= shape[i];
    }
    tardigrade::Tensor t(shape, requiresGrad);
    for (size_t i = 0; i < totalElements; ++i)
    {
        std::cin >> t.data()[i];
    }
    return t;
}

} // un-named namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <op_name> [args...]" << std::endl;
        return 1;
    }

    std::string opName = argv[1];

    try
    {
        if (opName == "zeros")
        {
            tardigrade::Shape shape;
            for (int i = 2; i < argc; ++i)
            {
                shape.push_back(std::stoi(argv[i]));
            }
            tardigrade::Tensor t = tardigrade::Tensor::zeros(shape);
            printTensorResult(t, opName);
        }
        else if (opName == "ones")
        {
            tardigrade::Shape shape;
            for (int i = 2; i < argc; ++i)
            {
                shape.push_back(std::stoi(argv[i]));
            }
            tardigrade::Tensor t = tardigrade::Tensor::ones(shape);
            printTensorResult(t, opName);
        }
        else if (opName == "fill")
        {
            double val = std::stod(argv[2]);
            tardigrade::Shape shape;
            for (int i = 3; i < argc; ++i)
            {
                shape.push_back(std::stoi(argv[i]));
            }
            tardigrade::Tensor t = tardigrade::Tensor::fill(shape, val);
            printTensorResult(t, opName);
        }
        else if (opName == "item")
        {
            tardigrade::Tensor t = readTensorFromCin();
            double val = t.item();
            double doubleVal = static_cast<double>(t);
            std::cout << "{\n  \"item\": " << val << ",\n  \"doubleVal\": " << doubleVal << "\n}\n";
        }
        else if (opName == "member_methods")
        {
            tardigrade::Tensor A = readTensorFromCin(true);
            tardigrade::Tensor clonedA = A.clone();
            
            tardigrade::Tensor gradTensor = tardigrade::Tensor::fill(A.shape(), 0.5);
            A.setGrad(gradTensor);

            int d0 = (A.rank() > 0) ? A.dim(0) : 0;
            A.zeroGrad();

            // Exercise Storage default constructor & const getters & const indexing
            tardigrade::Storage defaultStorage;
            const tardigrade::Storage constStorage(5);
            const double *constData = constStorage.GetData();
            double val0 = constStorage[0];

            // Exercise const Tensor variadic operator() and operator[]
            const tardigrade::Tensor constA = A;
            double constVal = constA(0, 0);
            tardigrade::Tensor subSquare = constA[0];

            // 1D transpose & free function transpose
            tardigrade::Tensor tens1d({5});
            tardigrade::Tensor trans1d = tens1d.transpose();
            tardigrade::Tensor freeTrans = tardigrade::transpose(A);

            // Exercise ClearGraph()
            A.ClearGraph();

            printTensorResult(clonedA, opName);
        }
        else if (opName == "inplace_ops")
        {
            tardigrade::Tensor A = readTensorFromCin(false);
            tardigrade::Tensor B = readTensorFromCin(false);
            A += B;
            A -= B;
            printTensorResult(A, opName);
        }
        else if (opName == "select_setselect")
        {
            tardigrade::Tensor A = readTensorFromCin(false);
            int dim = std::stoi(argv[2]);
            int index = std::stoi(argv[3]);
            tardigrade::Tensor sub = A.select(dim, index);

            tardigrade::Tensor src = tardigrade::Tensor::fill(sub.shape(), 9.9);
            A.setSelect(dim, index, src);

            tardigrade::Tensor vec1d({3});
            vec1d.fill(5.0);
            tardigrade::Tensor scalarSub = vec1d.select(0, 1);

            tardigrade::Tensor tens3d({2, 3, 4});
            tens3d.fill(1.0);
            tardigrade::Tensor sub3d = tens3d.select(1, 1);
            tens3d.setSelect(1, 1, sub3d);

            printTensorResult(A, opName);
        }
        else if (opName == "slice_setslice")
        {
            tardigrade::Tensor A = readTensorFromCin(false);
            int dim = std::stoi(argv[2]);
            int start = std::stoi(argv[3]);
            int end = std::stoi(argv[4]);

            tardigrade::Tensor slc = A.slice(dim, start, end);
            tardigrade::Tensor src = tardigrade::Tensor::fill(slc.shape(), 7.7);
            A.setSlice(dim, start, end, src);

            // Free function slice with requiresGrad=true
            tardigrade::Tensor mat2d = tardigrade::Tensor::ones({4, 4}, true);
            tardigrade::Tensor freeSlc = tardigrade::slice(mat2d, 1, 3);

            // Directly exercise SliceNode Backward
            if (freeSlc.gradNode() != nullptr)
            {
                std::vector<tardigrade::Tensor> gradOutputs = { tardigrade::Tensor::ones(freeSlc.shape()) };
                freeSlc.gradNode()->Backward(gradOutputs);
            }

            printTensorResult(A, opName);
        }
        else if (opName == "add" || opName == "sub" || opName == "mul" || opName == "div")
        {
            tardigrade::Tensor A = readTensorFromCin(true);
            tardigrade::Tensor B = readTensorFromCin(true);

            tardigrade::Tensor C;
            if (opName == "add")
            {
                C = A + B;
            }
            else if (opName == "sub")
            {
                C = A - B;
            }
            else if (opName == "mul")
            {
                C = A * B;
            }
            else if (opName == "div")
            {
                C = A / B;
            }

            tardigrade::Tensor loss = tardigrade::sum(C);
            loss.Backward();

            std::cout << "{\n";
            std::cout << "  \"op\": \"" << opName << "\",\n";
            std::cout << "  \"shape\": ";
            printJsonShape(std::cout, C.shape());
            std::cout << ",\n  \"data\": ";
            std::vector<double> cData(C.data(), C.data() + C.size());
            printJsonVector(std::cout, cData);
            std::cout << ",\n  \"gradA\": ";
            std::vector<double> gA(A.grad().data(), A.grad().data() + A.grad().size());
            printJsonVector(std::cout, gA);
            std::cout << ",\n  \"gradB\": ";
            std::vector<double> gB(B.grad().data(), B.grad().data() + B.grad().size());
            printJsonVector(std::cout, gB);
            std::cout << "\n}\n";
        }
        else if (opName == "matmul")
        {
            tardigrade::Tensor A = readTensorFromCin(true);
            tardigrade::Tensor B = readTensorFromCin(true);

            tardigrade::Tensor C = A % B;
            tardigrade::Tensor loss = tardigrade::sum(C);
            loss.Backward();

            std::cout << "{\n";
            std::cout << "  \"op\": \"matmul\",\n";
            std::cout << "  \"shape\": ";
            printJsonShape(std::cout, C.shape());
            std::cout << ",\n  \"data\": ";
            std::vector<double> cData(C.data(), C.data() + C.size());
            printJsonVector(std::cout, cData);
            std::cout << ",\n  \"gradA\": ";
            std::vector<double> gA(A.grad().data(), A.grad().data() + A.grad().size());
            printJsonVector(std::cout, gA);
            std::cout << ",\n  \"gradB\": ";
            std::vector<double> gB(B.grad().data(), B.grad().data() + B.grad().size());
            printJsonVector(std::cout, gB);
            std::cout << "\n}\n";
        }
        else if (opName == "exp" || opName == "log")
        {
            tardigrade::Tensor A = readTensorFromCin(true);
            tardigrade::Tensor C = (opName == "exp") ? tardigrade::exp(A) : tardigrade::log(A);

            tardigrade::Tensor loss = tardigrade::sum(C);
            loss.Backward();

            std::cout << "{\n";
            std::cout << "  \"op\": \"" << opName << "\",\n";
            std::cout << "  \"shape\": ";
            printJsonShape(std::cout, C.shape());
            std::cout << ",\n  \"data\": ";
            std::vector<double> cData(C.data(), C.data() + C.size());
            printJsonVector(std::cout, cData);
            std::cout << ",\n  \"gradA\": ";
            std::vector<double> gA(A.grad().data(), A.grad().data() + A.grad().size());
            printJsonVector(std::cout, gA);
            std::cout << "\n}\n";
        }
        else if (opName == "reshape" || opName == "permute" || opName == "transpose")
        {
            tardigrade::Tensor A = readTensorFromCin(false);
            tardigrade::Tensor C;

            if (opName == "reshape")
            {
                int newRank = 0;
                std::cin >> newRank;
                tardigrade::Shape newShape(newRank);
                for (int i = 0; i < newRank; ++i)
                {
                    std::cin >> newShape[i];
                }
                C = A.reshape(newShape);
            }
            else if (opName == "permute")
            {
                int dimsCount = 0;
                std::cin >> dimsCount;
                std::vector<int> dims(dimsCount);
                for (int i = 0; i < dimsCount; ++i)
                {
                    std::cin >> dims[i];
                }
                C = tardigrade::permute(A, dims);
            }
            else if (opName == "transpose")
            {
                C = tardigrade::transpose(A);
            }

            printTensorResult(C, opName);
        }
        else if (opName == "chained_autograd_graph")
        {
            tardigrade::Tensor A = readTensorFromCin(true);
            tardigrade::Tensor B = readTensorFromCin(true);

            tardigrade::Tensor A1 = A + 0.1;
            tardigrade::Tensor B1 = B + 0.1;

            tardigrade::Tensor C = A1 + B1;
            tardigrade::Tensor D = C - A1;
            tardigrade::Tensor E = D * B1;
            tardigrade::Tensor F = E / A1;
            tardigrade::Tensor G = tardigrade::exp(F);
            tardigrade::Tensor H = tardigrade::log(G);

            tardigrade::Tensor img4d({1, 1, 4, 4}, true);
            tardigrade::Tensor img4d_chained = img4d + 0.0;
            tardigrade::Tensor col = tardigrade::im2col(img4d_chained, 2, 2, 1, 1, 0, 0);

            tardigrade::Tensor loss = tardigrade::sum(H, -1, true) + tardigrade::sum(col, -1, true);
            loss.Backward();

            printTensorResult(H, opName);
        }
        else if (opName == "broadcasting_branches")
        {
            tardigrade::Tensor A({2, 3});
            A.fill(2.0);
            tardigrade::Tensor B({1, 3});
            B.fill(3.0);
            tardigrade::Tensor C({2, 1});
            C.fill(4.0);

            bool isBcast1 = tardigrade::isBroadcastable(A.shape(), B.shape());
            bool isBcast2 = tardigrade::isBroadcastable(A.shape(), C.shape());
            bool isBcastFail = tardigrade::isBroadcastable({2, 3}, {4, 5});

            tardigrade::Tensor R1 = A + B;
            tardigrade::Tensor R2 = C + A;
            printTensorResult(R1 + R2, opName);
        }
        else if (opName == "scalar_ops")
        {
            tardigrade::Tensor A = readTensorFromCin(false);
            double scalar = 2.5;

            tardigrade::Tensor add1 = A + scalar;
            tardigrade::Tensor add2 = scalar + A;
            tardigrade::Tensor sub1 = A - scalar;
            tardigrade::Tensor sub2 = scalar - A;
            tardigrade::Tensor mul1 = A * scalar;
            tardigrade::Tensor mul2 = scalar * A;
            tardigrade::Tensor div1 = A / scalar;
            tardigrade::Tensor div2 = scalar / A;
            tardigrade::Tensor div3 = tardigrade::div(A, scalar);

            tardigrade::Tensor res = add1 + add2 + sub1 + sub2 + mul1 + mul2 + div1 + div2 + div3;
            printTensorResult(res, opName);
        }
        else if (opName == "comparison_ops")
        {
            tardigrade::Tensor A = readTensorFromCin(false);
            tardigrade::Tensor B = readTensorFromCin(false);
            double scalar = 1.0;

            tardigrade::Tensor eq1 = (A == B);
            tardigrade::Tensor eq2 = (A == scalar);
            tardigrade::Tensor eq3 = (scalar == B);

            tardigrade::Tensor neq1 = (A != B);
            tardigrade::Tensor neq2 = (A != scalar);
            tardigrade::Tensor neq3 = (scalar != B);

            tardigrade::Tensor lt1 = (A < B);
            tardigrade::Tensor lt2 = (A < scalar);
            tardigrade::Tensor lt3 = (scalar < B);

            tardigrade::Tensor gt1 = (A > B);
            tardigrade::Tensor gt2 = (A > scalar);
            tardigrade::Tensor gt3 = (scalar > B);

            tardigrade::Tensor le1 = (A <= B);
            tardigrade::Tensor le2 = (A <= scalar);
            tardigrade::Tensor le3 = (scalar <= B);

            tardigrade::Tensor ge1 = (A >= B);
            tardigrade::Tensor ge2 = (A >= scalar);
            tardigrade::Tensor ge3 = (scalar >= B);

            tardigrade::Tensor res = eq1 + eq2 + eq3 + neq1 + neq2 + neq3 + lt1 + lt2 + lt3 + gt1 + gt2 + gt3 + le1 + le2 + le3 + ge1 + ge2 + ge3;
            printTensorResult(res, opName);
        }
        else if (opName == "sum" || opName == "reduce_max")
        {
            tardigrade::Tensor A = readTensorFromCin(true);
            int axis = -1;
            std::cin >> axis;

            tardigrade::Tensor C = (opName == "sum") ? tardigrade::sum(A, axis, true) : tardigrade::reduce_max(A, axis, true);
            
            tardigrade::Tensor C_neg1_kd = tardigrade::sum(A, -1, true);

            tardigrade::Tensor loss = tardigrade::sum(C);
            loss.Backward();

            std::cout << "{\n";
            std::cout << "  \"op\": \"" << opName << "\",\n";
            std::cout << "  \"shape\": ";
            printJsonShape(std::cout, C.shape());
            std::cout << ",\n  \"data\": ";
            std::vector<double> cData(C.data(), C.data() + C.size());
            printJsonVector(std::cout, cData);
            std::cout << ",\n  \"gradA\": ";
            std::vector<double> gA(A.grad().data(), A.grad().data() + A.grad().size());
            printJsonVector(std::cout, gA);
            std::cout << "\n}\n";
        }
        else if (opName == "sum_multi_axes")
        {
            tardigrade::Tensor A = readTensorFromCin(false);
            std::vector<int> axes = {0, 1};
            tardigrade::Tensor C = tardigrade::sum(A, axes, false);
            printTensorResult(C, opName);
        }
        else if (opName == "concat")
        {
            tardigrade::Tensor A = readTensorFromCin(true);
            tardigrade::Tensor B = readTensorFromCin(true);
            int axis = std::stoi(argv[2]);
            
            tardigrade::Tensor C = tardigrade::concat({A, B}, axis);
            tardigrade::Tensor loss = tardigrade::sum(C);
            loss.Backward();

            // Directly exercise ConcatNode::Backward for 100% Function Coverage
            auto concatNode = std::make_shared<tardigrade::ConcatNode>();
            concatNode->m_inputs = {A, B};
            concatNode->m_axis = axis;
            std::vector<tardigrade::Tensor> gradOuts = { tardigrade::Tensor::ones(C.shape()) };
            concatNode->Backward(gradOuts);

            printTensorResult(C, opName);
        }


        else if (opName == "im2col_col2im")
        {
            tardigrade::Tensor X = readTensorFromCin(true);
            tardigrade::Tensor col = tardigrade::im2col(X, 1, 1, 1, 1, 0, 0);
            tardigrade::Tensor reconstructedX = tardigrade::col2im(col, X.shape(), 1, 1, 1, 1, 0, 0);
            printTensorResult(reconstructedX, opName);
        }
        else if (opName == "convolve")
        {
            tardigrade::Tensor X = readTensorFromCin(true);
            tardigrade::Tensor K = readTensorFromCin(true);
            int stride = 1, padding = 0;
            std::cin >> stride >> padding;

            tardigrade::Tensor Y = tardigrade::convolve(X, K, stride, padding);
            tardigrade::Tensor loss = tardigrade::sum(Y);
            loss.Backward();

            std::cout << "{\n";
            std::cout << "  \"op\": \"convolve\",\n";
            std::cout << "  \"shape\": ";
            printJsonShape(std::cout, Y.shape());
            std::cout << ",\n  \"data\": ";
            std::vector<double> yData(Y.data(), Y.data() + Y.size());
            printJsonVector(std::cout, yData);
            std::cout << ",\n  \"gradX\": ";
            std::vector<double> gX(X.grad().data(), X.grad().data() + X.grad().size());
            printJsonVector(std::cout, gX);
            std::cout << ",\n  \"gradK\": ";
            std::vector<double> gK(K.grad().data(), K.grad().data() + K.grad().size());
            printJsonVector(std::cout, gK);
            std::cout << "\n}\n";
        }
        else if (opName == "exceptions")
        {
            int caughtCount = 0;

            try { tardigrade::Tensor t({2, 3}); t.item(); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); t.reshape({5, 5}); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); t.select(0, 10); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); tardigrade::Tensor src({3}); t.setSelect(0, 10, src); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); t.slice(0, 5, 2); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); tardigrade::Tensor src({1, 3}); t.setSlice(0, 5, 2, src); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); t.permute({0}); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor A({2, 3}); tardigrade::Tensor B({4, 5}); A += B; } catch (...) { caughtCount++; }
            try { tardigrade::Tensor A({2, 3}); tardigrade::Tensor B({4, 5}); A -= B; } catch (...) { caughtCount++; }
            try { tardigrade::Tensor A({2, 3}); tardigrade::Tensor B({4, 5}); A.matmul(B); } catch (...) { caughtCount++; }
            try { tardigrade::broadcastShapes({2, 3}, {4, 5}); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor::normalizeAxis(10, 3); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); t.calculateIndex({1, 2, 3}); } catch (...) { caughtCount++; }
            try { std::vector<tardigrade::Tensor> emptyVec; tardigrade::concat(emptyVec, 0); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor A({2, 3}); tardigrade::Tensor B({2, 3, 4}); tardigrade::concat({A, B}, 0); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor t({2, 3}); tardigrade::im2col(t, 3, 3, 1, 1, 0, 0); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor col({9, 25}); tardigrade::col2im(col, {5, 5}, 3, 3, 1, 1, 0, 0); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor X({2, 3}); tardigrade::Tensor K({1, 1, 3, 3}); tardigrade::convolve(X, K); } catch (...) { caughtCount++; }
            try { tardigrade::Tensor X({1, 2, 5, 5}); tardigrade::Tensor K({1, 3, 3, 3}); tardigrade::convolve(X, K); } catch (...) { caughtCount++; }

            std::cout << "{\n  \"exceptions_tested\": " << caughtCount << "\n}\n";
        }
        else
        {
            std::cerr << "Unknown operation: " << opName << std::endl;
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception encountered in C++ runner: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
