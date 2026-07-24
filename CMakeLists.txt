cmake_minimum_required(VERSION 3.10)
project(Tardigrade CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Source files
set(SOURCES
    Tardigrade/main.cpp
    Tardigrade/tensor/Tensor.cpp
    Tardigrade/tensor/Autograd.cpp
    Tardigrade/Activation.cpp
    Tardigrade/DataLoader.cpp
    Tardigrade/Layer.cpp
    Tardigrade/Loss.cpp
    Tardigrade/Model.cpp
    Tardigrade/Optimizer.cpp
    Tardigrade/Metric.cpp
)

# Header files
set(HEADERS
    Tardigrade/tensor/Tensor.hpp
    Tardigrade/tensor/Autograd.hpp
    Tardigrade/Activation.hpp
    Tardigrade/DataLoader.hpp
    Tardigrade/Layer.hpp
    Tardigrade/Loss.hpp
    Tardigrade/Model.hpp
    Tardigrade/Optimizer.hpp
    Tardigrade/Metric.hpp
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Tardigrade
    ${CMAKE_CURRENT_SOURCE_DIR}/Tardigrade/tensor
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
)
