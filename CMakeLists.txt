cmake_minimum_required(VERSION 3.10)
project(Tardigrade CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
    Tardigrade/third_party/stb/stb_image.h
    Tardigrade/third_party/stb/stb_image_resize2.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Tardigrade
    ${CMAKE_CURRENT_SOURCE_DIR}/Tardigrade/tensor
    ${CMAKE_CURRENT_SOURCE_DIR}/Tardigrade/third_party
)
