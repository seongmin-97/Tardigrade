#!/bin/bash
# Create build directory
mkdir -p build
# Enter build directory
cd build
# Configure project with CMake
cmake ..
# Build the project
cmake --build .
