cmake_minimum_required(VERSION 3.10)
project(TardigradeUnitTest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Coverage flags (Clang / GCC)
option(ENABLE_COVERAGE "Enable code coverage instrumentation" ON)
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
        add_link_options(-fprofile-instr-generate -fcoverage-mapping)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(--coverage)
        add_link_options(--coverage)
    endif()
endif()

# Core tardigrade sources needed for Tensor testing
set(TARDIGRADE_SOURCES
    ${CMAKE_SOURCE_DIR}/Tardigrade/tensor/Tensor.cpp
    ${CMAKE_SOURCE_DIR}/Tardigrade/tensor/Autograd.cpp
)

add_executable(test_tensor_runner
    test_tensor_runner.cpp
    ${TARDIGRADE_SOURCES}
)

target_include_directories(test_tensor_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/Tardigrade
    ${CMAKE_SOURCE_DIR}/Tardigrade/tensor
    ${CMAKE_SOURCE_DIR}/Tardigrade/third_party
)
