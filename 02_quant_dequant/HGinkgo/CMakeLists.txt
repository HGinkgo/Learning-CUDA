cmake_minimum_required(VERSION 3.18)
project(low_precision_reference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;86)
endif()

add_library(low_precision_reference quant.cpp)
target_include_directories(low_precision_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_cpu test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE low_precision_reference)

add_executable(test_precision_types test_precision_types.cpp)
target_link_libraries(test_precision_types PRIVATE low_precision_reference)

add_library(low_precision_cuda cuda_quant.cu)
target_include_directories(low_precision_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(low_precision_cuda PRIVATE low_precision_reference)
set_target_properties(low_precision_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_cuda test_cuda.cu)
target_link_libraries(test_cuda PRIVATE low_precision_cuda low_precision_reference)
set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(benchmark benchmark.cu)
target_link_libraries(benchmark PRIVATE low_precision_cuda low_precision_reference)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_test(NAME cpu_reference COMMAND test_cpu)
add_test(NAME precision_types_cpu COMMAND test_precision_types)
add_test(NAME cuda_reference COMMAND test_cuda)
add_test(NAME benchmark_output
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark_output.sh $<TARGET_FILE:benchmark>)
