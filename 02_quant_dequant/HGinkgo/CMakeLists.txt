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

set(LP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LP_NVIDIA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backends/nvidia/include)

add_library(low_precision_reference common/quant.cpp)
target_include_directories(low_precision_reference PUBLIC ${LP_INCLUDE_DIR})

add_library(low_precision_io common/io.cpp)
target_include_directories(low_precision_io PUBLIC ${LP_INCLUDE_DIR})
target_link_libraries(low_precision_io PUBLIC low_precision_reference)

add_executable(test_cpu tests/test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE low_precision_reference)

add_executable(test_precision_types tests/test_precision_types.cpp)
target_link_libraries(test_precision_types PRIVATE low_precision_reference)

add_executable(test_io tests/test_io.cpp)
target_link_libraries(test_io PRIVATE low_precision_io)

add_library(low_precision_cuda backends/nvidia/cuda_quant.cu)
target_include_directories(low_precision_cuda PUBLIC ${LP_INCLUDE_DIR} ${LP_NVIDIA_INCLUDE_DIR})
target_link_libraries(low_precision_cuda PRIVATE low_precision_reference)
set_target_properties(low_precision_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_cuda tests/test_cuda.cu)
target_link_libraries(test_cuda PRIVATE low_precision_cuda low_precision_reference)
set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(benchmark apps/benchmark.cu)
target_link_libraries(benchmark PRIVATE low_precision_cuda low_precision_reference)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(quant_cli apps/quant_cli.cpp)
target_link_libraries(quant_cli PRIVATE low_precision_io low_precision_reference)

add_executable(cuda_quant_cli apps/cuda_quant_cli.cu)
target_link_libraries(cuda_quant_cli PRIVATE low_precision_cuda low_precision_io low_precision_reference)
set_target_properties(cuda_quant_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_test(NAME cpu_reference COMMAND test_cpu)
add_test(NAME precision_types_cpu COMMAND test_precision_types)
add_test(NAME io_round_trip COMMAND test_io)
add_test(NAME cli_round_trip
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh $<TARGET_FILE:quant_cli>)
add_test(NAME cuda_cli_round_trip
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_cli.sh
                 $<TARGET_FILE:cuda_quant_cli> $<TARGET_FILE:quant_cli>)
set_tests_properties(cuda_cli_round_trip PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME cuda_reference COMMAND test_cuda)
add_test(NAME benchmark_output
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark_output.sh $<TARGET_FILE:benchmark>)
add_test(NAME cli_benchmark_smoke
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/apps/benchmark_cli.sh $<TARGET_FILE:quant_cli> 32 33)
