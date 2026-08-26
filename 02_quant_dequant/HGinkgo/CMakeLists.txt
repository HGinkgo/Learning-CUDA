cmake_minimum_required(VERSION 3.18)

set(LP_BACKEND "nvidia" CACHE STRING "GPU backend: nvidia or iluvatar")
set_property(CACHE LP_BACKEND PROPERTY STRINGS nvidia iluvatar)

if(LP_BACKEND STREQUAL "iluvatar")
    if(DEFINED ENV{COREX_VERSION} AND NOT "$ENV{COREX_VERSION}" STREQUAL "")
        set(_corex_version "$ENV{COREX_VERSION}")
    else()
        set(_corex_version "4.4.0")
    endif()
    set(ILUVATAR_SDK_ROOT "/usr/local/corex-${_corex_version}" CACHE PATH
        "Iluvatar CoreX SDK root")
    if(NOT EXISTS "${ILUVATAR_SDK_ROOT}/bin/clang++")
        message(FATAL_ERROR "Iluvatar CoreX compiler not found under ${ILUVATAR_SDK_ROOT}")
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "${ILUVATAR_SDK_ROOT}/bin/clang++" CACHE FILEPATH
            "Iluvatar host compiler")
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${ILUVATAR_SDK_ROOT}/bin/clang++" CACHE FILEPATH
            "Iluvatar device compiler")
    endif()
endif()

project(low_precision_reference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(LP_BACKEND STREQUAL "iluvatar")
        set(CMAKE_CUDA_ARCHITECTURES ivcore11)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75;86)
    endif()
endif()

if(NOT LP_BACKEND STREQUAL "nvidia" AND NOT LP_BACKEND STREQUAL "iluvatar")
    message(FATAL_ERROR "Unsupported LP_BACKEND='${LP_BACKEND}'")
endif()

set(LP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LP_NVIDIA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backends/nvidia/include)
set(LP_CUDA_SOURCE backends/nvidia/cuda_quant.cu)
if(LP_BACKEND STREQUAL "iluvatar")
    set(LP_CUDA_SOURCE backends/iluvatar/cuda_quant.cu)
endif()

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

add_library(low_precision_cuda ${LP_CUDA_SOURCE})
target_include_directories(low_precision_cuda PUBLIC ${LP_INCLUDE_DIR} ${LP_NVIDIA_INCLUDE_DIR})
target_link_libraries(low_precision_cuda PRIVATE low_precision_reference)
if(LP_BACKEND STREQUAL "iluvatar")
    target_link_directories(low_precision_cuda PUBLIC "${ILUVATAR_SDK_ROOT}/lib64")
    target_link_libraries(low_precision_cuda PUBLIC "${ILUVATAR_SDK_ROOT}/lib64/libcudart.so")
endif()
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
