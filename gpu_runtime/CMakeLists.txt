cmake_minimum_required(VERSION 3.24)
project(triple_pendulum_cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# RTX 50-series/Blackwell. Override this at configure time if needed.
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(triple_pendulum_cuda triple_pendulum_cuda.cu)
set_target_properties(triple_pendulum_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
