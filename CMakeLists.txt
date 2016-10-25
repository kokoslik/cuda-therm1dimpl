project(therm1dimpl)
cmake_minimum_required(VERSION 3.1)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-D_FORCE_INLINES\" ")

find_package(CUDA REQUIRED)
cuda_add_executable(${PROJECT_NAME} therm1dimpl.cu)
