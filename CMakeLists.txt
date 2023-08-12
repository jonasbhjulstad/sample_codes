cmake_minimum_required(VERSION 3.27)

project(Sample_Codes)

set(LOG_DIRECTORY ${PROJECT_BINARY_DIR})

set(SYCL_CUDA_OPT -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64)
function(add_sample_executable src_name)
    add_executable(${src_name} ${src_name}.cpp)
    target_compile_options(${src_name} PUBLIC -fsycl ${SYCL_CUDA_OPT} -Wno-deprecated-declarations -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -Wno-unknown-cuda-version)
    target_link_options(${src_name} PUBLIC -fsycl ${SYCL_CUDA_OPT} -Wno-unknown-cuda-version)
    target_compile_definitions(${src_name} PUBLIC -DLOG_DIRECTORY="${LOG_DIRECTORY}")
endfunction()
find_package(IntelSYCL REQUIRED)

add_sample_executable(separate)
add_sample_executable(nd_range)
