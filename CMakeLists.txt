cmake_minimum_required(VERSION 3.16)
project(video_pipeline_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# 平台选择 (编译时指定)
# =============================================================================
# 使用方法:
#   - Rockchip 平台：cmake -DPLATFORM=rockchip ..
#   - NVIDIA 平台：  cmake -DPLATFORM=nvidia ..
#   - 自动检测：    cmake .. (默认)
# =============================================================================

set(PLATFORM "auto" CACHE STRING "Target platform: auto, rockchip, nvidia")
set_property(CACHE PLATFORM PROPERTY STRINGS auto rockchip nvidia)

if(PLATFORM STREQUAL "rockchip")
    set(ROCKCHIP_PLATFORM ON)
    add_definitions(-DROCKCHIP_PLATFORM)
    message(STATUS "Building for Rockchip platform (MPP + RGA + RKNN)")
elseif(PLATFORM STREQUAL "nvidia")
    set(NVIDIA_PLATFORM ON)
    add_definitions(-DNVIDIA_PLATFORM)
    message(STATUS "Building for NVIDIA platform (NVDEC + CUDA + TensorRT)")
else()
    message(STATUS "Building for auto-detect platform")
endif()

# =============================================================================
# 后端选择 (可选，用于裁剪不需要的后端)
# =============================================================================

option(ENABLE_MPP_DECODER "Enable Rockchip MPP decoder" ON)
option(ENABLE_NVDEC_DECODER "Enable NVIDIA NVDEC decoder" ON)
option(ENABLE_RGA_PREPROC "Enable Rockchip RGA preprocessor" ON)
option(ENABLE_CUDA_PREPROC "Enable NVIDIA CUDA preprocessor" ON)
option(ENABLE_RKNN_INFER "Enable Rockchip RKNN inference" ON)
option(ENABLE_TRT_INFER "Enable NVIDIA TensorRT inference" ON)

# 根据平台设置默认后端
if(ROCKCHIP_PLATFORM)
    set(ENABLE_MPP_DECODER ON)
    set(ENABLE_RGA_PREPROC ON)
    set(ENABLE_RKNN_INFER ON)
    set(ENABLE_NVDEC_DECODER OFF)
    set(ENABLE_CUDA_PREPROC OFF)
    set(ENABLE_TRT_INFER OFF)
elseif(NVIDIA_PLATFORM)
    set(ENABLE_NVDEC_DECODER ON)
    set(ENABLE_CUDA_PREPROC ON)
    set(ENABLE_TRT_INFER ON)
    set(ENABLE_MPP_DECODER OFF)
    set(ENABLE_RGA_PREPROC OFF)
    set(ENABLE_RKNN_INFER OFF)
endif()

# =============================================================================
# 依赖查找
# =============================================================================

# FFmpeg (必需，用于解复用)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
)

# OpenCV (可选，用于可视化)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "OpenCV not found, visualization disabled")
endif()

# Rockchip MPP
if(ENABLE_MPP_DECODER OR ROCKCHIP_PLATFORM)
    find_path(MPP_INCLUDE_DIR rk_mpi.h
        PATHS /usr/include/rockchip /usr/local/include/rockchip /usr/include /usr/local/include
    )
    find_library(MPP_LIBRARY NAMES rockchip_mpp mpp
        PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/lib
    )
    if(MPP_INCLUDE_DIR AND MPP_LIBRARY)
        message(STATUS "MPP found: ${MPP_INCLUDE_DIR}")
    elseif(ROCKCHIP_PLATFORM)
        message(FATAL_ERROR "Rockchip MPP not found but PLATFORM=rockchip")
    endif()
endif()

# Rockchip RGA
if(ENABLE_RGA_PREPROC OR ROCKCHIP_PLATFORM)
    find_path(RGA_INCLUDE_DIR im2d.h
        PATHS /usr/include/rga /usr/local/include/rga /usr/include /usr/local/include
    )
    find_library(RGA_LIBRARY NAMES rga
        PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/lib
    )
    if(RGA_INCLUDE_DIR AND RGA_LIBRARY)
        message(STATUS "RGA found: ${RGA_INCLUDE_DIR}")
    elseif(ROCKCHIP_PLATFORM)
        message(FATAL_ERROR "Rockchip RGA not found but PLATFORM=rockchip")
    endif()
endif()

# RKNN
if(ENABLE_RKNN_INFER OR ROCKCHIP_PLATFORM)
    set(PROJECT_RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/include)
    set(PROJECT_RKNN_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/lib)

    find_path(RKNN_INCLUDE_DIR rknn_api.h
        PATHS ${PROJECT_RKNN_INCLUDE_DIR} /usr/include /usr/local/include /usr/include/rknn
    )
    find_library(RKNN_LIBRARY NAMES rknnrt
        PATHS ${PROJECT_RKNN_LIBRARY_DIR} /usr/lib /usr/lib/aarch64-linux-gnu
    )
    if(RKNN_INCLUDE_DIR AND RKNN_LIBRARY)
        message(STATUS "RKNN found: ${RKNN_INCLUDE_DIR}")
    elseif(ROCKCHIP_PLATFORM)
        message(FATAL_ERROR "RKNN not found but PLATFORM=rockchip")
    endif()
endif()

# NVIDIA CUDA
if(ENABLE_CUDA_PREPROC OR ENABLE_NVDEC_DECODER OR NVIDIA_PLATFORM)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_TARGET_DIR}")
    elseif(NVIDIA_PLATFORM)
        message(FATAL_ERROR "CUDA not found but PLATFORM=nvidia")
    endif()
endif()

# TensorRT
if(ENABLE_TRT_INFER)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS /usr/include/x86_64-linux-gnu /usr/local/cuda/include /usr/include
    )
    find_library(TENSORRT_LIBRARY nvinfer
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/lib
    )
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
    endif()
endif()

# =============================================================================
# 源文件列表
# =============================================================================

set(COMMON_SOURCES
    src/main.cpp
    src/ffmpeg_packet_source.cpp
)

set(DECODER_SOURCES "")
set(PREPROC_SOURCES "")
set(INFER_SOURCES "")
set(POSTPROC_SOURCES "")
set(VISUAL_SOURCES "")
set(ENCODER_SOURCES "")

# MPP 解码器
if(ENABLE_MPP_DECODER)
    if(NOT MPP_INCLUDE_DIR OR NOT MPP_LIBRARY)
        message(WARNING "MPP decoder requested but not found, disabling")
    else()
        list(APPEND DECODER_SOURCES
            src/backends/mpp_decoder.cpp
            src/backends/decoder_factory.cpp
        )
        add_compile_definitions(ENABLE_MPP_DECODER)
    endif()
endif()

# NVDEC 解码器
if(ENABLE_NVDEC_DECODER)
    list(APPEND DECODER_SOURCES
        src/backends/nvdec_decoder.cpp
        src/backends/decoder_factory.cpp
    )
    add_compile_definitions(ENABLE_NVDEC_DECODER)
endif()

# RGA 预处理
if(ENABLE_RGA_PREPROC)
    if(NOT RGA_INCLUDE_DIR OR NOT RGA_LIBRARY)
        message(WARNING "RGA preprocessor requested but not found, disabling")
    else()
        list(APPEND PREPROC_SOURCES
            src/backends/rga_preprocessor.cpp
            src/backends/preproc_factory.cpp
        )
        add_compile_definitions(ENABLE_RGA_PREPROC)
    endif()
endif()

# CUDA 预处理
if(ENABLE_CUDA_PREPROC)
    if(CUDAToolkit_FOUND)
        list(APPEND PREPROC_SOURCES
            src/backends/cuda_preprocessor.cpp
            src/backends/preproc_factory.cpp
        )
        add_compile_definitions(ENABLE_CUDA_PREPROC)
    else()
        message(WARNING "CUDA preprocessor requested but CUDA not found, disabling")
    endif()
endif()

# 推理后端
if(ENABLE_RKNN_INFER)
    if(RKNN_INCLUDE_DIR AND RKNN_LIBRARY)
        list(APPEND INFER_SOURCES
            src/backends/rknn_infer.cpp
            src/backends/infer_factory.cpp
        )
        add_compile_definitions(ENABLE_RKNN_INFER)
    else()
        message(WARNING "RKNN inference requested but not found, disabling")
    endif()
endif()

if(ENABLE_TRT_INFER)
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        list(APPEND INFER_SOURCES
            src/backends/trt_infer.cpp
            src/backends/infer_factory.cpp
        )
        add_compile_definitions(ENABLE_TRT_INFER)
    else()
        message(WARNING "TensorRT inference requested but not found, disabling")
    endif()
endif()

# 后处理 (YOLO 后处理)
list(APPEND POSTPROC_SOURCES
    src/backends/yolo_postproc.cpp
    src/backends/postproc_factory.cpp
)

# 编码器 (NVENC)
if(ENABLE_NVDEC_DECODER)
    list(APPEND ENCODER_SOURCES
        src/backends/nvenc_encoder.cpp
        src/backends/encoder_factory.cpp
    )
    add_compile_definitions(ENABLE_NVENC_ENCODER)
endif()

# 可视化 (OpenCV)
if(OpenCV_FOUND)
    list(APPEND VISUAL_SOURCES
        src/backends/opencv_visualizer.cpp
    )
    add_compile_definitions(ENABLE_VISUAL)
endif()

# =============================================================================
# 可执行文件
# =============================================================================

add_executable(video_pipeline
    ${COMMON_SOURCES}
    ${DECODER_SOURCES}
    ${PREPROC_SOURCES}
    ${INFER_SOURCES}
    ${POSTPROC_SOURCES}
    ${ENCODER_SOURCES}
    ${VISUAL_SOURCES}
)

target_include_directories(rk_video_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FFMPEG_INCLUDE_DIRS}
)

if(MPP_INCLUDE_DIR)
    target_include_directories(video_pipeline PRIVATE ${MPP_INCLUDE_DIR})
endif()

if(RGA_INCLUDE_DIR)
    target_include_directories(video_pipeline PRIVATE ${RGA_INCLUDE_DIR})
endif()

if(RKNN_INCLUDE_DIR)
    target_include_directories(video_pipeline PRIVATE ${RKNN_INCLUDE_DIR})
endif()

if(CUDAToolkit_FOUND)
    target_include_directories(video_pipeline PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(video_pipeline PRIVATE
    PkgConfig::FFMPEG
)

if(OpenCV_FOUND)
    target_link_libraries(video_pipeline PRIVATE ${OpenCV_LIBS})
    target_include_directories(video_pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(MPP_LIBRARY)
    target_link_libraries(video_pipeline PRIVATE ${MPP_LIBRARY})
endif()

if(RGA_LIBRARY)
    target_link_libraries(video_pipeline PRIVATE ${RGA_LIBRARY})
endif()

if(RKNN_LIBRARY)
    target_link_libraries(video_pipeline PRIVATE ${RKNN_LIBRARY})
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(video_pipeline PRIVATE CUDA::cudart)
endif()

# RPATH 设置
set_target_properties(video_pipeline PROPERTIES
    BUILD_RPATH "${PROJECT_RKNN_LIBRARY_DIR}"
    INSTALL_RPATH "${PROJECT_RKNN_LIBRARY_DIR}"
)

# =============================================================================
# 编译配置摘要
# =============================================================================

message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Platform: ${PLATFORM}")
message(STATUS "MPP Decoder: ${ENABLE_MPP_DECODER}")
message(STATUS "NVDEC Decoder: ${ENABLE_NVDEC_DECODER}")
message(STATUS "RGA Preproc: ${ENABLE_RGA_PREPROC}")
message(STATUS "CUDA Preproc: ${ENABLE_CUDA_PREPROC}")
message(STATUS "RKNN Infer: ${ENABLE_RKNN_INFER}")
message(STATUS "TensorRT Infer: ${ENABLE_TRT_INFER}")
message(STATUS "Postproc: ON")
message(STATUS "NVENC Encoder: ${ENABLE_NVENC_ENCODER}")
message(STATUS "===========================")
message(STATUS "")
