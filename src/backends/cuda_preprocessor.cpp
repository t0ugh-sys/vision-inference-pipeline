#include "backends/cuda_preprocessor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace {

void checkCudaStatus(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

inline std::uint8_t clampToByte(float value) {
  if (value < 0.0f) {
    return 0;
  }
  if (value > 255.0f) {
    return 255;
  }
  return static_cast<std::uint8_t>(value);
}

}  // namespace

CudaPreprocessor::CudaPreprocessor() = default;

CudaPreprocessor::~CudaPreprocessor() = default;

void CudaPreprocessor::setGpuId(int gpu_id) {
  gpu_id_ = gpu_id;
}

RgbImage CudaPreprocessor::convertAndResize(
    const DecodedFrame& frame,
    int outputWidth,
    int outputHeight) {
  checkCudaStatus(cudaSetDevice(gpu_id_), "Failed to set CUDA device");

  if (frame.width <= 0 || frame.height <= 0) {
    throw std::runtime_error("CUDA preprocessor received an invalid frame size");
  }
  if (frame.yData.empty() || frame.uvData.empty()) {
    throw std::runtime_error("CUDA preprocessor requires NV12 frame data");
  }
  if (frame.horizontalStride < frame.width) {
    throw std::runtime_error("CUDA preprocessor received an invalid horizontal stride");
  }

  RgbImage output;
  output.width = outputWidth;
  output.height = outputHeight;
  output.data.resize(static_cast<std::size_t>(outputWidth * outputHeight * 3));

  const int srcWidth = frame.width;
  const int srcHeight = frame.height;
  const int srcStride = frame.horizontalStride;
  const int uvStride = srcStride;

  for (int y = 0; y < outputHeight; ++y) {
    for (int x = 0; x < outputWidth; ++x) {
      const float srcX = static_cast<float>(x) * static_cast<float>(srcWidth) / static_cast<float>(outputWidth);
      const float srcY = static_cast<float>(y) * static_cast<float>(srcHeight) / static_cast<float>(outputHeight);

      const int x0 = std::clamp(static_cast<int>(srcX), 0, srcWidth - 1);
      const int y0 = std::clamp(static_cast<int>(srcY), 0, srcHeight - 1);

      const float Y = static_cast<float>(frame.yData[static_cast<std::size_t>(y0 * srcStride + x0)]);
      const int uvX = (x0 / 2) * 2;
      const int uvY = y0 / 2;
      const std::size_t uvIndex = static_cast<std::size_t>(uvY * uvStride + uvX);
      if (uvIndex + 1 >= frame.uvData.size()) {
        throw std::runtime_error("CUDA preprocessor received truncated UV data");
      }

      const float U = static_cast<float>(frame.uvData[uvIndex]) - 128.0f;
      const float V = static_cast<float>(frame.uvData[uvIndex + 1]) - 128.0f;
      const float C = std::max(0.0f, Y - 16.0f);

      const std::size_t outputIndex = static_cast<std::size_t>((y * outputWidth + x) * 3);
      output.data[outputIndex + 0] = clampToByte(1.164f * C + 1.596f * V);
      output.data[outputIndex + 1] = clampToByte(1.164f * C - 0.392f * U - 0.813f * V);
      output.data[outputIndex + 2] = clampToByte(1.164f * C + 2.017f * U);
    }
  }

  return output;
}
