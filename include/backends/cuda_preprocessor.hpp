#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

#include <cstdint>

/**
 * NVIDIA CUDA 预处理器
 * 使用 CUDA 进行 NV12 到 RGB 的转换和缩放
 */
class CudaPreprocessor : public IPreprocessorBackend {
 public:
  CudaPreprocessor();
  ~CudaPreprocessor() override;

  RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight) override;

  std::string name() const override { return "NVIDIA CUDA"; }

  /** 设置 GPU 设备 ID */
  void setGpuId(int gpu_id);

 private:
  int gpu_id_ = 0;
};
