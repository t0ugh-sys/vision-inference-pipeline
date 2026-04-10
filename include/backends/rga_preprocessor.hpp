#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

/**
 * Rockchip RGA 硬件预处理器
 * 支持 NV12/BGR/RGB 格式转换 + 缩放
 */
class RgaPreprocessor : public IPreprocessorBackend {
 public:
  RgaPreprocessor();
  ~RgaPreprocessor() override;

  RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight) override;

  std::string name() const override { return "Rockchip RGA"; }
};
