#pragma once

#include "encoder_interface.hpp"

/**
 * NVIDIA NVENC 硬件编码器
 * 使用 FFmpeg 的 NVENC API
 */
class NvencEncoder : public IEncoderBackend {
 public:
  NvencEncoder();
  ~NvencEncoder() override;

  void init(const EncoderConfig& config) override;
  void encode(const RgbImage& frame, int64_t pts) override;
  void flush() override;
  void close() override;

  std::string name() const override { return "NVENC"; }

 private:
  void checkAvStatus(int status, const char* message);

  AVCodecContext* codecCtx_ = nullptr;
  AVFrame* swFrame_ = nullptr;
  AVFrame* hwFrame_ = nullptr;
  AVBufferRef* hwDeviceCtx_ = nullptr;
  FILE* outputFile_ = nullptr;
  bool initialized_ = false;
};
