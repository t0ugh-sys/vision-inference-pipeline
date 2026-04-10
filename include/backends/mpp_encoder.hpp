#pragma once

#include "../encoder_interface.hpp"

#include <fstream>

/**
 * Rockchip MPP 硬件编码器
 */
class MppEncoder : public IEncoderBackend {
 public:
  MppEncoder();
  ~MppEncoder() override;

  void init(const EncoderConfig& config) override;
  void encode(const RgbImage& frame, int64_t pts) override;
  void flush() override;
  void close() override;

  std::string name() const override { return "MPP Encoder"; }

 private:
  void* context_ = nullptr;
  void* api_ = nullptr;
  void* encoder_ = nullptr;
  std::ofstream outputFile_;
  bool initialized_ = false;
};