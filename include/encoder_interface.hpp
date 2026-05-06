#pragma once

#include "pipeline_types.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct EncoderConfig {
  std::string outputPath;
  int width = 0;
  int height = 0;
  int horStride = 0;
  int verStride = 0;
  PixelFormat inputFormat = PixelFormat::kUnknown;
  int fps = 0;
  int fpsNum = 0;
  int fpsDen = 1;
  int bitrate = 0;
  std::string codec = "h264";
  int lowLatency = -1;
};

class IEncoderBackend {
 public:
  virtual ~IEncoderBackend() = default;

  virtual void init(const EncoderConfig& config) = 0;
  virtual void encode(const RgbImage& frame, int64_t pts) = 0;
  virtual bool supportsDecodedFrameInput() const { return false; }
  virtual void encodeDecodedFrame(const DecodedFrame& frame, int64_t pts) {
    (void)frame;
    (void)pts;
    throw std::runtime_error("Decoded-frame input is not supported by this encoder backend");
  }
  virtual void flush() = 0;
  virtual void close() = 0;
  virtual std::string name() const = 0;
};

enum class EncoderBackendType {
  kAuto,
  kNvidiaNvEnc,
  kRockchipMpp,
  kCpu,
};

EncoderBackendType detectAvailableEncoderBackend();

std::unique_ptr<IEncoderBackend> createEncoderBackend(EncoderBackendType type = EncoderBackendType::kAuto);
