#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class VideoCodec {
  kUnknown,
  kH264,
  kH265,
};

struct EncodedPacket {
  std::vector<std::uint8_t> data;
  std::int64_t pts = 0;
  bool keyFrame = false;
  bool endOfStream = false;
};

struct DecodedFrame {
  int width = 0;
  int height = 0;
  int horizontalStride = 0;
  int verticalStride = 0;
  int chromaStride = 0;
  int dmaFd = -1;
  std::int64_t pts = 0;
  bool isOnDevice = false;
  std::uintptr_t deviceY = 0;
  std::uintptr_t deviceUv = 0;
  std::shared_ptr<void> nativeHandle;
  // Y plane data (NV12 Y)
  std::vector<std::uint8_t> yData;
  // UV plane data (NV12 UV, interleaved)
  std::vector<std::uint8_t> uvData;
};

struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> data;
};

struct InputSourceConfig {
  std::string uri;
};

struct ModelConfig {
  std::string modelPath;
  int inputWidth = 640;
  int inputHeight = 640;
};
