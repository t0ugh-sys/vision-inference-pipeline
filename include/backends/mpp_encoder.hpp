#pragma once

#include "../encoder_interface.hpp"

extern "C" {
#include <rk_type.h>
#include <rk_venc_cfg.h>
}

#include <fstream>

struct MppApi_t;
typedef struct MppApi_t MppApi;

class MppEncoder : public IEncoderBackend {
 public:
  MppEncoder();
  ~MppEncoder() override;

  void init(const EncoderConfig& config) override;
  void encode(const RgbImage& frame, int64_t pts) override;
  bool supportsDecodedFrameInput() const override { return true; }
  void encodeDecodedFrame(const DecodedFrame& frame, int64_t pts) override;
  void flush() override;
  void close() override;

  std::string name() const override { return "MPP Encoder"; }

 private:
  void writePacket(void* packet);
  void drainPackets(bool untilEos);

  MppCtx context_ = nullptr;
  MppApi* api_ = nullptr;
  MppEncCfg config_ = nullptr;
  MppBufferGroup bufferGroup_ = nullptr;
  MppBuffer packetBuffer_ = nullptr;
  std::ofstream outputFile_;
  bool initialized_ = false;
  bool flushSubmitted_ = false;
  int width_ = 0;
  int height_ = 0;
  int horStride_ = 0;
  int verStride_ = 0;
  PixelFormat inputFormat_ = PixelFormat::kUnknown;
};
